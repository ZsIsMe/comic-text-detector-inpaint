"""Experimental boundary-trend preview; source images and JSON are read-only."""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from bubble_completion import complete_region, draw_dashed_outline
from bubble_neck_split import isolate_balloon, split_mask
from preview_split_centers import (load_image, save_png, group_shared_components,
                                   calculate_from_mask, draw_center)


def symmetry_panel(region, completed, outline, debug, old, new, index, accepted):
    """Separate actual computation shapes so overlap cannot hide asymmetry."""
    image = np.full((*region.shape, 3), 255, np.uint8)
    image[completed > 0] = (239, 248, 241)
    edge = cv2.morphologyEx(completed, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    edge[cv2.dilate(outline, np.ones((5, 5), np.uint8)) > 0] = 0
    image[edge > 0] = (60, 60, 60)
    draw_dashed_outline(image, outline, (180, 50, 200))
    x, y, w, h = cv2.boundingRect(completed)
    if accepted and 'axis' in debug:
        p = np.asarray(debug['axis']['point'])
        d = np.asarray(debug['axis']['direction'])
        d /= np.linalg.norm(d)
        for t in range(-max(w, h), max(w, h), 15):
            a, b = np.rint(p+t*d).astype(int), np.rint(p+(t+6)*d).astype(int)
            cv2.line(image, tuple(a), tuple(b), (175, 175, 175), 1, cv2.LINE_AA)
    cv2.circle(image, tuple(np.rint(old).astype(int)), 4, (100, 100, 100), -1)
    if accepted:
        draw_center(image, new, index)
    crop = image[max(0, y-15):min(image.shape[0], y+h+15),
                 max(0, x-15):min(image.shape[1], x+w+15)]
    panel = cv2.copyMakeBorder(crop, 30, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    label = f'{index+1}: mirror' if debug.get('shape') == 'mirrored_outline' else f'{index+1}: rectangle'
    if not accepted:
        label = f'{index+1}: hold'
    cv2.putText(panel, label, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, .5, (70, 70, 70), 1, cv2.LINE_AA)
    return panel


def save_panels(output, name, panels):
    height = max(panel.shape[0] for panel in panels)
    padded = [cv2.copyMakeBorder(p, 0, height-p.shape[0], 0, 0, cv2.BORDER_CONSTANT,
                                 value=(255, 255, 255)) for p in panels]
    save_png(output/name, np.concatenate(padded, axis=1))


def render_group(core, gray, canvas, simple, body, regions, cuts, items, indices, panels=None):
    records = []
    palette = [(0, 140, 255), (210, 40, 200), (200, 170, 0)]
    for cut in cuts:
        for image in (canvas, simple):
            cv2.line(image, tuple(cut['a']), tuple(cut['b']), (255, 80, 0), 2, cv2.LINE_AA)
    for number, (region, item, index) in enumerate(zip(regions, items, indices)):
        # A group may contain three lobes: use only cuts bordering this lobe.
        relevant = []
        neighborhood = cv2.dilate(region, np.ones((9, 9), np.uint8))
        for cut in cuts:
            line = np.zeros_like(region)
            cv2.line(line, tuple(cut['a']), tuple(cut['b']), 255, 1)
            if np.count_nonzero((line > 0) & (neighborhood > 0)) > .6*np.count_nonzero(line):
                relevant.append(cut)
        completed, outline, debug = complete_region(region, body, relevant)
        before = calculate_from_mask(core, gray, item, region, 'auto', smooth_radius=10)
        after = calculate_from_mask(core, gray, item, completed, 'auto', smooth_radius=10)
        old, new = np.array(before['center']), np.array(after['center'])
        ox, oy = np.rint(old).astype(int)
        nx, ny = np.rint(new).astype(int)
        inside = 0 <= ny < region.shape[0] and 0 <= nx < region.shape[1] and region[ny, nx] > 0
        record = {'index': index+1, **debug, 'before': before, 'after': after,
                  'displacement': (new-old).tolist(), 'inside_layout_region': bool(inside)}
        accepted = debug['status'] in ('completed', 'unchanged') and inside
        record['completion_accepted'] = bool(accepted)
        draw_dashed_outline(canvas, outline, palette[number % len(palette)])
        cv2.circle(canvas, (ox, oy), 5, (110, 110, 110), -1, cv2.LINE_AA)
        if np.linalg.norm(new-old) >= 3:
            cv2.arrowedLine(canvas, (ox, oy), (nx, ny), (60, 150, 40), 1, cv2.LINE_AA, tipLength=.3)
        if accepted:
            draw_center(canvas, new, index)
            draw_center(simple, new, index)
        elif inside:
            for image in (canvas, simple):
                cv2.circle(image, (ox, oy), 7, (110, 110, 110), 2, cv2.LINE_AA)
                cv2.putText(image, f'{index+1} hold', (ox+10, oy-9),
                            cv2.FONT_HERSHEY_SIMPLEX, .45, (90, 90, 90), 1, cv2.LINE_AA)
        else:
            # Keep out-of-partition candidates visibly distinct, never accepted green.
            cv2.drawMarker(canvas, (nx, ny), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 16, 2)
        if panels is not None:
            panels.append(symmetry_panel(region, completed, outline, debug, old, new, index, accepted))
        records.append(record)
    return records


def process_page(core, root, name, items, output):
    stem = Path(name).stem
    original = load_image(root/name, False)
    gray = core.to_gray(load_image(root/'inpainted'/(stem+'.png'), False))
    canvas, simple = original.copy(), original.copy()
    seeds = [core.seed_from_item(item, gray.shape[1], gray.shape[0]) for item in items]
    selections = [core.get_best_component_mask(gray, item) for item in items]
    masks = [best[0] if best is not None else None for best in selections]
    report = []
    for indices in group_shared_components(masks):
        if len(indices) < 2 or masks[indices[0]] is None:
            continue
        mask = masks[indices[0]]
        group_seeds = [seeds[i] for i in indices]
        body, cleanup = isolate_balloon(gray, mask, group_seeds)
        if body is None:
            report.append({'items': indices, 'status': 'isolation_failed', 'debug': cleanup})
            continue
        regions, cuts, geometry = split_mask(body, group_seeds, mask, core.to_gray(original))
        if regions is None:
            report.append({'items': indices, 'status': 'split_failed', 'debug': geometry})
            continue
        panels = []
        result = render_group(core, gray, canvas, simple, body, regions, cuts,
                              [items[i] for i in indices], indices, panels)
        save_panels(output, stem+'_shapes_'+'_'.join(str(i+1) for i in indices)+'.png', panels)
        report.append({'items': [i+1 for i in indices], 'cuts': cuts, 'centers': result})
    save_png(output/(stem+'_trends.png'), canvas)
    save_png(output/(stem+'_new_centers.png'), simple)
    roi = {'12-19': (230, 780, 600, 1150), '12-20': (570, 35, 1030, 420)}.get(stem)
    if roi:
        x1, y1, x2, y2 = roi
        save_png(output/(stem+'_trends_detail.png'), canvas[y1:y2, x1:x2])
    return report


def process_crop(core, path, output):
    """Inspect the user's rectangular example, using its enclosed white region.

    Close text-sized holes, but do not paint the source image. Fixed sample
    seeds are explicit for this diagnostic crop, not a general text detector.
    """
    original = load_image(path, False)
    gray = core.to_gray(original)
    if gray.shape != (546, 566):
        raise ValueError('This sample uses explicit seeds for the supplied 566 x 546 crop only')
    white = (gray > 190).astype(np.uint8)*255
    contours, hierarchy = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    body = np.zeros_like(gray)
    # Only an enclosed white component (not the outer page/background).
    eligible = [c for c in contours if all((cv2.boundingRect(c)[0] > 0,
                cv2.boundingRect(c)[1] > 0,
                sum(cv2.boundingRect(c)[::2]) < gray.shape[1],
                sum(cv2.boundingRect(c)[1::2]) < gray.shape[0]))]
    if not eligible:
        raise ValueError('No enclosed balloon in supplied crop')
    cv2.drawContours(body, [max(eligible, key=cv2.contourArea)], -1, 255, -1)
    seeds = [(340, 180), (170, 310)]
    regions, cuts, debug = split_mask(body, seeds, body, gray)
    if regions is None:
        raise ValueError(f'Rectangular sample could not split: {debug["reason"]}')
    items = [{'xyxy_pixel': [x-30, y-50, x+30, y+50]} for x, y in seeds]
    canvas, simple = original.copy(), original.copy()
    panels = []
    records = render_group(core, gray, canvas, simple, body, regions, cuts, items, [0, 1], panels)
    save_panels(output, 'rectangle_shapes.png', panels)
    save_png(output/'rectangle_trends.png', canvas)
    save_png(output/'rectangle_new_centers.png', simple)
    return {'seed_source': 'explicit diagnostic sample points', 'cuts': cuts, 'centers': records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('json_path', type=Path)
    parser.add_argument('--module-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--pages', nargs='+', default=['12-19', '12-20'])
    parser.add_argument('--all-pages', action='store_true', help='Preview every page in the source JSON')
    parser.add_argument('--rectangle-crop', type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.module_dir))
    import layout_core as core
    data = json.loads(args.json_path.read_text(encoding='utf-8'))['transMap']
    report = {'experimental': True, 'method': 'visible_outline_symmetry_v2', 'pages': {}}
    for name, items in data.items():
        if not args.all_pages and Path(name).stem not in args.pages:
            continue
        report['pages'][name] = process_page(core, args.json_path.parent, name, items, args.output_dir)
        print(name, 'done', flush=True)
    if args.rectangle_crop:
        report['rectangle_crop'] = process_crop(core, args.rectangle_crop, args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'completion_report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
