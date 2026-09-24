"""Copy visually reviewed suspect previews; never move or alter the originals."""
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from preview_split_centers import load_image, save_png


SOURCE = Path('artifacts/completion_symmetry_all_pages')
OUTPUT = Path('artifacts/completion_symmetry_review')
# Manual inspection of all 49 group-shape previews, not an automatic classifier.
SELECTION = [
    ('12-01', [4, 5], '輕微', '4 號下端留下小凹口'),
    ('12-02', [4, 5], '明顯', '4 號下端出現深 V 形凹口'),
    ('12-04', [2, 3], '明顯', '2 號下端出現 V 形折返'),
    ('12-04', [5, 6], '明顯', '5 號下端補邊有尖角'),
    ('12-05', [5, 6], '明顯', '5 號上側原有凹陷及補邊不平順，需核對輪廓來源'),
    ('12-06', [3, 4], '輕微', '3 號下端接合有小凹口'),
    ('12-09', [3, 4], '使用者指出', '與使用者截圖一致；3 號橙色補邊在交界內形成 V 形折返'),
    ('12-10', [7, 8], '明顯', '7 號上側仍有明顯凹陷，需確認是否應屬於氣泡主體'),
    ('12-10', [9, 10], '明顯', '9 號上下保留／補出局部突起，輪廓不平順'),
    ('12-13', [2, 3], '輕微', '2 號下端接合有小尖角'),
    ('12-14', [4, 5], '明顯', '4 號左側 V 形凹口未消除'),
    ('12-14', [6, 7], '輕微', '6 號下端留下小凹口'),
    ('12-19', [1, 2], '明顯', '1 號下端出現 V 形缺口'),
    ('13-04', [4, 5], '明顯', '4 號下端有深凹口'),
    ('13-06', [5, 6], '明顯', '5 號下端有 V 形凹口'),
    ('13-10', [4, 5], '明顯', '4 號左側補全仍形成尖角／折返'),
    ('13-11', [7, 8], '輕微', '7 號下端接合有小凹口'),
    ('13-19', [1, 2], '輕微', '1 號下端有輕微凹口'),
    ('13-19', [7, 8], '明顯', '8 號上端補邊呈 V 形凹口'),
    ('13-20', [5, 6], '明顯', '5 號下端有深凹口'),
]


def copy_verified(source, target):
    shutil.copy2(source, target)
    assert hashlib.sha256(source.read_bytes()).digest() == hashlib.sha256(target.read_bytes()).digest()


def collect(stem, group, category, severity, reason):
    key = stem+'_items_'+'_'.join(map(str, group['items']))
    destination = OUTPUT/category/key
    destination.mkdir(parents=True, exist_ok=True)
    for suffix in ('_trends.png', '_new_centers.png', '_shapes_'+'_'.join(map(str, group['items']))+'.png'):
        source = SOURCE/(stem+suffix)
        copy_verified(source, destination/source.name)
    preview = load_image(SOURCE/(stem+'_trends.png'), False)
    points = []
    for center in group['centers']:
        for stage in ('before', 'after'):
            r = center[stage]['outer_rect']
            points.extend([(r['left'], r['top']), (r['left']+r['width'], r['top']+r['height'])])
    for cut in group['cuts']:
        points.extend([cut['a'], cut['b']])
    points = np.array(points)
    left, top = np.maximum(0, points.min(axis=0)-40).astype(int)
    right, bottom = np.minimum([preview.shape[1], preview.shape[0]], points.max(axis=0)+40).astype(int)
    save_png(destination/(key+'_detail.png'), preview[top:bottom, left:right])
    record = {'page': stem, 'items': group['items'], 'severity': severity,
              'review_reason': reason, 'source_group': group,
              'folder': str(destination.resolve())}
    (destination/'review.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    return record


def main():
    report = json.loads((SOURCE/'completion_report.json').read_text())
    records = []
    for stem, ids, severity, reason in SELECTION:
        group = next(g for g in report['pages'][stem+'.jpg'] if g['items'] == ids)
        records.append(collect(stem, group, '01_visual_notches', severity, reason))
    for name, groups in report['pages'].items():
        for group in groups:
            held = [c for c in group.get('centers', []) if not c['completion_accepted']]
            if held:
                reason = '；'.join(f"{c['index']} 號：{c['status']}" for c in held)
                records.append(collect(Path(name).stem, group, '02_hold', '程式已保留', reason))
    (OUTPUT/'review_manifest.json').write_text(json.dumps({
        'selection_method': 'manual visual screening of all 49 group previews plus existing hold flags',
        'source_unchanged': True, 'records': records,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print('visual groups:', len(SELECTION), 'hold groups:', len(records)-len(SELECTION))
    print('unique pages:', len({r['page'] for r in records}))
    print('copied PNGs:', len(list(OUTPUT.rglob('*.png'))))


if __name__ == '__main__':
    main()
