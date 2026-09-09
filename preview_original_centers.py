"""No-model balloon splitting from inpainted pixels; cuts and centers only."""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from bubble_neck_split import isolate_balloon, split_mask
from preview_split_centers import (load_image, save_png, group_shared_components,
                                    calculate_from_mask, draw_center)


def process_page(core, root, name, items, output, center_mode='auto'):
    stem = Path(name).stem
    clean = load_image(root/'inpainted'/(stem+'.png'),False)
    original = load_image(root/name,False)
    gray = core.to_gray(clean)
    seeds = [core.seed_from_item(item,gray.shape[1],gray.shape[0]) for item in items]
    selections = [core.get_best_component_mask(gray,item) for item in items]
    masks = [best[0] if best is not None else None for best in selections]
    groups = group_shared_components(masks)
    final_regions = {}
    records = []
    cuts = []
    for indices in groups:
        mask = masks[indices[0]]
        debug = {'items':[i+1 for i in indices]}
        if mask is None:
            records.append({**debug,'reason':'no_component'})
            continue
        group_seeds = [seeds[i] for i in indices]
        body, cleanup = isolate_balloon(gray,mask,group_seeds)
        debug['cleanup'] = cleanup
        if body is None:
            records.append({**debug,'reason':cleanup['reason']})
            continue
        regions, group_cuts, geometry = split_mask(body,group_seeds,raw_mask=mask,
                                                  original_gray=core.to_gray(original))
        debug.update(geometry)
        records.append(debug)
        if regions is None:
            continue
        reconstructed = body.copy()
        for cut in group_cuts:
            cv2.line(reconstructed,tuple(cut['a']),tuple(cut['b']),0,3)
        _, labels = cv2.connectedComponents(reconstructed,connectivity=4)
        for index,region in zip(indices,regions):
            x,y=np.rint(seeds[index]).astype(int)
            assert np.array_equal(region>0, labels==labels[y,x])
            assert not np.any((region>0)&(mask==0))
            final_regions[index]=region
        assert np.all(np.sum(np.stack(regions)>0,axis=0)<=1)
        cuts.extend({**cut,'items':[i+1 for i in indices]} for cut in group_cuts)
    for cut in cuts:
        for preview in (original,clean):
            cv2.line(preview,tuple(cut['a']),tuple(cut['b']),(255,80,0),3,cv2.LINE_AA)
    centers=[]
    for index,item in enumerate(items):
        record={'index':index+1,'input_center':seeds[index]}
        try:
            baseline=core.calculate_layout(gray,item,center_mode)
            record['unsplit_center']=baseline['layout_debug']['new_center_pixel']
        except ValueError:
            record['unsplit_center']=None
        region=final_regions.get(index)
        if region is None:
            record['status']='unresolved'
        else:
            result=calculate_from_mask(core,gray,item,region,center_mode,smooth_radius=10)
            center=result['center']
            x,y=np.rint(center).astype(int)
            inside=bool(0<=y<gray.shape[0] and 0<=x<gray.shape[1] and region[y,x])
            record.update(result)
            record['center_inside_region']=inside
            record['status']='calculated' if inside else 'candidate_outside_region'
            if inside:
                for preview in (original,clean):
                    draw_center(preview,center,index)
        centers.append(record)
    save_png(output/(stem+'_split_centers.png'),clean)
    save_png(output/(stem+'_on_original.png'),original)
    if stem in ('12-19','12-20'):
        roi=(230,780,600,1150) if stem=='12-19' else (570,35,1030,420)
        x1,y1,x2,y2=roi
        save_png(output/(stem+'_detail.png'),original[y1:y2,x1:x2])
    return {'method':'original_pixels_concavity','cuts':cuts,'groups':records,'items':centers}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('json_path',type=Path)
    parser.add_argument('--module-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--pages',nargs='*',default=[])
    parser.add_argument('--center-mode',choices=['auto','outer','inner','average'],default='auto')
    args=parser.parse_args()
    sys.path.insert(0,str(args.module_dir))
    import layout_core as core
    data=json.loads(args.json_path.read_text(encoding='utf-8'))['transMap']
    args.output_dir.mkdir(parents=True,exist_ok=True)
    report_name = ('report_'+ '_'.join(Path(p).stem for p in args.pages)+'.json'
                   if args.pages else 'report.json')
    report={}
    for name,items in data.items():
        if args.pages and Path(name).stem not in args.pages:
            continue
        result=process_page(core,args.json_path.parent,name,items,args.output_dir,args.center_mode)
        report[name]=result
        print(name,len(result['cuts']),'cuts;',sum(i['status']!='calculated' for i in result['items']),'unresolved',flush=True)
        (args.output_dir/report_name).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
