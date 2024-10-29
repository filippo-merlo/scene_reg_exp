import json
import os
from os import path as osp
import os.path as osp
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
from utils import get_refcoco_df, read_panoptic_anns, target_category_score
from decimal import Decimal
import argparse
from configuration import Config

tqdm.pandas()

def main(args, config):
    
    # prepare panoptic anns
    refcoco_path = osp.join(config.ref_base, args.dataset)
    p_ann_pth = osp.join(config.panoptic_anns, 'panoptic_train2017.json')
    p_img_dir = osp.join(config.panoptic_anns, 'panoptic_train2017')

    print('reading RefCOCO annotations...')
    refcoco_df = get_refcoco_df(refcoco_path)

    with open(osp.join(refcoco_path, 'instances.json')) as f:
        refcoco_data = json.load(f)
        refcoco_categories = {r['id']: r for r in refcoco_data['categories']}

    panoptic_df, panoptic_categories = read_panoptic_anns(p_ann_pth)

    merged_categories = [
        (panoptic_categories.get(i, dict()).get('name', None), refcoco_categories.get(i, dict()).get('name', None)) 
        for i in refcoco_categories.keys()]
    assert not False in [m==n for m,n in merged_categories]

    for split in ['val', 'testA', 'testB']:
        
        print(f'processing split {split}...')

        if not osp.isdir(args.out_dir):
            os.makedirs(args.out_dir)
        path = osp.join(args.out_dir, f'{args.dataset}-{split}-target-class-in-context.json')

        if not os.path.isfile(path):
            
            _refcoco_df = refcoco_df.copy()
            _refcoco_df = _refcoco_df.reset_index().groupby('ann_id').agg({'sent_id':list, 'caption':list, 'ref_id':'first', 'refcoco_split':'first', 'coco_split':'first', 'image_id':'first', 'bbox':'first', 'category_id':'first'})
            _refcoco_split = _refcoco_df.loc[_refcoco_df.refcoco_split == split]

            kwargs = {'panoptic_categories':panoptic_categories, 'panoptic_df':panoptic_df, 'panoptic_img_dir':p_img_dir}

            _refcoco_split['target_class_in_context_ratio'] = _refcoco_split.progress_apply(lambda x: target_category_score(x, **kwargs), axis=1)

            print(f'write results to {path}')
            _refcoco_split.reset_index()[
                ['ann_id', 'target_class_in_context_ratio']
            ].to_json(path, orient='records')
            
        else: 
            print(f'skipping -- file {path} already exists')
            
    print('Done!')
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', default='refcoco')
    parser.add_argument('--out_dir', default='./generated/target_context_classes')
    
    args = parser.parse_args()
    
    print(vars(args))
    
    config = Config()
    
    main(args, config)