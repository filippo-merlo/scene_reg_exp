import json
from os import path as osp
import os.path as osp
from tqdm.autonotebook import tqdm
from utils import get_refcoco_df
from configuration import Config
import argparse

tqdm.pandas()

def main(args, config):

    refcoco_df = get_refcoco_df(config.ref_dir).groupby('ann_id').first()

    split_df = refcoco_df.loc[refcoco_df.refcoco_split == args.split].groupby('ann_id').first()

    unique_img_split_df = split_df.groupby('image_id').sample(random_state=args.random_state)
    sample = unique_img_split_df.sample(args.sample_size, random_state=args.random_state)
    
    sample_ids = sample.index.to_list()
    
    out_path = osp.join(args.out_path, f'{args.dataset}-{args.split}-sample_ids.json')
    print(f'write ids to {out_path}')
    with open(out_path, 'w') as f:
        json.dump(sample_ids, f)

if __name__ == '__main__':
    
    config = Config()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='refcoco')
    parser.add_argument('--split', default='testB', choices=['val', 'testA', 'testB'])
    parser.add_argument('--out_path', default=osp.abspath('./generated/id_files'))
    parser.add_argument('--sample_size', default=200, type=int)
    parser.add_argument('--random_state', default=111, type=int)
    args = parser.parse_args()
    
    main(args, config)