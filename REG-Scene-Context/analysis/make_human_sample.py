import json
import os
import os.path as osp
from tqdm.autonotebook import tqdm
from utils import get_refcoco_df
import argparse
from configuration import Config

def main(args, config):
    refcoco_path = osp.join(config.ref_base, args.dataset)
    refcoco_df = get_refcoco_df(refcoco_path)
    
    split_df = refcoco_df.loc[refcoco_df.refcoco_split == args.split]
    sample = split_df.groupby('ann_id').sample(random_state=args.random_state)
    
    out = [{
        'ann_id': entry.ann_id,
        'generated': entry.caption
        } for _, entry in sample.iterrows()
    ]
    
    out_filename = f'{args.dataset}_{args.split.lower()}_human_context:global_noise:0-0_epoch:00_generated.json'
    out_path = osp.join(args.out_dir, out_filename)
    print(f'write results to {out_path}')
    with open(out_path, 'w') as f:
        json.dump(out, f)

if __name__ == '__main__':
    
    config = Config()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_state', default=111)
    parser.add_argument('--out_dir', default=None)
    parser.add_argument('--dataset', default='refcoco')
    parser.add_argument('--split', default='testB')
    args = parser.parse_args()
    
    if args.out_dir is None:
        args.out_dir = osp.join(osp.curdir, 'generated', 'generated_expressions', args.dataset, 'HUMAN', 'noise_0-0_global')
    args.out_dir = osp.abspath(args.out_dir)
    
    if not osp.isdir(args.out_dir):
        os.makedirs(args.out_dir)
        print(f'create directory {args.out_dir}')
       
    print(vars(args)) 
    
    main(args, config)