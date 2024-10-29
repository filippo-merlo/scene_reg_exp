import json
import os.path as osp
import argparse
import pickle
from glob import glob


def convert_files(args):
    q = osp.join(args.input_dir, args.dataset, args.architecture, '**' , 
                 f'{args.dataset}_{args.split.lower()}_*_generated*.pkl')
    files = sorted(glob(q))
    print('found files:')
    for f in files:
        print(osp.split(f)[-1])
    print('\n')
    
    for file in files:

        generated = []
        with open(file, 'rb') as f:
            data = pickle.load(f)
            for d in data:
                generated.append(
                    {
                        'ann_id': d['ann_id'],
                        'generated': d['expression_string']
                    }
                )   
        
        out_file = file.replace('.pkl', '.json')

        print(f'{osp.split(file)[-1]} -> {osp.split(out_file)[-1]}')
        with open(out_file, 'w') as f:
            json.dump(generated, f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir')
    parser.add_argument('--dataset', default='refcoco')
    parser.add_argument('--architecture', choices=['TRF', 'CLIP_GPT'])
    parser.add_argument('--split', default='val', choices=['testA', 'testB', 'val'])

    args = parser.parse_args()

    convert_files(args)
