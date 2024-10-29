from glob import glob
import os.path as osp
import os
import json
import sys
import shutil
import argparse

def hash_filename(fname):
    file_hash = hash(fname)
    pos_hash = file_hash & sys.maxsize
    return pos_hash

def main(args):
    
    if not osp.isdir(args.output_dir):
        print(f'create dir {args.output_dir}')
        os.makedirs(args.output_dir)

    # collect files
    
    html_input_files = glob(f'{args.input_dir}/*.html')
    csv_input_files = glob(f'{args.input_dir}/*.html')

    assert len(html_input_files) == len(csv_input_files), f'mismatch between HTML and CSV files in {args.input_dir}'
    print(f'{len(html_input_files)} HTML files, {len(csv_input_files)} CSV files')
    
    # create dicts for filename conversion

    html_conversion = dict()
    csv_conversion = dict()

    for html_filepath in html_input_files:
        
        html_filename = osp.split(html_filepath)[-1]
        file_root = osp.splitext(html_filename)[0]
        
        hashed_filename = hex(hash_filename(file_root))
        
        html_conversion[f'{file_root}.html'] = f'{hashed_filename}.html'
        csv_conversion[f'{file_root}.csv'] = f'{hashed_filename}.csv'
        
    conversions = {'html': html_conversion, 'csv': csv_conversion}
    conversions_path = osp.join(args.output_dir, 'conversions.json')
    print(f'writing conversions to {conversions_path}')
    with open(conversions_path, 'w') as f:
        json.dump(conversions, f)
        
    # rename HTML and CSV files using the conversion dicts

    for scr_file, dst_file in html_conversion.items():
        src_path = osp.join(args.input_dir, scr_file)
        dst_path = osp.join(args.output_dir, dst_file)
        print(f'copying {src_path} to {dst_path}')
        shutil.copy2(src_path, dst_path)
        
    for scr_file, dst_file in csv_conversion.items():
        src_path = osp.join(args.input_dir, scr_file)
        dst_path = osp.join(args.output_dir, dst_file)
        print(f'copying {src_path} to {dst_path}')
        shutil.copy2(src_path, dst_path)
        
    print('DONE!')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='./generated/identification_samples')
    parser.add_argument('--output_dir', default='./generated/masked_identification_samples')
    args = parser.parse_args()
    
    main(args)