import argparse
import os
import os.path as osp
import json
import pickle
import csv

import base64
from io import BytesIO
import pandas as pd
from configuration import Config
from utils import get_refcoco_df, get_annotated_image


ENTRY_TEMPLATE = """

<p>

<div>
<img src="data:image/jpeg;base64, {b64_image}"/>

<br>

<table>
  <tr>
    <td>index:</td>
    <td>{index}</td>
  </tr>
  <tr>
    <td>ann_id:<td>
    <td>{ann_id}</td>
  </tr>
  <tr>
    <td>expression:</td>
    <td>{generated}</td>
  </tr>
</table> 
</div>

</p>

<hr>
"""

HTML_TEMPLATE = """ <!DOCTYPE html>
    <html>
    <head>
    <title>Page Title</title>
    </head>
    <body>

    {body}

    </body>
    </html> """
    

def img_to_b64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    b64_str = base64.b64encode(buffered.getvalue()).decode('ASCII')
    return b64_str


def parse_filename(filename):

    file_stem = os.path.splitext(filename)[0]
    dataset, split, architecture, context, noise, epoch, _ = file_stem.split('_')
    context = context.split(':')[-1]
    noise = noise.split(':')[-1]
    epoch = epoch.split(':')[-1]
    
    return dataset, split, architecture, context, noise, epoch


def main(args, config):
    
    dataset, split, architecture, context, noise, epoch = parse_filename(
        osp.split(args.expression_file)[-1])
    
    if not osp.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    
    outfile_stem = f"{dataset}_{split}_{architecture}_{context}_noise:{noise}_sample"
    csv_outfile = osp.join(args.output_dir, outfile_stem + '.csv')
    html_outfile = osp.join(args.output_dir, outfile_stem + '.html')
    
    if args.overwrite_existing_files or not (osp.isfile(csv_outfile) and osp.isfile(html_outfile)):
        
        print(f'processing file {args.expression_file} with ids from {args.id_file}')
        
        # import refcoco
        refcoco_df = get_refcoco_df(config.ref_dir).groupby('ann_id').first()

        # import generated expressions
        
        if args.expression_file.endswith('.pkl'):
            # vanilla transformer
            generated = []
            with open(args.expression_file, 'rb') as f:
                data = pickle.load(f)
                for d in data:
                    generated.append(
                        {
                            'ann_id': d['ann_id'],
                            'generated': d['expression_string']
                        }
                    )
        elif args.expression_file.endswith('.json'):
            # CLIP-GPT model
            with open(args.expression_file, 'r') as f:
                generated = json.load(f)
        else:
            raise Exception(f'expression file has to be JSON or pickle: {args.expression_file}')
        
        generated_df = pd.DataFrame(generated).set_index('ann_id')

        # load ids
        with open(args.id_file, 'r') as f:
            ids = json.load(f)

        # restrict to ids and merge
        refcoco_sample = refcoco_df.loc[ids]
        generated_sample = generated_df.loc[ids]

        merged = pd.merge(
            refcoco_sample,
            generated_sample,
            left_on='ann_id',
            right_on='ann_id'
        )
            
        # generate html file with samples
        html_entries = []

        # generate csv file for annotation
        with open(csv_outfile, 'w', newline='') as csvfile:
            fieldnames = ['idx', 'ann_id', 'expression', 'annotation']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for i, entry in merged.reset_index().iterrows():
                
                # get infos
                ann_id = entry['ann_id']
                generated = entry['generated']
                image = get_annotated_image(entry, config.coco_dir)
                
                # write to csv
                writer.writerow({'idx': i, 'ann_id': ann_id, 'expression': generated, 'annotation': ''})

                
                # convert image to b64 and create html entry
                b64_image = img_to_b64(image)
                
                html_entry = ENTRY_TEMPLATE.format(
                b64_image=b64_image,
                index=i,
                ann_id=ann_id,
                generated=generated
                ).strip()
                
                html_entries.append(html_entry)
                
        # combine html entries and write to file
        entries_str = '\n'.join(html_entries)
        with open(html_outfile, 'w') as f:
            f.write(HTML_TEMPLATE.format(body=entries_str))
            
        print(f'{args.expression_file}: DONE')
        
    else:
        print('Files already found:')
        print(csv_outfile)
        print(html_outfile)
        

if __name__ == '__main__':
    
    config = Config()
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--expression_file', required=True)
    parser.add_argument('--id_file', required=True)
    parser.add_argument('--dataset', default='refcoco')
    parser.add_argument('--output_dir', default=osp.join(config.project_path, 'generated', 'identification_samples'))
    parser.add_argument('--overwrite_existing_files', action='store_true')
    
    args = parser.parse_args()
    
    for k, v in vars(args).items():
        print(f'{k}: {v}')
    
    main(args, config)