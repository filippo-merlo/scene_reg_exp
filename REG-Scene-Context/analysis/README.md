# Analysis

Preparation Pipeline:

1. Train models
2. Create expressions
    - filename schema: 
    ```(refcoco|refcoco+)_(val|testA|testB)_(trf|clpgpt)_context:(global|nocontext|scene)_noise:\d-\d_epoch:\d\d_generated.(json|pkl)```
    - pickle files (trf) have to be list of dict with keys `ann_id` and `expression_string`
    - json files (clpgpt) have to be list of dict with keys `ann_id` and `generated`
3. Create sample IDs with `make_sample_ids.py`
4. Create HTML/CSV pairs with `make_eval_samples.py`
5. Mask filenames with `mask_names_in_identification_samples.py`
6. Annotate & evaluate

Annotation Evaluation Pipeline:

1. Compile annotations using ```compile_annotations.ipynb```
2. Place annotations in ```generated/compiled_annotations```
3. Compute target class coverage in context using ```compute_target_class_coverage.py```
4. Annotation results: Run code in ```merged_annotation_eval.ipynb```
5. Target class coverage stats: ```target_class_eval.ipynb```

Quality Evaluation Pipeline:

1. Compute automatic quality metrics using ```compute_automatic_metrics.sh```
2. Compile results and render tables in ```automatic_metrics_eval.ipynb```

Generated data can be downloaded from here: https://drive.google.com/drive/folders/1Bu1a57HweS8-1jaF5momkmSVxbwkyxQj
(to be placed in ```generated``` folder)
