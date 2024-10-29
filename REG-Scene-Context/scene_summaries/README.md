To generate symbolic scene features:
- Update the file ```config.py``` with custom settings (e.g. data locations)
- Download the 2017 Panoptic Train/Val annotations from https://cocodataset.org/ and unzip it as specified in ```config.py```
- Run ```scene_context_extraction.py``` to extract the features (arguments as specified in the file can be used for settings on the fly)

Note that this code defaults to using on existing annotations for panoptic segmentation (which is the setting reported in the paper). To instead use segmentations as predicted by a Detectron2 model, change the value for ```extraction_type``` in ```config.py``` to ```predicted```.
