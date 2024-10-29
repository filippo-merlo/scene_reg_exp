from os.path import join, abspath, dirname, isdir


class Config():
    """
    config wrapper containing settings for RefExp annotation
    (change if needed)
    """

    def __init__(self, project_root=None) -> None:
        
        self.extraction_type = 'annotated'  # 'predicted' or 'annotated'

        # project root (relative to this file)
        self.project_root = dirname(abspath(__file__)) if project_root is None else project_root

        # dataset paths
        self.data_root = abspath(
            'PATH_TO_DATASETS')  # e.g. './data'

        self.coco_base = join(self.data_root, 'COCO')

        self.referit_base = join(self.data_root, 'ReferIt')
        self.refcoco_base = join(self.referit_base, 'refcoco')

        self.vg_base = join(self.data_root, 'VisualGenome')
        
        self.panoptic_ann_dir = join(self.coco_base, 'annotations', 'panoptic_train2017.json')
        self.panoptic_img_dir = join(self.coco_base, 'annotations', 'panoptic_train2017')

        # output directory
        self.features_out = join(self.project_root, 'data', 'extracted_features')
        self.intermediate_out = join(self.project_root, 'data', 'intermediate')
        
        # Detectron Settings
        self.device = 'cpu'
        self.detectron_model_type = "PS"
        self.detectron_score_thresh = 0.7

    def get_model_config(method:str) -> dict:
        """
        get method specific information for feature extraction with huggingface

        Args:
            method (str): method description ('resnet', 'vit', 'vithybrid' or 'bit')

        Raises:
            NotImplementedError: Raised if method has invalid value

        Returns:
            dict: huggingface path for pretrained model, huggingface model object, feature dimensions
        """

        import transformers

        if method == 'resnet':
            return {
            'pretrained_model': "microsoft/resnet-152", 
            'Model': transformers.ResNetModel, 
            'd_feats': 2048,
        }
        elif method == 'vit':
            return {
            'pretrained_model': "google/vit-base-patch16-224",
            'Model': transformers.ViTModel,
            'd_feats': 768,
        }
        elif method == 'vithybrid':
            return {
            'pretrained_model': "google/vit-hybrid-base-bit-384",
            'Model': transformers.ViTHybridModel,
            'd_feats': 768,
        }
        elif method == 'bit':
            return {
            'pretrained_model': "google/bit-50",
            'Model': transformers.BitModel,
            'd_feats': 2048,
        }
        else:
            raise NotImplementedError(
                'requested method is not implemented')


if __name__ == '__main__':
    config = Config()

    print(f'data_root : {config.data_root} (valid: {isdir(config.data_root)})')
    print(f'coco_base : {config.coco_base} (valid: {isdir(config.coco_base)})')
    print(f'referit_base : {config.referit_base} (valid: {isdir(config.referit_base)})')
    print(f'refcoco_base : {config.refcoco_base} (valid: {isdir(config.refcoco_base)})')
    print(f'vg_base : {config.vg_base} (valid: {isdir(config.vg_base)})')
    print(f'features_out : {config.features_out} (valid: {isdir(config.features_out)})')
    print(f'intermediate_out : {config.intermediate_out} (valid: {isdir(config.intermediate_out)})')
