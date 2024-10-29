from __future__ import annotations
import numpy as np
from utils import read_panoptic_anns, imgid2catid


class AnnSceneExtractor():
    """
    Extracting scene summaries from annotations.
    """

    def __init__(self, config) -> None:
        """
        SceneExtractor class constructor, inherits from the Detector constructor.

        Args:
            config (Config): Config instance
        """
        self.coco_base = config.coco_base
        self.panoptic_ann_dir = config.panoptic_ann_dir
        self.panoptic_img_dir = config.panoptic_img_dir
        
        self.panoptic_df, self.raw_categories = read_panoptic_anns(config.panoptic_ann_dir)
        
        self.categories_map = dict()
        self.category_idx_to_raw_id = dict()
        for i, (r_i, c) in enumerate(self.raw_categories.items()):
            self.categories_map[i] = c['name']
            self.category_idx_to_raw_id[i] = r_i
        self.raw_id_to_category_idx = {v: k for k, v in self.category_idx_to_raw_id.items()}
        
    
    def pixel_ids_from_image_id(self, image_id, map_to_continuous_ids=True) -> np.ndarray:
        """
        Read coco image and return pixel categories

        Args:
            image_id (int): COCO Image ID
            coco_split (str, optional): COCO split / image directory. Defaults to "train2014".

        Returns:
            np.ndarray: category ids for pixels in the image
        """
        pixel_ids = imgid2catid(image_id, self.panoptic_df, self.panoptic_img_dir)
        
        if map_to_continuous_ids:
            pixel_ids = np.vectorize(self.raw_id_to_category_idx.__getitem__)(pixel_ids)
        
        return pixel_ids