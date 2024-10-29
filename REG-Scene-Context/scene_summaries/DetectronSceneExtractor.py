# based on this tutorial: https://www.youtube.com/watch?v=Pb3opEFP94U
from __future__ import annotations

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo
import os.path as osp
from utils import filename_from_id

import cv2
import numpy as np


class Detector:
    """
    Wrapper for image processing with detectron2.
    Supports:
        Object Detection,
        Instance Segmentation,
        Keypoint Detection,
        LVIS Segmentation,
        Panoptic Segmentation
    """

    def __init__(self, model_type="PS", device="cuda", score_thresh=0.7) -> None:
        """
        Detector class constructor.

        Args:
            model_type (str, optional):
                Predictor / processing type:
                    "OD" (Object Detection),
                    "IS" (Instance Segmentation),
                    "KP" (Keypoint Detection),
                    "LIVS" (LVIS Segmentation) or
                    "PS" (Panoptic Segmentation).
                Defaults to "PS".
            device (str, optional): Run on GPU (CUDA) or CPU. Defaults to "cuda".
            score_thresh (float, optional): Threshold for Fast R-CNN bounding boxes. Defaults to 0.7.
        """
        self.cfg = get_cfg()
        self.model_type = model_type
        self.device = device

        # Load Model config and pretrained model
        if self.model_type == "OD":  # object detection
            self.cfg.merge_from_file(
                model_zoo.get_config_file(
                    "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
                )
            )
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
            )
        elif self.model_type == "IS":  # instance segmentation
            self.cfg.merge_from_file(
                model_zoo.get_config_file(
                    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
                )
            )
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            )
        elif self.model_type == "KP":  # keypoint detection
            self.cfg.merge_from_file(
                model_zoo.get_config_file(
                    "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
                )
            )
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
            )
        elif self.model_type == "LIVS":  # LVIS Segmentation
            self.cfg.merge_from_file(
                model_zoo.get_config_file(
                    "LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml"
                    #'LVISv1-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml'
                )
            )
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml"
                #'LVISv1-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml'
            )
        elif self.model_type == "PS":  # Panoptic Segmentation
            self.cfg.merge_from_file(
                model_zoo.get_config_file(
                    "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"
                )
            )
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"
            )

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
        self.cfg.MODEL.DEVICE = self.device

        self.predictor = DefaultPredictor(self.cfg)
        self.metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
        self.metadata.segment_classes = (
            self.metadata.thing_classes + self.metadata.stuff_classes
        )

    def visualize(self, image: np.ndarray) -> None:
        """
        Run the processing pipeline and display the results with cv2.imshow

        Args:
            image (np.ndarray): input image as numpy array
                (BGR pixel values, shape: height x width x channels)
        """

        if self.model_type != "PS":
            predictions = self.predictor(image)

            viz = Visualizer(
                image[:, :, ::-1],
                metadata=self.metadata,
                instance_mode=ColorMode.IMAGE_BW,  # IMAGE, IMAGE_BW, SEGMENTATION
            )

            output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))

        else:
            predictions, segmentInfo = self.predictor(image)["panoptic_seg"]
            viz = Visualizer(
                image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
            )
            output = viz.draw_panoptic_seg_predictions(
                predictions.to("cpu"), segmentInfo
            )

        cv2.imshow("Result", output.get_image()[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def open_image(self, imagePath: str) -> np.ndarray:
        """
        Read image file with cv2.imread

        Args:
            imagePath (str): path to input image

        Returns:
            np.ndarray: numpy array with BGR pixel values
                (shape: height x width x channels)
        """
        return cv2.imread(imagePath)

    def process_image(self, image: np.ndarray) -> dict:
        """
        Run the processing pipeline

        Args:
            image (np.ndarray): input image as numpy array
                (BGR pixel values, shape: height x width x channels)

        Returns:
            dict: prediction results
                (segment ids for image pixels, information for detected segments)
        """
        return self.predictor(image)


class DetectronSceneExtractor(Detector):
    """
    Extracting scene summaries with detectron2.
    Inherits from the Detector class.
    """

    def __init__(self, config) -> None:
        """
        SceneExtractor class constructor, inherits from the Detector constructor.

        Args:
            model_type (str, optional):
                Predictor / processing type. Has to be set to "PS" or left implicit
                (as summary extraction relies on Panoptic Segmentation).
                Defaults to "PS".
            device (str, optional): Run on GPU (CUDA) or CPU. Defaults to "cuda".
            score_thresh (float, optional): Threshold for Fast R-CNN bounding boxes. Defaults to 0.7.
        """
        self.coco_base = config.coco_base
        super(DetectronSceneExtractor, self).__init__(
            model_type=config.detectron_model_type,
            score_thresh=config.detectron_score_thresh,
            device=config.device,
        )

        self.categories_map = {
            i: c for i, c in enumerate(self.metadata.segment_classes)
        }

    # pixelwise category map (underlying structure for scene summaries)

    def get_category_map(self, image: np.ndarray) -> np.ndarray:
        """
        Map pixels from input image to category ids
        (as determined by Panoptic Segmentation)

        Args:
            image (np.ndarray): input image as numpy array
                (BGR pixel values, shape: height x width x channels)

        Returns:
            np.ndarray: category ids for pixels in the input image
        """

        # process image
        processed = self.process_image(image)
        # get pixel ids and detected things / stuff
        pixel_ids, segmentInfo = processed["panoptic_seg"]

        # map segment ids to category ids
        # (id 0 defaults to 'things' / 1st entry in list of stuff classes)
        pixel_ids[pixel_ids == 0] = len(self.metadata.thing_classes)

        for seg in segmentInfo:
            if seg["isthing"]:
                cat_id = seg["category_id"]  # no offset for things
                pixel_ids[pixel_ids == seg["id"]] = cat_id
            else:
                cat_id = seg["category_id"] + len(
                    self.metadata.thing_classes
                )  # offset for stuff
                pixel_ids[pixel_ids == seg["id"]] = cat_id

        return pixel_ids.cpu().numpy()

    def pixel_ids_from_image_id(self, image_id, coco_split="train2014") -> np.ndarray:
        """
        Read coco image and return pixel categories

        Args:
            image_id (int): COCO Image ID
            coco_split (str, optional): COCO split / image directory. Defaults to "train2014".

        Returns:
            np.ndarray: category ids for pixels in the image
        """
        filename = filename_from_id(image_id)
        img_path = osp.join(self.coco_base, coco_split, filename)

        # read image
        image = self.open_image(img_path)
        # get category ids for pixels
        pixel_ids = self.get_category_map(image)

        return pixel_ids
