from __future__ import annotations

from scipy.special import softmax
import numpy as np

np.seterr(all="raise")


class SceneExtractor:
    """
    Extracting scene summaries with detectron2 or based on panoptic segmentation annotations.
    """

    def __init__(self, config) -> None:
        """
        SceneExtractor class constructor

        Args:
            config (Config object): configuration for Scene Extractor
        """
        self.config = config
        if config.extraction_type == "predicted":
            # use detectron2 as backbone
            from DetectronSceneExtractor import DetectronSceneExtractor

            self.backbone = DetectronSceneExtractor(config)
        elif config.extraction_type == "annotated":
            # rely on panoptic segmentation annotations
            from AnnSceneExtractor import AnnSceneExtractor

            self.backbone = AnnSceneExtractor(config)
        else:
            raise NotImplementedError(
                'config.extraction_type has to be "predicted" or "annotated"'
            )

        print(
            f'Extraction type "{config.extraction_type}":',
            f"Initializing SceneExtractor with backbone {self.backbone.__class__.__name__}",
        )

        self.categories_map = self.backbone.categories_map

    def pixel_ids_from_image_id(self, *args, **kwargs):
        """use backbone method"""
        return self.backbone.pixel_ids_from_image_id(*args, **kwargs)

    # masking methods - weight or mask values in pixel ids before extracting the final summary
    #   - mask_target
    #   (DetectronSceneExtractor has more)

    def mask_target(
        self, pixel_ids: np.ndarray, bb: tuple, mask_id: int = -1
    ) -> np.ndarray:
        """
        mask out target region from pixel ids, given a bounding box

        Args:
            pixel_ids (np.ndarray): category ids for image pixels
            bb (tuple): bounding box in x, y, w, h format
            mask_id (int, optional): value used for masking. Defaults to -1.

        Returns:
            np.ndarray: category ids for image pixels, with masked out target region
        """

        (x_min, y_min), (x_max, y_max) = self.transform_bbox_format(bb)

        # mask target region
        pixel_ids = np.copy(pixel_ids)
        assert pixel_ids.ndim in (2, 3)
        if pixel_ids.ndim == 2:
            pixel_ids[y_min:y_max, x_min:x_max] = mask_id
        else:
            pixel_ids[y_min:y_max, x_min:x_max, :] = mask_id

        return pixel_ids

    # summary extraction method - get (transformed) pixel id map and compute scene summary
    #   - extract_summary_from_pixel_ids

    def extract_summary_from_pixel_ids(
        self,
        pixel_ids: np.ndarray,
        exclude_mask_id: int = -1,
        weights: np.ndarray = None,
        softmax_results: bool = False,
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """
        Extract scene summary vector from image pixel ids
        (obtained via SceneExtractor.get_category_map)

        Args:
            pixel_ids (np.ndarray): category ids for image pixels
            exclude_mask_id (int, optional): ID value to be masked out.
                No masking out if set to None. Defaults to -1.
            weights (np.ndarray or NoneType):
                Weights for weighted counts with same shape as pixel_ids

        Returns:
            tuple[np.ndarray, tuple[np.ndarray,np.ndarray]]:
                scene summary vector and tuple with category ids and respective areas occupied
        """

        # exclude masking values
        if exclude_mask_id is not None:
            mask = np.logical_not(pixel_ids == exclude_mask_id)
            pixel_ids = pixel_ids[mask]
            if weights is not None:
                weights = weights[mask]

        # flatten arrays for np.bincount
        pixel_ids = pixel_ids.flatten()
        weights = weights.flatten() if weights is not None else weights

        # get counts and area proportions for detected segments
        counts = np.bincount(
            pixel_ids, minlength=len(self.categories_map), weights=weights
        )
        total_area = max(np.product(pixel_ids.shape), 1)  # default to one if area is 0

        segment_areas = counts / total_area

        if softmax_results:
            # apply softmax only to values > 0
            mask = segment_areas > 0
            segment_areas[mask] = softmax(segment_areas[mask])

        return segment_areas

    # helper methods
    #   - transform_bbox_format
    #   - bounding_box_center
    #   - get_distance_map

    @staticmethod
    def transform_bbox_format(bb: tuple) -> tuple[tuple, tuple]:
        """
        transform bounding box from x, y, w, h to corner points format

        Args:
            bb (tuple): bounding box in x, y, w, h format

        Returns:
            tuple[tuple, tuple]: bounding box in corner points format
        """

        # get bounding box coordinates (round since integers are needed)
        x, y, w, h = round(bb[0]), round(bb[1]), round(bb[2]), round(bb[3])

        # calculate minimum and maximum values for x and y dimension
        x_min, x_max = x, x + w
        y_min, y_max = y, y + h

        # return bounding box as corner points
        return (x_min, y_min), (x_max, y_max)

    def bounding_box_center(self, bb: tuple) -> tuple:
        """
        get center point from bounding box

        Args:
            bb (tuple): bounding box in x, y, w, h format

        Returns:
            tuple: center point (x_center, y_center)
        """

        x, y, w, h = bb
        x_center = round(x + w / 2)
        y_center = round(y + h / 2)

        return x_center, y_center

    @staticmethod
    def get_distance_map(
        pixel_ids: np.ndarray,
        point: tuple,
        normalize: bool = True,
        return_proximity: bool = True,
    ) -> np.ndarray:
        """
        create distance map between point in image and the remaining coordinates

        Args:
            pixel_ids (np.ndarray): input pixel ids (or image) as numpy array
            point (tuple): x and y coordinates for reference point
            normalize (bool, optional): Flag for normalizing distances to values between 0 and 1. Defaults to False.
            return_proximity (bool, optional): Return proximity instead of distance (i.e. 1 - distance). Defaults to True.

        Returns:
            np.ndarray: map with distances between coordinates and reference point
        """

        # initialize position matrix
        if pixel_ids.ndim == 2:
            img_x, img_y = pixel_ids.shape
        elif pixel_ids.ndim == 3:
            img_x, img_y, _ = pixel_ids.shape
        else:
            raise ValueError(
                f"invalid number of dimensions for input image with shape {pixel_ids.ndim}"
            )
        idx_matrix = np.indices((img_x, img_y))  # shape (2, img_x, img_y)
        pos_matrix = np.copy(idx_matrix).astype(float)

        # create matrix containing the repeated reference point
        point = point[::-1]  # (x,y) -> (y,x)
        point_matrix = np.copy(pos_matrix)
        point_matrix[0, :, :] = point[0]
        point_matrix[1, :, :] = point[1]

        # euclidean distance between position matrix and reference point (as matrix distance)
        dist_map = pos_matrix - point_matrix  #  pointwise difference
        dist_map = np.square(dist_map)  #  square values
        dist_map = dist_map.sum(axis=0)  # sum x and y values -> shape (img_x, img_y)
        dist_map = np.sqrt(dist_map)  # square root

        if normalize:
            # max distance: distance between upper left (0,0) and lower right (img_x, img_y)
            max_distance = np.sqrt(
                np.sum(np.square((img_x, img_y)))
            )  # more precisely: (img_x - 0, img_y - 0)
            # normalize distance map
            dist_map = dist_map / max_distance

        if return_proximity:
            return 1 - dist_map

        return dist_map
