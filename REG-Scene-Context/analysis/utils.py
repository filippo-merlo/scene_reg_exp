import numpy as np
from PIL import Image, ImageDraw
from os.path import join, isfile
import torch
import matplotlib.pyplot as plt
import json
import pandas as pd
import matplotlib.patches as patches
from skimage import io
from decimal import Decimal


def get_refcoco_df(path):
    """get RefCOCO* annotations as pd.DataFrame

    Args:
        path (string): path to RefCOCO* base dir

    Returns:
        pd.DataFrame: RefCOCO* annotations
    """
    filepath = join(path, 'instances.json')
    with open(filepath) as file:
        instances = json.load(file)
        instances = pd.DataFrame(instances['annotations']).set_index('id')

    filename = 'refs(umd).p' if path.endswith('refcocog') else 'refs(unc).p'  # different file name for RefCOCOg
    filepath = join(path, filename)
    captions = pd.read_pickle(filepath)
    captions = split_sentences(pd.DataFrame(captions))

    captions = pd.merge(captions,
                        instances[['image_id', 'bbox', 'category_id']],
                        left_on='ann_id',
                        right_on='id').set_index('sent_id')

    return captions


def read_panoptic_anns(ann_path):
    with open(ann_path, "r") as f:
        anns = json.load(f)

    ann_df = pd.DataFrame(anns["annotations"]).set_index("image_id")
    categories = {c["id"]: c for c in anns["categories"]}
    categories[0] = {"supercategory": "void", "isthing": 0, "id": 0, "name": "void"}

    return ann_df, categories


def split_sentences(df):
    """
        split sentences in refcoco df
    """
    rows = []

    def coco_split(row):
        for split in ['train', 'val', 'test']:
            if split in row['file_name']:
                return split
        return None

    def unstack_sentences(row):
        nonlocal rows
        for i in row.sentences:
            rows.append({
                'sent_id': i['sent_id'],
                'ann_id': row['ann_id'],
                'caption': i['sent'],
                'ref_id': row['ref_id'],
                'refcoco_split': row['split'],
                'coco_split': coco_split(row)
            })

    df.apply(lambda x: unstack_sentences(x), axis=1)

    return pd.DataFrame(rows)


def get_img(fname, img_dir):
    return Image.open(join(img_dir, fname))


def get_entry_image(entry, coco_dir):
    
    image_file = filename_from_id(entry['image_id'], prefix='COCO_train2014_')
    
    image_filepath = join(coco_dir, 'train2014', image_file)
    assert isfile(image_filepath)
    image = Image.open(image_filepath)
    
    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image

def get_annotated_image(entry, coco_dir, bbox_color='blue', width=3):
    image = get_entry_image(entry, coco_dir)
    bbox_xyxy = xywh_to_xyxy(entry['bbox'])
    
    draw = ImageDraw.Draw(image)
    draw.rectangle(bbox_xyxy, outline=bbox_color, width=width)
    
    return image


def rgb2segid(img_array):
    ids = img_array[:, :, 0] + img_array[:, :, 1] * 256 + img_array[:, :, 2] * 256**2
    return ids


def interpolate_2d(tensor, shape):
    return (
        torch.nn.functional.interpolate(tensor.unsqueeze(0).unsqueeze(0), shape)
        .squeeze(0)
        .squeeze(0)
    )


def segid2catid(id_array, segments_info):
    cat_ids = np.zeros_like(id_array)

    for s in segments_info:
        # get ids
        pixel_id = s["id"]
        cat_id = s["category_id"]
        # get mask from pixel ids array
        m = id_array == pixel_id
        # apply mask to cat_ids and set value to cat_id
        cat_ids[m] = cat_id

    return cat_ids


def imgid2catid(image_id, ann_df, img_dir, return_img=False):
    entry = ann_df.loc[image_id]  # get entry
    img_array = np.array(
        get_img(entry.file_name, img_dir)
    )  # read image & convert to np array
    pxl_ids = rgb2segid(img_array)  # get segment ids
    cat_ids = segid2catid(pxl_ids, entry.segments_info)  # convert to category ids

    if return_img:
        return img_array, cat_ids
    return cat_ids


def convert_bbox_format(bb):
    """convert bounding box from xywh to corner coordinates format"""
    # get bounding box coordinates (round since integers are required)
    x, y, w, h = round(bb[0]), round(bb[1]), round(bb[2]), round(bb[3])

    # calculate minimum and maximum values for x and y dimension
    x_min, x_max = x, x + w
    y_min, y_max = y, y + h

    return x_min, x_max, y_min, y_max


def xywh_to_xyxy(bb):

    x, y, w, h = map(Decimal, map(str, bb))

    # upper left
    x1 = x
    y1 = y
    # lower right
    x2 = x + w
    y2 = y + h

    out = x1, y1, x2, y2

    return tuple(map(float, out))


def get_panoptic_attention_map(
    att, image_id, panoptic_categories, panoptic_df, panoptic_img_dir, bbox=None
):
    """get the weighted sum (model attention as weights) and ground truth distribution of classes in a panoptic segmentation map

    Args:
        att (tensor): attention map (14x14)
        image_id (int): the id of the requested image
        panoptic_categories (dict): map from panoptic category idx to annotations
        panoptic_df (DataFrame): segment information and annotation file names
        panoptic_img_dir (str): base directory for panoptic annotation files
        bbox (tuple or NoneType, optional): Bounding box for masking out target before counting class pixels. Defaults to None.

    Returns:
        tuple: attention scores over categories, ground truth distribution (category coverage)
    """

    # read and reshape pixel ids
    pixel_id_map = imgid2catid(image_id, panoptic_df, panoptic_img_dir)

    # interpolate attention to image size
    att = interpolate_2d(att, pixel_id_map.shape)

    # get classes the model is attending to
    flatten_pixel_ids = pixel_id_map.flatten()
    flatten_weights = att.flatten()

    if bbox is not None:
        # mask out target if specified
        mask = np.ones_like(pixel_id_map, dtype=bool)
        x_min, x_max, y_min, y_max = convert_bbox_format(bbox)
        mask[y_min:y_max, x_min:x_max] = False
        flatten_mask = mask.flatten()

        flatten_pixel_ids = flatten_pixel_ids[flatten_mask]
        flatten_weights = flatten_weights[flatten_mask]

    n_categories = max(panoptic_categories.keys()) + 1

    att_scores = np.bincount(
        flatten_pixel_ids, minlength=n_categories, weights=flatten_weights
    )
    gt_scores = np.bincount(
        flatten_pixel_ids, minlength=n_categories  # without weights
    )

    return att_scores, gt_scores


def get_attended_classes(att, image_id, panoptic_categories, panoptic_df, panoptic_img_dir, bbox=None):
    
    att_scores, gt_scores = get_panoptic_attention_map(
        torch.tensor(att), 
        image_id, 
        panoptic_categories, 
        panoptic_df, 
        panoptic_img_dir, 
        bbox
    )

    # normalize with total attention mass / coverage
    norm_att_scores = (att_scores / att_scores.sum()) if att_scores.sum() > 0 else att_scores
    norm_gt_scores = (gt_scores / gt_scores.sum()) if gt_scores.sum() > 0 else gt_scores

    # zip with category info
    n_categories = norm_att_scores.shape[0]
    zipped_scores = zip(
        range(n_categories), norm_att_scores, norm_gt_scores
    )
    zipped_scores = [
        (category_id, (panoptic_categories[category_id], att, gt)) 
        for category_id, att, gt in zipped_scores 
        if category_id in panoptic_categories.keys()
    ]
    
    return zipped_scores


def get_attended_classes_2(
    att,
    image_id,
    panoptic_categories,
    panoptic_df,
    panoptic_img_dir,
    bbox=None,
    return_sorted=False,
    gt_threshold=0.0,
):
    """get normalized and raw attention scores and ground truth coverage for panoptic categories

    Args:
        att (tensor): _description_
        image_id (int): the id of the requested image
        panoptic_categories (dict): map from panoptic category idx to annotations
        panoptic_df (DataFrame): segment information and annotation file names
        panoptic_img_dir (str): base directory for panoptic annotation files
        bbox (tuple or NoneType, optional): Bounding box for masking out target before counting class pixels. Defaults to None.
        return_sorted (bool, optional): Sort results by normalized scores. Defaults to False.

    Returns:
        list: list of normalized / raw attention scores, ground truth coverage and annotations for panoptic categories present in the image
        normalized scores ~ factor of deviation from expected attention mass allocated to class
            (0: no attention on class; 1: attention mass on category corresponds to category coverage in image; >1: attention mass exceeds category coverage)
    """

    # get attention and coverage scores per category
    att_scores, gt_scores = get_panoptic_attention_map(
        att, image_id, panoptic_categories, panoptic_df, panoptic_img_dir, bbox
    )

    # normalize gt coverage (sum up to 1)
    norm_gt_scores = gt_scores / gt_scores.sum()

    # normalize and apply threshold
    below_threshold_mask = norm_gt_scores < gt_threshold
    # set classes in att_scores and gt_scores to 0 which fall below gt threshold
    att_scores[below_threshold_mask] = 0
    gt_scores[below_threshold_mask] = 0
    norm_gt_scores = (
        gt_scores / gt_scores.sum()
    )  # re-compute normalized gt coverage (to account for threshold)

    # normalize attentions (sum up to 1)
    if (
        att_scores.sum() > 0
    ):  # can be zero e.g. for context if all attention is on target
        norm_att_scores = att_scores / att_scores.sum()
    else:
        norm_att_scores = att_scores

    # normalize attention scores by class coverage
    cover_norm_att_scores = np.zeros_like(norm_att_scores, dtype=float)
    class_mask = gt_scores > 0  # mask to prevent division by zero error
    cover_norm_att_scores[class_mask] = (
        norm_att_scores[class_mask] / norm_gt_scores[class_mask]
    )

    # zip with category info
    n_categories = att_scores.shape[0]
    zipped_scores = zip(
        range(n_categories),
        cover_norm_att_scores,
        norm_att_scores,
        norm_gt_scores,
        gt_scores,
    )

    # filter out classes which do not occur or fall below threshold
    zipped_scores = [z for z in zipped_scores if z[-1] > 0]

    # merge with class annotations
    zipped_scores = [
        (norm_att_score, att_score, gt_score, panoptic_categories.get(idx, None))
        for idx, norm_att_score, att_score, gt_score, _ in zipped_scores
    ]

    if return_sorted:
        # sort by normalized attention score
        zipped_scores = sorted(zipped_scores, key=lambda x: x[0], reverse=True)

    return zipped_scores


def cat_ids_from_img(img_array, segments_info):
    pxl_ids = rgb2segid(img_array)
    cat_ids = segid2catid(pxl_ids, segments_info)
    return cat_ids


def filename_from_id(image_id, prefix="", file_ending=".jpg"):
    """
    get image filename from id: pad image ids with zeroes,
    add file prefix and file ending
    """
    padded_ids = str(image_id).rjust(12, "0")
    filename = prefix + padded_ids + file_ending

    return filename


def patches_from_bb(e, linewidth=2, edgecolor="g", facecolor="none"):
    bbox = e.bbox
    # Create a Rectangle patch
    return patches.Rectangle(
        (bbox[0], bbox[1]),
        bbox[2],
        bbox[3],
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    
    
def display_target(
    entry,
    coco_base,
    linewidth=2,
    target_color="g",
    facecolor="none",
    dpi="figure",
    save=False,
):
    # Create figure and axes
    _, ax = plt.subplots(figsize=(10, 10))

    # Retrieve & display the image

    image_file = filename_from_id(entry.image_id, prefix="COCO_train2014_")
    image_filepath = join(coco_base, "train2014", image_file)
    image = io.imread(image_filepath)
    ax.imshow(image)

    # Add the patch to the Axes
    ax.add_patch(
        patches_from_bb(
            entry, edgecolor=target_color, linewidth=linewidth, facecolor=facecolor
        )
    )

    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    if save:
        print(entry)
        filename = f"example_{entry.image_id}_{entry.name}.jpg"
        plt.savefig(filename, bbox_inches="tight", dpi=dpi)
        print(f"saved to {filename}")

    plt.show()


def get_context_classes(
    entry, panoptic_categories, panoptic_df, panoptic_img_dir, norm_counts=True
):
    pixel_id_map = imgid2catid(entry.image_id, panoptic_df, panoptic_img_dir)

    mask = np.ones_like(pixel_id_map, dtype=bool)
    x_min, x_max, y_min, y_max = convert_bbox_format(entry.bbox)
    mask[y_min:y_max, x_min:x_max] = False

    target_map = pixel_id_map[y_min:y_max, x_min:x_max]
    context_map = pixel_id_map.copy()
    context_map[~mask] = 0

    context_ids, context_pixel_counts = np.unique(
        pixel_id_map[mask], return_counts=True
    )

    if norm_counts:
        context_pixel_counts = context_pixel_counts / context_pixel_counts.sum()

    context_categories = list(
        zip([panoptic_categories[i] for i in context_ids], context_pixel_counts)
    )
    context_categories = sorted(context_categories, key=lambda x: x[1], reverse=True)

    return context_categories, (pixel_id_map, target_map, context_map)


def target_category_score(
    entry,
    panoptic_categories,
    panoptic_df,
    panoptic_img_dir
    ):
    context_categories, _ = get_context_classes(
        entry,
        panoptic_categories=panoptic_categories,
        panoptic_df=panoptic_df,
        panoptic_img_dir=panoptic_img_dir,
        norm_counts=True
    )

    context_category_scores = {c["id"]: score for c, score in context_categories}
    target_score = context_category_scores.get(entry.category_id, 0.0)

    return target_score


def decompose_att_vector(att, vis_dim=196):
    """split an attention vector in target/location/context (on first dimension)"""
    t_att = att[:vis_dim]
    l_att = att[vis_dim:-vis_dim]
    c_att = att[-vis_dim:]
    
    assert len(t_att) + len(l_att) + len(c_att) == len(att)
    
    return t_att, l_att, c_att


def reshape_2d_att(att, wh_dim=14):
    """reshape a 1d attention vector into a 2d attention vector"""
    assert att.shape[-1] == wh_dim**2
    return att.reshape(14,14)