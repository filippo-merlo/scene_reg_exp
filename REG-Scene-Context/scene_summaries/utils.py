from __future__ import annotations

from os.path import join
import pandas as pd
import json
from PIL import Image
import numpy as np

##########################
# General Data Handling  #
##########################


def get_refcoco_data(path):
    """fetch data from RefCOCO*

    Args:
        path (string): path to RefCOCO* base dir

    Returns:
        tuple: RefCOCO* data (pd.DataFrame), split IDs (dict -> dict -> list)
    """
    annotations = get_refcoco_df(path)

    # partitions: ['train', 'testB', 'testA', 'val']
    partitions = list(pd.unique(annotations.refcoco_split))

    image_ids, refexp_ids = {}, {}

    for part in partitions:
        image_ids[part] = list(
            annotations.loc[annotations.refcoco_split == part].image_id.unique()
        )
        refexp_ids[part] = annotations.loc[
            annotations.refcoco_split == part
        ].index.to_list()

    split_ids = {"image_ids": image_ids, "refexp_ids": refexp_ids}

    return (annotations, split_ids)


def get_refcoco_df(path):
    """get RefCOCO* annotations as pd.DataFrame

    Args:
        path (string): path to RefCOCO* base dir

    Returns:
        pd.DataFrame: RefCOCO* annotations
    """
    filepath = join(path, "instances.json")
    with open(filepath) as file:
        instances = json.load(file)
        instances = pd.DataFrame(instances["annotations"]).set_index("id")

    ann_file = "refs(umd).p" if path.lower().endswith("refcocog") else "refs(unc).p"

    filepath = join(path, ann_file)
    annotations = pd.read_pickle(filepath)
    annotations = split_sentences(pd.DataFrame(annotations))

    annotations = pd.merge(
        annotations, instances[["image_id", "bbox"]], left_on="ann_id", right_on="id"
    ).set_index("sent_id")

    return annotations


def split_sentences(df):
    """split RefExp DataFrame by sentences

    Args:
        df (pd.DataFrame): DataFrame containing RefExp annotations

    Returns:
        pd.DataFrame: RefExp annotations with entries for individual sentences
    """
    rows = []

    def coco_split(row):
        for split in ["train", "val", "test"]:
            if split in row["file_name"]:
                return split
        return None

    def unstack_sentences(row):
        nonlocal rows
        for i in row.sentences:
            rows.append(
                {
                    "sent_id": i["sent_id"],
                    "ann_id": row["ann_id"],
                    "refexp": i["sent"],
                    "ref_id": row["ref_id"],
                    "refcoco_split": row["split"],
                    "coco_split": coco_split(row),
                }
            )

    df.apply(lambda x: unstack_sentences(x), axis=1)

    return pd.DataFrame(rows)


def import_data(config, collapse_by="ann_id") -> tuple[list[pd.DataFrame], list[str]]:

    ann_df, _ = get_refcoco_data(config.refcoco_base)

    if collapse_by is not None:
        index_column = ann_df.index.name
        ann_df = (
            ann_df.reset_index()
            .groupby(collapse_by)
            .first()
            .reset_index()
            .set_index(index_column)
        )

    train_data = ann_df.loc[ann_df.refcoco_split == "train"].to_dict(orient="records")
    val_data = ann_df.loc[ann_df.refcoco_split == "val"].to_dict(orient="records")
    testA_data = ann_df.loc[ann_df.refcoco_split == "testA"].to_dict(orient="records")
    testB_data = ann_df.loc[ann_df.refcoco_split == "testB"].to_dict(orient="records")

    dsets = [train_data, val_data, testA_data, testB_data]
    splits = ["train", "val", "testA", "testB"]

    return dsets, splits


def filename_from_id(image_id, prefix="COCO_train2014_", extension=".jpg"):
    """get image filename from COCO image ID

    Args:
        image_id (int): Image ID
        prefix (str, optional): Prefix for image filename. Defaults to 'COCO_train2014_'.
        extension (str, optional): File extension. Defaults to '.jpg'.

    Returns:
        str: Image filename
    """

    padded_ids = str(image_id).rjust(12, "0")
    filename = prefix + padded_ids + extension

    return filename


########################
# Panoptic Annotations #
########################


def read_panoptic_anns(ann_path):
    with open(ann_path, "r") as f:
        anns = json.load(f)

    ann_df = pd.DataFrame(anns["annotations"]).set_index("image_id")
    categories = {c["id"]: c for c in anns["categories"]}
    categories[0] = {"supercategory": "void", "isthing": 0, "id": 0, "name": "void"}

    return ann_df, categories


def rgb2segid(img_array):
    ids = img_array[:, :, 0] + img_array[:, :, 1] * 256 + img_array[:, :, 2] * 256**2
    return ids


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
        Image.open(join(img_dir, entry.file_name))
    )  # read image & convert to np array
    pxl_ids = rgb2segid(img_array)  # get segment ids
    cat_ids = segid2catid(pxl_ids, entry.segments_info)  # convert to category ids

    if return_img:
        return img_array, cat_ids
    return cat_ids
