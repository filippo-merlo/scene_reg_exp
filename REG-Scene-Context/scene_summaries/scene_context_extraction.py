import argparse
import os

from config import Config
from utils import import_data
import os.path as osp
import h5py
from tqdm import tqdm
from SceneExtractor import SceneExtractor


def extract_dataset(input_data, out_path, extractor, info) -> None:
    """
    extract scene summaries from refcoco splits and store in hdf5 file

    Args:
        input_data (pd.DataFrame): DataFrame containing information for the current split
        out_path (str): path for output file
        extractor (SceneExtractor): scene extractor object
    """

    n_items = len(input_data)
    segment_classes = list(extractor.categories_map.values())

    with h5py.File(out_path, "w") as f:

        f.create_dataset("ann_ids", (n_items, 1), dtype=int)
        f.create_dataset("full_feats", (n_items, len(segment_classes)))
        f.create_dataset("context_feats", (n_items, len(segment_classes)))
        f.create_dataset("weighted_context_feats", (n_items, len(segment_classes)))

        print(
            f"Create output file {out_path} with datasets: ",
            "ann_ids",
            "full_feats",
            "context_feats",
            "weighted_context_feats",
            sep="\n",
        )

        # write metadata
        f.attrs["split"] = info.get("split")
        f.attrs["segment_classes"] = segment_classes

    with h5py.File(out_path, "a") as f:
        ids_dset = f["ann_ids"]
        full_features_dset = f["full_feats"]
        context_features_dset = f["context_feats"]
        weighted_context_features_dset = f["weighted_context_feats"]

        for idx, entry in tqdm(enumerate(input_data), total=len(input_data)):

            ann_id = entry["ann_id"]
            image_id = entry["image_id"]
            pixel_ids = extractor.pixel_ids_from_image_id(image_id)

            # extract full summary
            full_summary = extractor.extract_summary_from_pixel_ids(
                pixel_ids, exclude_mask_id=None, weights=None
            )

            # extract context summary
            masked_pixel_ids = extractor.mask_target(
                pixel_ids, entry["bbox"], mask_id=-1
            )
            context_summary = extractor.extract_summary_from_pixel_ids(
                masked_pixel_ids, exclude_mask_id=-1
            )

            # extract weighted context summary
            bb_center = extractor.bounding_box_center(entry["bbox"])
            proximity_map = extractor.get_distance_map(
                pixel_ids=pixel_ids,
                point=bb_center,
                normalize=True,
                return_proximity=True,
            )

            weighted_context_summary = extractor.extract_summary_from_pixel_ids(
                masked_pixel_ids, exclude_mask_id=-1, weights=proximity_map
            )

            # write to file
            ids_dset[idx] = ann_id
            full_features_dset[idx] = full_summary
            context_features_dset[idx] = context_summary
            weighted_context_features_dset[idx] = weighted_context_summary

def main(args, config) -> None:
    """
    initialize extractor, read data and extract scene summaries

    Args:
        args (argparse.Namespace): script arguments
        config (Config): config object
    """

    # initialize extractor
    scene_extractor = SceneExtractor(config)

    # read data
    print("read data")
    dsets, splits = import_data(config)

    if not osp.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    # prepare list of datasets to be extracted
    zipped_dsets = zip(dsets, splits)
    if args.exclude_train and not args.extract_split:
        # all but train set
        print("exclude train split")
        zipped_dsets = [
            (dset, split) for dset, split in zipped_dsets if split != "train"
        ]
    elif args.extract_split:
        # only specific split
        print(f"restrict to {args.extract_split} split")
        zipped_dsets = [
            (dset, split) for dset, split in zipped_dsets if split == args.extract_split
        ]
    else:
        # all splits
        zipped_dsets = list(zipped_dsets)

    for i, (dset, split) in enumerate(zipped_dsets):
        print(
            f"({i+1} / {len(zipped_dsets)}) starting summary extraction for {split} split..."
        )
        out_path = osp.join(args.out_dir, f"scene_summaries_{config.extraction_type}_{split}.h5")

        info = {"split": split}

        # extract current dataset
        extract_dataset(dset, out_path, scene_extractor, info)


if __name__ == "__main__":

    config = Config()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cpu", "cuda"],
    )

    parser.add_argument(
        "--out_dir",
        default=config.features_out,
        type=osp.abspath,
    )

    parser.add_argument("--exclude_train", action="store_true")

    parser.add_argument(
        "--extract_split",
        default=None,
        type=str.lower,
        choices=["train", "val", "testa", "testb"],
    )

    args = parser.parse_args()
    print(vars(args))

    main(args, config)
