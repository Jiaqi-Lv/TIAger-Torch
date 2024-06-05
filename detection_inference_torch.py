import json
import os
import shutil
from multiprocessing import Pool

import numpy as np
import skimage
import skimage.measure
import torch
from tiatoolbox.models.engine.semantic_segmentor import SemanticSegmentor
from tiatoolbox.tools.patchextraction import get_patch_extractor
from tiatoolbox.wsicore.wsireader import VirtualWSIReader, WSIReader
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import Challenge_Config, Config
from utils import (
    check_coord_in_mask,
    collate_fn,
    get_det_models,
    get_mask_with_asap,
    get_mpp_from_level,
    imagenet_normalise,
    is_l1,
    px_to_mm,
)


def detections_in_tile(image_tile, det_models):
    patch_size = 128
    stride = 100
    tile_reader = VirtualWSIReader.open(image_tile)

    patch_extractor = get_patch_extractor(
        input_img=tile_reader,
        method_name="slidingwindow",
        patch_size=(patch_size, patch_size),
        stride=(stride, stride),
        resolution=0,
        units="level",
    )

    predictions = []
    batch_size = 32

    dataloader = DataLoader(
        patch_extractor, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    for i, imgs in enumerate(dataloader):
        imgs = torch.permute(imgs, (0, 3, 1, 2))
        imgs = imgs / 255
        imgs = imagenet_normalise(imgs)
        imgs = imgs.to("cuda").float()

        val_predicts = np.zeros(shape=(imgs.size()[0], 128, 128), dtype=np.float32)

        with torch.no_grad():
            for det_model in det_models:
                temp_out = det_model(imgs)
                temp_out = torch.sigmoid(temp_out)
                temp_out = temp_out.detach().cpu().numpy()[:, 0, :, :]

                val_predicts += temp_out

        val_predicts = val_predicts / 3
        predictions.extend(list(val_predicts))

    return predictions, patch_extractor.coordinate_list


def tile_detection_stats(predictions, coordinate_list, x, y, mpp, tissue_mask=None):
    tile_prediction = SemanticSegmentor.merge_prediction(
        (1024, 1024), predictions, coordinate_list
    )
    threshold = 0.99
    tile_prediction_mask = tile_prediction > threshold

    mask_label = skimage.measure.label(tile_prediction_mask)

    stats = skimage.measure.regionprops(mask_label, intensity_image=tile_prediction)
    output_points = []
    annotations = []
    for region in stats:
        centroid = region["centroid"]

        c, r, confidence = (
            centroid[1],
            centroid[0],
            region["mean_intensity"],
        )

        c1 = c + x
        r1 = r + y

        if not check_coord_in_mask(round(c1 / 4), round(r1 / 4), tissue_mask):
            continue

        prediction_record = {
            "point": [
                float(px_to_mm(c1, mpp[0])),
                float(px_to_mm(r1, mpp[1])),
                float(0.5009999871253967),
            ],
            "probability": float(confidence),
        }

        output_points.append(prediction_record)
        annotations.append((np.round(c1), np.round(r1)))
    return annotations, output_points


def detection_process(wsi_name):
    wsi_without_ext = os.path.splitext(wsi_name)[0]
    wsi_path = os.path.join(wsi_dir, wsi_name)
    print(f"Processing {wsi_path}")

    mask_path = os.path.join(temp_out_dir, f"{wsi_without_ext}.npy")
    mask = np.load(mask_path)[:, :, 0]

    if is_l1(mask):
        print("L1")
        input_mask = mask
    else:
        print("L2")
        tumor_stroma_mask_path = os.path.join(
            seg_out_dir, f"{wsi_without_ext}_stroma_bulk.npy"
        )
        tumor_stroma_mask = np.load(tumor_stroma_mask_path)
        input_mask = tumor_stroma_mask

    models = get_det_models()

    wsi = WSIReader.open(wsi_path)

    tile_extractor = get_patch_extractor(
        input_img=wsi,
        method_name="slidingwindow",
        patch_size=(1024, 1024),
        resolution=20,
        units="power",
        input_mask=input_mask,
    )
    # Each tile of size 1024x1024
    annotations = []
    output_dict = {
        "type": "Multiple points",
        "version": {"major": 1, "minor": 0},
        "points": [],
    }

    for i, tile in enumerate(
        tqdm(tile_extractor, leave=False, desc=f"{wsi_without_ext} progress")
    ):
        bounding_box = tile_extractor.coordinate_list[
            i
        ]  # (x_start, y_start, x_end, y_end)
        predictions, coordinates = detections_in_tile(tile, models)
        annotations_tile, output_points_tile = tile_detection_stats(
            predictions, coordinates, bounding_box[0], bounding_box[1]
        )
        annotations.extend(annotations_tile)
        output_dict["points"].extend(output_points_tile)

    output_path = os.path.join(det_out_dir, f"{wsi_without_ext}.json")
    with open(output_path, "w") as fp:
        json.dump(output_dict, fp, indent=4)

    output_path = os.path.join(det_out_dir, f"{wsi_without_ext}_points.json")
    with open(output_path, "w") as fp:
        json.dump(annotations, fp, indent=4)

    print("Detection mask saved")


def detection_process_l1(wsi_name, mask_name, IOConfig):
    """For TIGER Challenge Leaderboard 1"""

    input_dir = IOConfig.input_dir
    input_mask_dir = IOConfig.input_mask_dir
    det_out_dir = IOConfig.det_out_dir
    temp_out_dir = IOConfig.temp_out_dir

    wsi_without_ext = os.path.splitext(wsi_name)[0]
    wsi_path = os.path.join(input_dir, wsi_name)
    print(f"Processing {wsi_path}")

    # Load tissue mask
    print("Loading tissue mask")
    mask_path = os.path.join(input_mask_dir, mask_name)
    # mask_reader = WSIReader.open(mask_path)
    # input_mask = mask_reader.slide_thumbnail(resolution=5, units="power")[:, :, 0]
    _mpp = get_mpp_from_level(mask_path, 2)  # mpp at level 2 == 5x power
    input_mask = get_mask_with_asap(mask_path=mask_path, mpp=_mpp)

    models = get_det_models(IOConfig)

    wsi = WSIReader.open(wsi_path)
    mpp_info = wsi.info.as_dict()["mpp"]

    tile_extractor = get_patch_extractor(
        input_img=wsi,
        method_name="slidingwindow",
        patch_size=(1024, 1024),
        resolution=0,
        units="level",
        input_mask=input_mask,
    )
    # Each tile of size 1024x1024
    annotations = []
    output_dict = {
        "type": "Multiple points",
        "version": {"major": 1, "minor": 0},
        "points": [],
    }

    for i, tile in enumerate(
        tqdm(tile_extractor, leave=False, desc=f"{wsi_without_ext} progress")
    ):
        bounding_box = tile_extractor.coordinate_list[
            i
        ]  # (x_start, y_start, x_end, y_end)
        predictions, coordinates = detections_in_tile(tile, models)
        annotations_tile, output_points_tile = tile_detection_stats(
            predictions,
            coordinates,
            bounding_box[0],
            bounding_box[1],
            mpp_info,
            input_mask,
        )
        annotations.extend(annotations_tile)
        output_dict["points"].extend(output_points_tile)

    with open(
        os.path.join(temp_out_dir, f"detected-lymphocytes.json"), "w", encoding="utf-8"
    ) as fp:
        json.dump(output_dict, fp, ensure_ascii=False, indent=4)

    with open(os.path.join(temp_out_dir, f"{wsi_without_ext}_points.json"), "w") as fp:
        json.dump(annotations, fp, indent=4)

    final_path = os.path.join(det_out_dir, f"detected-lymphocytes.json")
    shutil.copyfile(
        os.path.join(temp_out_dir, f"detected-lymphocytes.json"),
        final_path,
    )

    print(f"Detection saved at {final_path}")


if __name__ == "__main__":
    # wsi_name_list = os.listdir(wsi_dir)
    # with Pool(5) as p:
    #     list(
    #         tqdm(
    #             p.imap(detection_process, wsi_name_list, chunksize=15),
    #             total=len(wsi_name_list),
    #             desc="Multiprocessing Progress",
    #         )
    #     )

    wsi_name = "104S.tif"
    mask_name = "104S_tissue.tif"
    IOConfig = Challenge_Config()
    detection_process_l1(wsi_name, mask_name, IOConfig)
