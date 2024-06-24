import logging
import os
from typing import Union

import cv2
import numpy as np
import torch
from tiatoolbox.models.engine.semantic_segmentor import SemanticSegmentor
from tiatoolbox.tools.patchextraction import get_patch_extractor
from tiatoolbox.wsicore.wsireader import WSIReader
from torch.utils.data import DataLoader
from torchvision.transforms.v2.functional import resize
from tqdm.auto import tqdm

if logging.getLogger().hasHandlers():
    logging.getLogger().handlers.clear()
from tiatoolbox import logger

from config import Challenge_Config, Config
from utils import (
    calc_ratio,
    collate_fn,
    convert_tissue_masks_for_l1,
    dist_to_px,
    get_bulk,
    get_mask_with_asap,
    get_mpp_from_level,
    get_seg_models,
    imagenet_normalise,
    is_l1,
)


def process_patches(
    dataloader: DataLoader, models: list[torch.nn.Module]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    tumor_predictions: list[np.ndarray] = []
    stroma_predictions: list[np.ndarray] = []

    output_pred: list[np.ndarray] = []

    for imgs in tqdm(dataloader, leave=False):
        imgs = torch.permute(imgs, (0, 3, 1, 2))
        imgs = imgs / 255
        imgs = imagenet_normalise(imgs)
        imgs = imgs.to("cuda").float()

        val_predicts = torch.zeros(
            size=(imgs.size()[0], 3, 512, 512), device="cuda", dtype=float
        )

        with torch.no_grad():
            for seg_model in models:
                temp_out = seg_model(imgs)
                temp_out = torch.nn.functional.softmax(temp_out, dim=1)
                val_predicts += temp_out

            val_predicts = val_predicts / len(models)
            val_predicts = val_predicts.detach().cpu().numpy()

        pred = np.argmax(val_predicts, axis=1, keepdims=True)
        pred = pred[:, 0, :, :].astype(np.uint8)
        output_pred.append(pred)

    for batch_output in output_pred:
        tumor_map = np.zeros((batch_output.shape[0], 512, 512), dtype=np.uint8)
        stroma_map = np.zeros(
            (batch_output.shape[0], 512, 512), dtype=np.uint8
        )
        tumor_map[np.where(batch_output == 1)] = 1
        stroma_map[np.where(batch_output == 2)] = 1

        for i in range(imgs.size()[0]):
            down_tumor_map = cv2.resize(
                tumor_map[i], (256, 256), interpolation=cv2.INTER_NEAREST
            ).astype("uint8")
            down_stroma_map = cv2.resize(
                stroma_map[i], (256, 256), interpolation=cv2.INTER_NEAREST
            ).astype("uint8")

            tumor_predictions.append(down_tumor_map)
            stroma_predictions.append(down_stroma_map)

    return tumor_predictions, stroma_predictions


def merge_and_save_masks(
    predictions: list[np.ndarray],
    coords: list,
    dimensions: tuple[int, int],
    temp_out_dir: str,
    wsi_without_ext: str,
    mask_type: str,
) -> None:
    logger.info(f"Merging {mask_type} masks")
    if len(predictions) == 0:
        mask = np.zeros(shape=(dimensions[1], dimensions[0]), dtype=np.uint8)
    else:
        mask = SemanticSegmentor.merge_prediction(
            (dimensions[1], dimensions[0]), predictions, coords
        )
        mask = (mask > 0).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5)))

    out_path = os.path.join(temp_out_dir, f"{wsi_without_ext}_{mask_type}.npy")
    np.save(out_path, mask)
    logger.info(f"{mask_type} mask saved at {out_path}")


def tumor_stroma_segmentation(
    wsi_path: str,
    mask: np.ndarray,
    models: list[torch.nn.Module],
    IOConfig: Union[Config, Challenge_Config],
) -> None:
    temp_out_dir = IOConfig.temp_out_dir

    wsi_without_ext = os.path.splitext(os.path.basename(wsi_path))[0]
    image = WSIReader.open(wsi_path)

    _mpp = get_mpp_from_level(wsi_path, 1)  # mpp at level 1 == 10x power

    patch_extractor = get_patch_extractor(
        input_img=image,
        method_name="slidingwindow",
        patch_size=(512, 512),
        stride=(256, 256),
        resolution=_mpp,
        units="mpp",
        input_mask=mask,
    )

    tumor_predictions: list[np.ndarray] = []
    stroma_predictions: list[np.ndarray] = []

    batch_size = 32
    dataloader = DataLoader(
        patch_extractor,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    tumor_predictions, stroma_predictions = process_patches(dataloader, models)

    _mpp = get_mpp_from_level(wsi_path, 2)  # mpp at level 2 == 5x power
    down_dimensions = image.slide_dimensions(resolution=_mpp, units="mpp")
    down_coords = patch_extractor.coordinate_list / 2
    down_coords = np.round(down_coords).astype(int)

    merge_and_save_masks(
        tumor_predictions,
        down_coords,
        down_dimensions,
        temp_out_dir,
        wsi_without_ext,
        "tumor",
    )
    merge_and_save_masks(
        stroma_predictions,
        down_coords,
        down_dimensions,
        temp_out_dir,
        wsi_without_ext,
        "stroma",
    )

    logger.info(f"{wsi_without_ext} tumor stroma segmentation complete")
    return 1


def generate_bulk_tumor_stroma(
    wsi_without_ext: str, IOConfig: Union[Config, Challenge_Config]
):
    temp_out_dir = IOConfig.temp_out_dir
    seg_out_dir = IOConfig.seg_out_dir
    # These masks are saved at 5x
    tumor_mask_path = os.path.join(
        temp_out_dir, f"{wsi_without_ext}_tumor.npy"
    )
    stroma_mask_path = os.path.join(
        temp_out_dir, f"{wsi_without_ext}_stroma.npy"
    )
    try:
        stroma_mask = np.load(stroma_mask_path)
        tumor_mask = np.load(tumor_mask_path)
    except:
        logger.warning("Failed to load tumor mask or stroma mask")
        return 0

    # Down-sample masks to 0.3125x

    stroma_mask = cv2.resize(
        stroma_mask,
        None,
        fx=0.0625,
        fy=0.0625,
        interpolation=cv2.INTER_NEAREST,
    ).astype("uint8")
    tumor_mask = cv2.resize(
        tumor_mask, None, fx=0.0625, fy=0.0625, interpolation=cv2.INTER_NEAREST
    ).astype("uint8")

    ratio = calc_ratio(tumor_mask)
    if ratio < 0.1:
        kernel_diameter = dist_to_px(500, 32)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_diameter, kernel_diameter)
        )
        tumor_mask = cv2.dilate(tumor_mask, kernel, iterations=1)

    bulk_mask = get_bulk(tumor_mask)
    if np.count_nonzero(bulk_mask) > 0:
        tumor_bulk = tumor_mask * bulk_mask
        stroma_bulk = stroma_mask * bulk_mask
    else:
        tumor_bulk = tumor_mask
        stroma_bulk = stroma_mask

    np.save(
        os.path.join(seg_out_dir, f"{wsi_without_ext}_tumor_bulk.npy"),
        tumor_bulk,
    )
    np.save(
        os.path.join(seg_out_dir, f"{wsi_without_ext}_stroma_bulk.npy"),
        stroma_bulk,
    )
    np.save(
        os.path.join(seg_out_dir, f"{wsi_without_ext}_bulk.npy"), bulk_mask
    )
    return 1


def tumor_stroma_process_l1(
    wsi_name: str, mask_name: str, IOConfig: Challenge_Config
):
    """For TIGER Challenge Leaderboard 1"""
    input_dir = IOConfig.input_dir
    input_mask_dir = IOConfig.input_mask_dir
    temp_out_dir = IOConfig.temp_out_dir

    wsi_path = os.path.join(input_dir, wsi_name)
    wsi_reader = WSIReader.open(wsi_path)
    mpp_info = wsi_reader.info.as_dict()["mpp"]

    wsi_without_ext = os.path.splitext(wsi_name)[0]

    logger.info(f"Processing {wsi_name}")

    # Load tissue mask
    logger.info("Loading tissue mask")
    mask_path = os.path.join(input_mask_dir, mask_name)

    _mpp = get_mpp_from_level(mask_path, 2)  # mpp at level 2 == 5x power
    mask = get_mask_with_asap(mask_path=mask_path, mpp=_mpp)
    if np.count_nonzero(mask) == 0:
        logger.info("Input mask is blank")

    toolbox_dim = wsi_reader.slide_dimensions(_mpp, "mpp")
    mask = cv2.resize(
        mask,
        (toolbox_dim[0], toolbox_dim[1]),
        interpolation=cv2.INTER_NEAREST,
    ).astype("uint8")

    logger.info(f"asap mask at 5x: {mask.shape}")

    logger.info(f"tiatoolbox tissue dim at 5: {toolbox_dim}")

    models = get_seg_models(IOConfig)
    logger.info("Running tissue segmentation")
    tumor_stroma_segmentation(wsi_path, mask, models, IOConfig)

    logger.info("Tissue segmentation complete")

    tumor_mask_path = os.path.join(
        temp_out_dir, f"{wsi_without_ext}_tumor.npy"
    )
    stroma_mask_path = os.path.join(
        temp_out_dir, f"{wsi_without_ext}_stroma.npy"
    )
    convert_tissue_masks_for_l1(
        mask_path, tumor_mask_path, stroma_mask_path, IOConfig, mpp_info
    )
    logger.info("Segmentation mask saved as tif file")
    return 1


def tumor_stroma_process(wsi_name: str, IOConfig: Config):
    input_dir = IOConfig.input_dir
    input_mask_dir = IOConfig.input_mask_dir
    seg_out_dir = IOConfig.seg_out_dir

    wsi_path = os.path.join(input_dir, wsi_name)
    wsi_without_ext = os.path.splitext(wsi_name)[0]

    logger.info(f"Processing {wsi_without_ext}")

    # seg_result_path = os.path.join(
    #     seg_out_dir, f"{wsi_without_ext}_tumor_stroma.npy"
    # )
    # if os.path.isfile(seg_result_path):
    #     logger.info(f"{wsi_without_ext} already processed")
    #     return 1

    # Generate tissue mask
    logger.info("Loading tissue mask")
    mask_path = os.path.join(input_mask_dir, f"{wsi_without_ext}.npy")
    mask = np.load(mask_path)[:, :, 0]

    if is_l1(mask):
        logger.info(f"{wsi_without_ext} is L1")
        return 1
    else:
        models = get_seg_models(IOConfig)
        logger.info("Running tissue segmentation")
        tumor_stroma_segmentation(wsi_path, mask, models, IOConfig)

        # Generate tumor bulk
        logger.info("Generating bulk tumor stroma")
        generate_bulk_tumor_stroma(wsi_without_ext, IOConfig)

        logger.info("Tumor stroma mask saved")
        return 1


if __name__ == "__main__":
    # wsi_name_list = os.listdir(wsi_dir)
    # for wsi_name in tqdm(wsi_name_list):
    #     tumor_stroma_process(wsi_name)
    # IOConfig = Challenge_Config()
    # wsi_name = [
    #     x for x in os.listdir(IOConfig.input_dir) if x.endswith(".tif")
    # ][0]
    # mask_name = [
    #     x for x in os.listdir(IOConfig.input_mask_dir) if x.endswith(".tif")
    # ][0]
    # tumor_stroma_process_l1(wsi_name, mask_name, IOConfig)

    IOConfig = Config()
    IOConfig.create_output_dirs()
    wsi_name = "TC_S01_P000124_C0001_B101.tif"
    mask_name = "TC_S01_P000124_C0001_B101.npy"
    tumor_stroma_process(wsi_name=wsi_name, IOConfig=IOConfig)
