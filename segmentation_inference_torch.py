import os

import cv2
import numpy as np
import torch
from tiatoolbox.models.engine.semantic_segmentor import SemanticSegmentor
from tiatoolbox.tools.patchextraction import get_patch_extractor
from tiatoolbox.wsicore.wsireader import VirtualWSIReader, WSIReader
from torch.utils.data import DataLoader
from torchvision.transforms.v2.functional import resize
from tqdm.auto import tqdm

from config import ChallengeConfig, DefaultConfig
from utils import (calc_ratio, convert_tissue_masks_for_l1, dist_to_px,
                   get_bulk, get_seg_models, imagenet_normalise, is_l1)

config = ChallengeConfig

output_dir = config.output_dir
wsi_dir = config.wsi_dir
temp_out_dir = config.temp_out_dir
seg_out_dir = config.seg_out_dir
det_out_dir = config.det_out_dir


def tumor_stroma_segmentation(wsi_path, mask, models):
    wsi_without_ext = os.path.splitext(os.path.basename(wsi_path))[0]
    image = WSIReader.open(wsi_path)

    patch_extractor = get_patch_extractor(
        input_img=image,
        method_name="slidingwindow",
        patch_size=(512, 512),
        stride=(384, 384),
        resolution=10,
        units="power",
        input_mask=mask,
        min_mask_ratio=0.3,
    )

    tumor_predictions: list[np.ndarray] = []
    stroma_predictions: list[np.ndarray] = []

    batch_size = 16
    dataloader = DataLoader(patch_extractor, batch_size=batch_size, shuffle=False)

    for i, imgs in enumerate(tqdm(dataloader, leave=False)):
        imgs = torch.permute(imgs, (0, 3, 1, 2))
        imgs = imgs / 255
        imgs = imagenet_normalise(imgs)
        imgs = imgs.to("cuda").float()

        val_predicts = np.zeros(shape=(imgs.size()[0], 3, 256, 256), dtype=np.float32)
        with torch.no_grad():
            for seg_model in models:
                temp_out = seg_model(imgs)
                temp_out = torch.nn.functional.softmax(temp_out, dim=1)
                temp_out = resize(temp_out, (256, 256), antialias=False)
                temp_out = temp_out.detach().cpu().numpy()

                val_predicts += temp_out

        pred = np.argmax(val_predicts, axis=1, keepdims=True)
        pred = pred[:, 0, :, :].astype(np.uint8)
        tumor_map = np.zeros((imgs.size()[0], 256, 256), dtype=np.uint8)
        stroma_map = np.zeros((imgs.size()[0], 256, 256), dtype=np.uint8)
        # 1 -> tumor, 2-> stroma
        tumor_map[np.where(pred == 1)] = 1
        stroma_map[np.where(pred == 2)] = 1
        tumor_predictions.extend(list(tumor_map))
        stroma_predictions.extend(list(stroma_map))

    print("Merging tumor masks")

    down_dimensions = image.slide_dimensions(resolution=5, units="power")
    down_coords = [coords // 2 for coords in patch_extractor.coordinate_list]

    tumor_mask = SemanticSegmentor.merge_prediction(
        (down_dimensions[1], down_dimensions[0]),
        tumor_predictions,
        down_coords,
    )
    tumor_mask = (tumor_mask > 0.5).astype(np.uint8)
    out_path = os.path.join(seg_out_dir, f"{wsi_without_ext}_tumor.npy")
    np.save(out_path, tumor_mask[:, :, 0])

    print("Merging stroma masks")
    stroma_mask = SemanticSegmentor.merge_prediction(
        (down_dimensions[1], down_dimensions[0]),
        stroma_predictions,
        down_coords,
    )
    stroma_mask = (stroma_mask > 0.5).astype(np.uint8)
    out_path = os.path.join(seg_out_dir, f"{wsi_without_ext}_stroma.npy")
    np.save(out_path, stroma_mask[:, :, 0])

    print(f"{wsi_without_ext} tumor stroma segmentation complete")
    return 1


def generate_bulk_tumor_stroma(wsi_without_ext):
    tumor_mask_path = os.path.join(seg_out_dir, f"{wsi_without_ext}_tumor.npy")
    stroma_mask_path = os.path.join(seg_out_dir, f"{wsi_without_ext}_stroma.npy")
    try:
        stroma_mask = np.load(stroma_mask_path)
        tumor_mask = np.load(tumor_mask_path)
    except:
        print("Failed to load tumor mask or stroma mask")
        return 0

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
    np.save(os.path.join(seg_out_dir, f"{wsi_without_ext}_bulk.npy"), bulk_mask)
    return 1


def tumor_stroma_process_l1(wsi_name):
    """For TIGER Challenge Leaderboard 1"""

    wsi_path = os.path.join(wsi_dir, wsi_name)
    wsi_without_ext = os.path.splitext(wsi_name)[0]

    print(f"Processing {wsi_without_ext}")

    seg_result_path = os.path.join(seg_out_dir, f"{wsi_without_ext}.tif")
    if os.path.isfile(seg_result_path):
        print(f"{wsi_without_ext} already processed")
        return 1

    # Load tissue mask
    print("Loading tissue mask")
    mask_path = os.path.join(temp_out_dir, f"{wsi_without_ext}_tissue.tif")
    mask_reader = WSIReader.open(mask_path)
    mask = mask_reader.slide_thumbnail(resolution=0.3125, units="power")[:, :, 0]
    # print("Loading tissue mask")
    # mask_path = os.path.join(temp_out_dir, f"{wsi_without_ext}.npy")
    # mask = np.load(mask_path)[:, :, 0]

    models = get_seg_models()
    print("Running tissue segmentation")
    tumor_stroma_segmentation(wsi_path, mask, models)

    print("Tissue segmentation complete")

    tumor_mask_path = os.path.join(seg_out_dir, f"{wsi_without_ext}_tumor.npy")
    stroma_mask_path = os.path.join(seg_out_dir, f"{wsi_without_ext}_stroma.npy")
    convert_tissue_masks_for_l1(wsi_without_ext, tumor_mask_path, stroma_mask_path)
    print("Segmentation mask saved as tif file")
    os.remove(tumor_mask_path)
    os.remove(stroma_mask_path)
    return 1


def tumor_stroma_process(wsi_name):
    if not os.path.exists(seg_out_dir):
        os.makedirs(seg_out_dir)

    wsi_path = os.path.join(wsi_dir, wsi_name)
    wsi_without_ext = os.path.splitext(wsi_name)[0]

    print(f"Processing {wsi_without_ext}")

    seg_result_path = os.path.join(seg_out_dir, f"{wsi_without_ext}_tumor_stroma.npy")
    if os.path.isfile(seg_result_path):
        print(f"{wsi_without_ext} already processed")
        return 1

    # Generate tissue mask
    print("Loading tissue mask")
    mask_path = os.path.join(temp_out_dir, f"{wsi_without_ext}.npy")
    mask = np.load(mask_path)[:, :, 0]

    if is_l1(mask):
        print(f"{wsi_without_ext} is L1")
        return 1
    else:
        models = get_seg_models()
        print("Running tissue segmentation")
        tumor_stroma_segmentation(wsi_path, mask, models)

        # Generate tumor bulk
        print("Generating bulk tumor stroma")
        generate_bulk_tumor_stroma(wsi_without_ext)

        print("Tumor stroma mask saved")
        return 1


if __name__ == "__main__":
    # wsi_name_list = os.listdir(wsi_dir)
    # for wsi_name in tqdm(wsi_name_list):
    #     tumor_stroma_process(wsi_name)

    wsi_name = "104S.tif"
    tumor_stroma_process_l1(wsi_name=wsi_name)
