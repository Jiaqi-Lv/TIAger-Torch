import json
import os

import numpy as np
import skimage
import skimage.measure
import torch
from tiatoolbox.models.engine.semantic_segmentor import SemanticSegmentor
from tiatoolbox.tools.patchextraction import get_patch_extractor
from tiatoolbox.wsicore.wsireader import VirtualWSIReader, WSIReader
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils import get_det_models, imagenet_normalise, is_l1, px_to_mm


output_dir = "/home/u1910100/cloud_workspace/GitHub/TIAger-Torch/output"
wsi_dir = "/home/u1910100/lab-private/it-services/TiGER/new_data/wsitils/images/"
temp_out_dir = os.path.join(output_dir, "temp_out/")
seg_out_dir = os.path.join(output_dir, "seg_out/")
det_out_dir = os.path.join(output_dir, "det_out/")

def detections_in_tile(image_tile, det_models):
    patch_size = 128
    overlap = 28
    tile_reader = VirtualWSIReader(image_tile, power=20)

    patch_extractor = get_patch_extractor(
        input_img=tile_reader,
        method_name="slidingwindow",
        patch_size=(patch_size, patch_size),
        stride=(overlap, overlap),
        resolution=20,
        units="power",
    )

    predictions = []
    batch_size = 128

    dataloader = DataLoader(patch_extractor, batch_size=batch_size, shuffle=False)

    for i, imgs in enumerate(tqdm(dataloader, leave=False)):
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


def tile_detection_stats(predictions, coordinate_list, x, y):
    tile_prediction = SemanticSegmentor.merge_prediction(
        (1024, 1024), predictions, coordinate_list
    )
    threshold = 0.99
    tile_prediction_mask = tile_prediction >= threshold

    mask_label = skimage.measure.label(tile_prediction_mask)

    stats = skimage.measure.regionprops(mask_label, intensity_image=tile_prediction)
    output_points = []
    annotations = []
    for region in stats:
        centroid = np.round(region["centroid"]).astype(int)

        c, r, confidence = (
            np.round(centroid[1]),
            np.round(centroid[0]),
            region["mean_intensity"],
        )

        c1 = c + x
        r1 = r + y
        prediction_record = {
            "point": [
                float(px_to_mm(c1, 0.5)),
                float(px_to_mm(r1, 0.5)),
                float(0.5009999871253967),
            ],
            "probability": float(confidence),
        }

        output_points.append(prediction_record)
        annotations.append((int(c1), int(r1)))
    return annotations, output_points


def detection_process(wsi_name):
    
    wsi_without_ext = os.path.splitext(wsi_name)[0]
    wsi_path = os.path.join(wsi_dir, wsi_name)
    print(f"Processing {wsi_path}")

    output_path = os.path.join(det_out_dir, f"{wsi_without_ext}_points.json")
    if os.path.exists(output_path):
        print("Already processed")
        return 1
    
    mask_path = os.path.join(temp_out_dir, f"{wsi_without_ext}.npy")
    mask = np.load(mask_path)[:,:,0]

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

    for i, tile in enumerate(tqdm(tile_extractor, leave=False)):
        bounding_box = tile_extractor.coordinate_list[
            i
        ]  # (x_start, y_start, x_end, y_end)
        predictions, coordinates = detections_in_tile(tile, models)
        annotations_tile, output_points_tile = tile_detection_stats(
            predictions, coordinates, bounding_box[0], bounding_box[1]
        )
        annotations.extend(annotations_tile)
        # output_dict["points"].extend(output_points_tile)

    # output_path = (
    #     os.path.join(det_out_dir, f"{wsi_without_ext}.json"
    # ))
    # with open(output_path, "w") as fp:
    #     json.dump(output_dict, fp, indent=4)

    output_path = os.path.join(det_out_dir, f"{wsi_without_ext}_points.json")
    with open(output_path, "w") as fp:
        json.dump(annotations, fp, indent=4)

    print("Detection mask saved")


if __name__ == "__main__":
    # wsi_name = "244B.tif"
    # detection_process(wsi_name)
    
    start = 62
    print(f"indices {start} - end")
    wsi_name_list = sorted(os.listdir(wsi_dir))
    wsi_name_list = wsi_name_list[start:]
    for wsi_name in tqdm(wsi_name_list, leave=True):
        detection_process(wsi_name=wsi_name)

