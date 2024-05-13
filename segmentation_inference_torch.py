import os
import numpy as np
import torch
from tiatoolbox.models.engine.semantic_segmentor import SemanticSegmentor
from tiatoolbox.tools.patchextraction import get_patch_extractor
from tiatoolbox.wsicore.wsireader import VirtualWSIReader, WSIReader, TIFFWSIReader
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils import (get_seg_models, get_tumor_bulk, get_tumor_stroma_mask, is_l1, imagenet_normalise)



output_dir = "/home/u1910100/cloud_workspace/GitHub/TIAger-Torch/output"
wsi_dir = "/home/u1910100/lab-private/it-services/TiGER/new_data/wsitils/images/"
temp_out_dir = os.path.join(output_dir, "temp_out/")
seg_out_dir = os.path.join(output_dir, "seg_out/")
det_out_dir = os.path.join(output_dir, "det_out/")


def tumor_stroma_segmentation(wsi_path, mask, models):
    wsi_without_ext = os.path.splitext(os.path.basename(wsi_path))[0]
    image = TIFFWSIReader.open(wsi_path)
    dimensions = image.slide_dimensions(resolution=10, units="power")

    patch_extractor = get_patch_extractor(
        input_img=image,
        method_name="slidingwindow",
        patch_size=(512, 512),
        stride=(384, 384),
        resolution=10,
        units="power",
        input_mask=mask,
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

        val_predicts = np.zeros(shape=(imgs.size()[0], 3, 512, 512), dtype=np.float32)
        with torch.no_grad():
            for seg_model in models:
                temp_out = seg_model(imgs)
                temp_out = torch.nn.functional.softmax(temp_out, dim=1)
                temp_out = temp_out.detach().cpu().numpy()

                val_predicts += temp_out

        pred = np.argmax(val_predicts, axis=1, keepdims=True)
        pred = pred[:, 0, :, :].astype(np.uint8)
        tumor_map = np.zeros((imgs.size()[0], 512, 512), dtype=np.uint8)
        stroma_map = np.zeros((imgs.size()[0], 512, 512), dtype=np.uint8)
        # 1 -> tumor, 2-> stroma
        tumor_map[np.where(pred == 1)] = 1
        stroma_map[np.where(pred == 2)] = 1
        tumor_predictions.extend(list(tumor_map))
        stroma_predictions.extend(list(stroma_map))

    tumor_mask = SemanticSegmentor.merge_prediction(
        (dimensions[1], dimensions[0]),
        tumor_predictions,
        patch_extractor.coordinate_list,
    )
    tumor_mask = (tumor_mask > 0.5).astype(np.uint8)
    tumor_mask_reader = VirtualWSIReader.open(tumor_mask, power=10, mode='bool')
    tumor_mask = tumor_mask_reader.slide_thumbnail(resolution=0.3125, units="power")
    out_path = os.path.join(seg_out_dir, f"{wsi_without_ext}_tumor.npy")
    np.save(out_path, tumor_mask[:,:,0])

    stroma_mask = SemanticSegmentor.merge_prediction(
        (dimensions[1], dimensions[0]),
        stroma_predictions,
        patch_extractor.coordinate_list,
    )
    stroma_mask = (stroma_mask > 0.5).astype(np.uint8)
    stroma_mask_reader = VirtualWSIReader.open(stroma_mask, power=10, mode="bool")
    stroma_mask = stroma_mask_reader.slide_thumbnail(resolution=0.3125, units="power")
    out_path = os.path.join(seg_out_dir, f"{wsi_without_ext}_stroma.npy")
    np.save(out_path, stroma_mask[:,:,0])

    print(f"{wsi_without_ext} tumor stroma segmentation complete")


def generate_bulk_tumor_stroma(wsi_without_ext):
    tumor_mask_path = os.path.join(seg_out_dir, f"{wsi_without_ext}_tumor.npy")
    stroma_mask_path = os.path.join(seg_out_dir, f"{wsi_without_ext}_stroma.npy")
    tumor_mask = np.load(tumor_mask_path)
    tumor_bulk = get_tumor_bulk(tumor_mask)

    stroma_mask = np.load(stroma_mask_path)
    tumor_stroma_mask = get_tumor_stroma_mask(tumor_bulk, stroma_mask)
    tumor_stroma_mask = tumor_stroma_mask.astype(np.uint8)
    out_path = os.path.join(seg_out_dir, f"{wsi_without_ext}_tumor_stroma.npy")
    np.save(out_path, tumor_stroma_mask)


def tumor_stroma_process(wsi_name):
    
    if not os.path.exists(seg_out_dir):
        os.makedirs(seg_out_dir)
    if not os.path.exists(det_out_dir):
        os.makedirs(det_out_dir)

    wsi_path = os.path.join(wsi_dir, wsi_name)
    wsi_without_ext = os.path.splitext(wsi_name)[0]

    print(f"Processing {wsi_without_ext}")

    seg_result_path = os.path.join(seg_out_dir, f"{wsi_without_ext}_tumor_stroma.npy")
    if os.path.isfile(seg_result_path):
        print(f"{wsi_without_ext} already processed")
        return 1
    
    print("Loading tissue mask")
    mask_path = os.path.join(temp_out_dir, f"{wsi_without_ext}.npy")
    mask = np.load(mask_path)[:,:,0]

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
    # wsi_name = "TC_S01_P000004_C0001_B103.tif"
    wsi_name_list = os.listdir(wsi_dir)
    for wsi_name in tqdm(wsi_name_list):
        tumor_stroma_process(wsi_name)
   
    # tumor_stroma_process(wsi_name)

