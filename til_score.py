import json
import os
from multiprocessing import Pool

import numpy as np
from tiatoolbox.wsicore.wsireader import VirtualWSIReader
from tissue_masker_lite import get_mask
from tqdm.auto import tqdm

from utils import create_til_score, is_l1


def til_score_process(wsi_name):
    output_dir = "output/"
    temp_out_dir = os.path.join(output_dir, "temp_out")
    wsi_without_ext = os.path.splitext(wsi_name)[0]
    print(f"Scoring {wsi_without_ext}")
    cell_points_dir = "/home/u1910100/GitHub/TIAger-Torch/output/det_out"
    cell_points_path = os.path.join(cell_points_dir, f"{wsi_without_ext}_points.json")
    wsi_dir = "/home/u1910100/Documents/Tiger_Data/wsitils/images/"
    wsi_path = os.path.join(wsi_dir, wsi_name)

    if not os.path.exists(cell_points_path):
        print("No detection map found")
        return 0

    stroma_bulk_mask_path = f"/home/u1910100/GitHub/TIAger-Torch/output/seg_out/{wsi_without_ext}_stroma_bulk.npy"
    tumor_bulk_mask_path = f"/home/u1910100/GitHub/TIAger-Torch/output/seg_out/{wsi_without_ext}_tumor_bulk.npy"
    bulk_mask_path = (
        f"/home/u1910100/GitHub/TIAger-Torch/output/seg_out/{wsi_without_ext}_bulk.npy"
    )
    mask_path = (
        f"/home/u1910100/GitHub/TIAger-Torch/output/temp_out/{wsi_without_ext}.npy"
    )

    tissue_mask = np.load(mask_path)

    if is_l1(tissue_mask):
        mask_reader = VirtualWSIReader.open(tissue_mask, mpp=8, power=1.25, mode="bool")
        mask = mask_reader.slide_thumbnail(resolution=32, units="mpp")[:, :, 0]
    else:
        mask = np.load(stroma_bulk_mask_path)
        # tumor_mask = np.load(tumor_bulk_mask_path)
        # mask[tumor_mask==1] = 1
        # mask = np.load(bulk_mask_path)
        # if np.count_nonzero(mask) <= 0:
        #     mask = tissue_mask

    score = create_til_score(wsi_path, cell_points_path, mask)

    output_tils_dir = os.path.join(output_dir, f"tils/")
    output_tils_path = os.path.join(output_tils_dir, f"{wsi_without_ext}.txt")
    with open(output_tils_path, "w") as file:
        file.write(str(score))
    print("TIL score saved")
    return score


if __name__ == "__main__":
    # wsi_name = "104S.tif"
    # score = til_score_process(wsi_name)
    # print(score)

    wsi_name_list = os.listdir("/home/u1910100/Documents/Tiger_Data/wsitils/images")
    with Pool(10) as p:
        p.map(til_score_process, wsi_name_list)
