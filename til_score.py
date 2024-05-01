import json
import os
from multiprocessing import Pool

import numpy as np
from tiatoolbox.wsicore.wsireader import VirtualWSIReader
from tqdm.auto import tqdm

from utils import create_til_score, is_l1


def til_score_process(wsi_name):
    output_dir = "output/"
    wsi_without_ext = os.path.splitext(wsi_name)[0]
    print(f"Scoring {wsi_without_ext}")
    cell_points_dir = "/home/u1910100/GitHub/TIAger-Torch/output/det_out"
    cell_points_path = os.path.join(cell_points_dir, f"{wsi_without_ext}_points.json")
    wsi_dir = "/home/u1910100/Documents/Tiger_Data/wsitils/images/"
    wsi_path = os.path.join(wsi_dir, wsi_name)

    tumor_stroma_mask_path = f"/home/u1910100/GitHub/TIAger-Torch/output/seg_out/{wsi_without_ext}_tumor_stroma.npy"
    mask_path = (
        f"/home/u1910100/GitHub/TIAger-Torch/output/temp_out/{wsi_without_ext}.npy"
    )

    tissue_mask = np.load(mask_path)
    if is_l1(tissue_mask):
        mask = np.load(mask_path)
        mask_reader = VirtualWSIReader.open(mask, mpp=8, power=1.25, mode="bool")
        mask = mask_reader.slide_thumbnail(resolution=32, units="mpp")
    else:
        mask = np.load(tumor_stroma_mask_path)

    score = create_til_score(wsi_path, cell_points_path, mask)

    output_tils_dir = os.path.join(output_dir, f"tils/")
    output_tils_path = os.path.join(output_tils_dir, f"{wsi_without_ext}.txt")
    with open(output_tils_path, "w") as file:
        file.write(str(score))
    return score


if __name__ == "__main__":
    # wsi_name = "104S.tif"
    # score = til_score_process(wsi_name)
    # print(score)

    wsi_name_list = os.listdir("/home/u1910100/Documents/Tiger_Data/wsitils/images")
    with Pool(2) as p:
        p.map(til_score_process, wsi_name_list)
