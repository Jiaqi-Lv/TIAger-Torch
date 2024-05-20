import json
import os
from multiprocessing import Pool

import numpy as np
from tiatoolbox.wsicore.wsireader import VirtualWSIReader
from tissue_masker_lite import get_mask
from tqdm.auto import tqdm

from config import DefaultConfig
from utils import create_til_score, is_l1

output_dir = DefaultConfig.output_dir
wsi_dir = DefaultConfig.wsi_dir
temp_out_dir = DefaultConfig.temp_out_dir
seg_out_dir = DefaultConfig.seg_out_dir
det_out_dir = DefaultConfig.det_out_dir
output_tils_dir = DefaultConfig.output_tils_dir


def til_score_process(wsi_name):
    wsi_without_ext = os.path.splitext(wsi_name)[0]
    print(f"Scoring {wsi_without_ext}")

    cell_points_path = os.path.join(det_out_dir, f"{wsi_without_ext}_points.json")
    wsi_path = os.path.join(wsi_dir, wsi_name)

    if not os.path.exists(cell_points_path):
        print(f"No detection map found for {wsi_without_ext}")
        return 0

    tumor_stroma_mask_path = os.path.join(
        seg_out_dir, f"{wsi_without_ext}_stroma_bulk.npy"
    )
    mask_path = os.path.join(temp_out_dir, f"{wsi_without_ext}.npy")

    tissue_mask = np.load(mask_path)[:, :, 0]

    if is_l1(tissue_mask):
        mask_reader = VirtualWSIReader.open(tissue_mask, mpp=8, power=1.25, mode="bool")
        mask = mask_reader.slide_thumbnail(resolution=32, units="mpp")[:, :, 0]
    else:
        mask = np.load(tumor_stroma_mask_path)

    score = create_til_score(wsi_path, cell_points_path, mask)

    output_tils_path = os.path.join(output_tils_dir, f"{wsi_without_ext}.txt")
    with open(output_tils_path, "w") as file:
        file.write(str(score))
    print(f"{wsi_without_ext} TIL score saved")
    return score


if __name__ == "__main__":
    wsi_name_list = os.listdir(wsi_dir)
    with Pool(10) as p:
        p.map(til_score_process, wsi_name_list)
