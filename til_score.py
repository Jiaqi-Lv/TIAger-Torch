import json
import os
from multiprocessing import Pool

import numpy as np
from tiatoolbox.wsicore.wsireader import VirtualWSIReader
from tqdm.auto import tqdm

from utils import create_til_score, is_l1


output_dir = "/home/u1910100/cloud_workspace/GitHub/TIAger-Torch/output"
wsi_dir = "/home/u1910100/lab-private/it-services/TiGER/new_data/wsitils/images/"
temp_out_dir = os.path.join(output_dir, "temp_out/")
seg_out_dir = os.path.join(output_dir, "seg_out/")
det_out_dir = os.path.join(output_dir, "det_out/")
output_tils_dir = os.path.join(output_dir, f"tils/")

def til_score_process(wsi_name):
    wsi_without_ext = os.path.splitext(wsi_name)[0]
    print(f"Scoring {wsi_without_ext}")
    cell_points_path = os.path.join(det_out_dir, f"{wsi_without_ext}_points.json")
    wsi_path = os.path.join(wsi_dir, wsi_name)

    tumor_stroma_mask_path = os.path.join(seg_out_dir, f"{wsi_without_ext}_stroma_bulk.npy")
    mask_path = os.path.join(temp_out_dir, f"{wsi_without_ext}.npy")

    tissue_mask = np.load(mask_path)[:,:,0]
    if is_l1(tissue_mask):
        mask_reader = VirtualWSIReader.open(tissue_mask, mpp=8, power=1.25, mode="bool")
        mask = mask_reader.slide_thumbnail(resolution=32, units="mpp")[:,:,0]
    else:
        mask = np.load(tumor_stroma_mask_path)

    score = create_til_score(wsi_path, cell_points_path, mask)

    
    output_tils_path = os.path.join(output_tils_dir, f"{wsi_without_ext}.txt")
    with open(output_tils_path, "w") as file:
        file.write(str(score))
    return score


if __name__ == "__main__":

    wsi_name_list = os.listdir(wsi_dir)
    with Pool(2) as p:
        p.map(til_score_process, wsi_name_list)
