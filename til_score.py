import json
import os
from multiprocessing import Pool

import pandas as pd
import scipy.stats
from tqdm.auto import tqdm

from utils import create_til_score


def til_score_process(wsi_name):
    wsi_without_ext = os.path.splitext(wsi_name)[0]
    cell_points_dir = "/home/u1910100/GitHub/TIAger-Torch/output/det_out"
    cell_points_path = os.path.join(cell_points_dir, f"{wsi_without_ext}_points.json")
    wsi_dir = "/home/u1910100/Documents/Tiger_Data/wsitils/images/"
    wsi_path = os.path.join(wsi_dir, wsi_name)

    tumor_stroma_mask_path = f"/home/u1910100/GitHub/TIAger-Torch/output/seg_out/{wsi_without_ext}_tumor_stroma.npy"
    score = create_til_score(wsi_path, cell_points_path, tumor_stroma_mask_path)
    return score


if __name__ == "__main__":
    wsi_name = "104S.tif"
    score = til_score_process(wsi_name)
    print(score)
