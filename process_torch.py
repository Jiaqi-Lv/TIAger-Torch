import os
import pickle
from pprint import pprint

import cv2
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import skimage
import skimage.measure
import torch
from tiatoolbox.models.engine.semantic_segmentor import SemanticSegmentor
from tiatoolbox.tools.patchextraction import get_patch_extractor
from tiatoolbox.wsicore.wsireader import VirtualWSIReader, WSIReader
from tissue_masker_lite import get_mask
from tqdm.auto import tqdm


def process():
    """Segmentation inference"""

    print("Collect TILs within stroma bulk region")
    """Detection inference"""


if __name__ == "__main__":
    if not os.path.exists("./output/seg_out"):
        os.makedirs("./output/seg_out")
    if not os.path.exists("./output/det_out"):
        os.makedirs("./output/det_out")

    process()
