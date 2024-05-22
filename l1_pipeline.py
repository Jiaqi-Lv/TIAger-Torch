import os

from config import ChallengeConfig
from detection_inference_torch import detection_process_l1
from segmentation_inference_torch import tumor_stroma_process_l1
from til_score import til_score_process

output_dir = ChallengeConfig.output_dir
wsi_dir = ChallengeConfig.wsi_dir
temp_out_dir = ChallengeConfig.temp_out_dir
seg_out_dir = ChallengeConfig.seg_out_dir
det_out_dir = ChallengeConfig.det_out_dir


def segmentation_detection(wsi_name, mask_name):
    tumor_stroma_process_l1(wsi_name=wsi_name, mask_name=mask_name)
    detection_process_l1(wsi_name=wsi_name, mask_name=mask_name)
    return 1


if __name__ == "__main__":
    # wsi_name = "104S.tif"
    wsi_name = [x for x in os.listdir(wsi_dir) if x.endswith(".tif")][0]
    mask_name = [x for x in os.listdir(temp_out_dir) if x.endswith(".tif")][0]
    segmentation_detection(wsi_name=wsi_name, mask_name=mask_name)
