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


def segmentation_detection(wsi_name):
    wsi_without_ext = os.path.splitext(wsi_name)[0]
    wsi_path = os.path.join(wsi_dir, wsi_name)

    if not os.path.exists(wsi_path):
        print(f"{wsi_path} can not be found")
        return 0

    tumor_stroma_process_l1(wsi_name=wsi_name)
    # detection_process_l1(wsi_name=wsi_name)
    return 1


if __name__ == "__main__":
    # wsi_name = "104S.tif"
    slide_file = [x for x in os.listdir("/input") if x.endswith(".tif")][0]
    segmentation_detection(wsi_name=slide_file)
