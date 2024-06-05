import os
import shutil

from config import Challenge_Config
from detection_inference_torch import detection_process_l1
from segmentation_inference_torch import tumor_stroma_process_l1

# from til_score import til_score_process


def segmentation_detection(wsi_name, mask_name, IOConfig):
    tumor_stroma_process_l1(wsi_name, mask_name, IOConfig)
    detection_process_l1(wsi_name, mask_name, IOConfig)

    # dummy til score for L1
    score = 50
    output_tils_path = os.path.join(IOConfig.temp_out_dir, "til-score.json")
    with open(output_tils_path, "w") as file:
        file.write(str(score))
    final_path = os.path.join(IOConfig.output_tils_dir, "til-score.json")
    shutil.copyfile(output_tils_path, final_path)
    print(f"TIL score saved at {final_path}")

    print("Process complete")
    return 1


if __name__ == "__main__":
    # wsi_name = "104S.tif"
    IOConfig = Challenge_Config(
        input_dir="/input", output_dir="/output", temp_out_dir="/tempoutput"
    )
    # IOConfig = Challenge_Config()
    IOConfig.create_output_dirs()
    wsi_name = [x for x in os.listdir(IOConfig.input_dir) if x.endswith(".tif")][0]
    mask_name = [x for x in os.listdir(IOConfig.input_mask_dir) if x.endswith(".tif")][
        0
    ]
    segmentation_detection(wsi_name, mask_name, IOConfig)
