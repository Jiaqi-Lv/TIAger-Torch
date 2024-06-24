import os
from tqdm.auto import tqdm

from tissue_masker_lite import get_mask

from config import Challenge_Config, Default_Config
from detection_inference_torch import detection_process
from segmentation_inference_torch import tumor_stroma_process
from til_score import til_score_process


def generate_til_score(wsi_name, IOConfig):
    wsi_dir = IOConfig.input_dir
    input_mask_dir = IOConfig.input_mask_dir

    wsi_without_ext = os.path.splitext(wsi_name)[0]
    wsi_path = os.path.join(wsi_dir, wsi_name)

    if not os.path.exists(wsi_path):
        print(f"{wsi_path} can not be found")
        return 0

    # check if tissue mask exists:
    mask_path = os.path.join(input_mask_dir, f"{wsi_without_ext}.npy")
    if not os.path.exists(mask_path):
        get_mask(wsi_path=wsi_path, save_dir=input_mask_dir, return_mask=False)

    tumor_stroma_process(wsi_name, IOConfig)
    detection_process(wsi_name, IOConfig)
    til_score_process(wsi_name, IOConfig)
    return 1


if __name__ == "__main__":
    IOConfig = Default_Config(
        input_dir="/home/u1910100/Documents/Tiger_Data/local_testing/images",
        output_dir="/home/u1910100/Documents/Tiger_Data/local_testing/prediction",
    )
    IOConfig.input_mask_dir = (
        "/home/u1910100/Documents/Tiger_Data/local_testing/masks"
    )
    IOConfig.create_output_dirs()

    files = os.listdir(IOConfig.input_dir)
    for wsi_name in tqdm(files, position=0, leave=True):
        # wsi_name = "104S.tif"
        try:
            generate_til_score(wsi_name, IOConfig)
        except Exception as e:
            print(e)
