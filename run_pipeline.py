import os
from tqdm.auto import tqdm

from tissue_masker_lite import get_mask

from config import Challenge_Config, Default_Config, DGX_Config
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
    IOConfig = DGX_Config()
    IOConfig.create_output_dirs()

    files = []
    with open("/home/u1910100/cloud_workspace/GitHub/TIAger-Torch/tcga_brca_files.txt", "r") as file:
        for line in file:
            name = line.rstrip()
            files.append(name)
    
    batch_num = 4
    # 0-300, 300-600, 600-900, 900-:
    batch_files = files[900:]

    for wsi_name in tqdm(batch_files, position=0, leave=True):
        # wsi_name = "104S.tif"
        try:
            generate_til_score(wsi_name, IOConfig)
        except Exception as e:
            print(f"{wsi_name} failed")
            print(e)
