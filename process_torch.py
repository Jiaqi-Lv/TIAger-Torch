import os

import torch
from tiatoolbox.wsicore.wsireader import VirtualWSIReader, WSIReader


def seg_inference(segModel, image, tissue_mask, dimensions, spacing, slide_file):
    pass


def detection_in_mask(
    detModel,
    image,
    tissue_mask,
    dimensions,
    spacing,
    slide_file,
    bulk_mask=None,
    nms=False,
):
    pass


class TIGERSegDet(object):
    def __init__(
        self,
        input_folder: str = "./input/images/",
        mask_folder: str = "./input/masks/",
        output_folder: str = "./output/",
        config_folder: str = "./configs/",
        model_folder: str = "./models/",
    ):
        self.input_folder: str = input_folder
        self.input_folder_masks: str = mask_folder
        self.output_folder: str = output_folder

    def process(self):
        level = 0
        tile_size = 1024

        """Segmentation inference"""
        slide_file = [x for x in os.listdir(self.input_folder) if x.endswith(".tif")][0]
        tissue_mask_slide_file = [
            x for x in os.listdir(self.input_folder_masks) if x.endswith(".tif")
        ][0]
        print(f"WSI: {slide_file}")
        print(f"Tissue mask: {tissue_mask_slide_file}")

        # open images
        image = WSIReader(os.path.join(self.input_folder, slide_file))
        tissue_mask = WSIReader(
            os.path.join(self.input_folder_masks, tissue_mask_slide_file)
        )

        # get image info
        dimensions = image.slide_dimensions(resolution=0, units="level")
        print(f"WSI dimension: {dimensions}")
        spacing = (
            0.5,
            0.5,
        )  # (10000/float(image.openslide_wsi.properties['tiff.XResolution']), 10000/float(image.openslide_wsi.properties['tiff.YResolution']))

        # segModel = get_model('segmentation')
        segModel1 = "/home/u1910100/GitHub/TIAger-Torch/runs/tissue/fold_1/model_59.pth"
        segModel2 = "/home/u1910100/GitHub/TIAger-Torch/runs/tissue/fold_3/model_41.pth"
        segModel3 = "/home/u1910100/GitHub/TIAger-Torch/runs/tissue/fold_4/model_35.pth"
        segModel = [segModel1, segModel2, segModel3]
        print("Segmenting tissue regions")
        seg_inference(segModel, image, tissue_mask, dimensions, spacing, slide_file)

        print("Collect TILs within stroma bulk region")
        """Detection inference"""
        # detModel = get_model('detection')
        detModel1 = "/home/u1910100/GitHub/TIAger-Torch/runs/cell/fold_1/model_55.pth"
        detModel2 = "/home/u1910100/GitHub/TIAger-Torch/runs/cell/fold_2/model_40.pth"
        detModel3 = "/home/u1910100/GitHub/TIAger-Torch/runs/cell/fold_3/model_30.pth"
        detModel = [detModel1, detModel2, detModel3]
        detection_in_mask(detModel, image, tissue_mask, dimensions, spacing, slide_file)


if __name__ == "__main__":
    if not os.path.exists("./output/segoutput"):
        os.makedirs("./output/segoutput")
    if not os.path.exists("./output/detoutput"):
        os.makedirs("./output/detoutput")
    if not os.path.exists("./output/bulkoutput"):
        os.makedirs("./output/bulkoutput")

    TIGERSegDet().process()
