import glob
import json
import math
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from detection_inference import (non_max_suppression_by_distance,
                                 process_image_tile_to_detections)
from efficientnet import keras as efn
from matplotlib import pyplot as plt
from shapely.ops import unary_union
from tensorflow.compat.v1.keras.applications import imagenet_utils
from tqdm import tqdm
from write_annotations import write_point_set

from segmentation_inference import (prepare_patching,
                                    process_image_tile_to_segmentation)
# from nms import slide_nms, to_wsd, dist_to_px
from utils import (cropping_center, folders_to_yml, get_centerpoints,
                   patchBoundsByOverlap, px_to_mm, slide_to_yml)

sys.path.append("/home/adams/Projects/tiatoolbox-1.0.1")
from tiatoolbox.utils.misc import imread
from tiatoolbox.wsicore.wsireader import (OpenSlideWSIReader, VirtualWSIReader,
                                          WSIMeta, WSIReader)


def get_model(model_type, weights_path=None):
    baseModel = efn.EfficientNetB0(
        weights=None, include_top=False, input_shape=(None, None, 3)
    )

    if model_type == "segmentation":
        model = efn.EfficiUNetB0(
            baseModel, seg_maps=3, seg_act="softmax", mode="normal"
        )
        if weights_path is None:
            model.load_weights(f"./weights/Segmentation_4.h5")
        else:
            model.load_weights(weights_path)

    if model_type == "detection":
        model = efn.EfficiUNetB0(baseModel, seg_maps=1, seg_act="sigmoid")
        if weights_path is None:
            model.load_weights("./weights/Detection_5.h5")
        else:
            model.load_weights(weights_path)
    return model


def write_json(data, path):
    path = Path(path)
    with path.open("wt") as handle:
        json.dump(data, handle, indent=4, sort_keys=False)


def seg_inference(segModel, image, tissue_mask, dimensions, spacing, slide_file):
    """Loop trough the tiles in the file performing central cropping of tiles, predict them with the segModel and write them to a mask"""
    level = 1  # 0
    mask_size = (256, 256)  # (512, 512)
    tile_size = (512, 512)  # (1024, 1024)
    dimensions = (dimensions[0] // 2, dimensions[1] // 2)
    spacing = (spacing[0] * 2, spacing[1] * 2)
    # create writers
    segmentation_writer = np.memmap(
        Path(f'output/segoutput/{slide_file.split(".")[0]}.npy'),
        dtype="uint8",
        mode="w+",
        shape=tuple(np.flip(dimensions)),
    )
    patch_info = prepare_patching(image, tile_size, mask_size, dimensions)
    for info in tqdm(patch_info):
        x, y, w, h, x1, y1, pad_t, pad_b, pad_l, pad_r = info
        tissue_mask_tile = tissue_mask.read_rect(
            (x, y), (w, h), resolution=level, units="level"
        )[..., 0]
        tissue_mask_tile = np.where(tissue_mask_tile >= 1, 1, 0)
        if not np.any(tissue_mask_tile):
            continue
        image_tile = image.read_rect((x, y), (w, h), resolution=level, units="level")
        image_tile = np.lib.pad(
            image_tile, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), "reflect"
        ).astype("uint8")
        tissue_mask_tile = np.lib.pad(
            tissue_mask_tile, ((pad_t, pad_b), (pad_l, pad_r)), "reflect"
        ).astype("uint8")
        # segmentation
        segmentation_mask = process_image_tile_to_segmentation(
            image_tile, tissue_mask_tile, segModel
        )
        end_y = min(y1 + tile_size[1] // 2, dimensions[1])
        end_x = min(x1 + tile_size[0] // 2, dimensions[0])
        segmentation_writer[y1:end_y, x1:end_x] = segmentation_mask[
            : min(end_y - y1, tile_size[1] // 2), : min(end_x - x1, tile_size[0] // 2)
        ]
    plt.imsave(
        f'output/segoutput/{slide_file.split(".")[0]}_ensemble_15_35_10.png',
        segmentation_writer,
    )


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
    level = 0
    tile_size = 1024

    output_dict = {
        "type": "Multiple points",
        "version": {"major": 1, "minor": 0},
        "points": [],
    }

    # if lb1 == True:
    #     tissue_label = int(1) # e.g. all of mask
    # else:
    #     tissue_label = int(2) # e.g. stroma only
    detection_writer = np.memmap(
        Path(f'output/detoutput/{slide_file.split(".")[0]}.npy'),
        dtype="uint8",
        mode="w+",
        shape=tuple(np.flip(dimensions)),
    )

    annotations = []
    # loop over image and get tiles with no overlap. Also write segmentation output to tiff
    for y in tqdm(range(0, dimensions[1], tile_size)):
        for x in range(0, dimensions[0], tile_size):
            tissue_mask_tile = tissue_mask.read_rect(
                (x, y), (tile_size, tile_size), resolution=level, units="level"
            )[..., 0]
            tissue_mask_tile = np.where(tissue_mask_tile >= 1, 1, 0)
            if not np.any(tissue_mask_tile):
                continue
            if bulk_mask is not None:
                bulk_mask_tile = bulk_mask.getUCharPatch(
                    startX=x, startY=y, width=tile_size, height=tile_size, level=level
                )
            else:
                bulk_mask_tile = 1

            if not np.any(bulk_mask_tile):
                continue

            image_tile = image.read_rect(
                (x, y), (tile_size, tile_size), resolution=level, units="level"
            )

            # detection
            detections, detection_map = process_image_tile_to_detections(
                image_tile, detModel
            )
            y_end = min(dimensions[1], y + tile_size)
            x_end = min(dimensions[0], x + tile_size)
            detection_writer[y:y_end, x:x_end] = detection_map[
                : min(tile_size, y_end - y), : min(tile_size, x_end - x)
            ]
            image_overlay = image.img
            for detection in detections:
                c, r, confidence = detection
                if tissue_mask_tile[r][c] != 1:  # tissue_label:
                    continue
                c1 = c + x
                r1 = r + y
                image_overlay = cv2.circle(image_overlay, (c1, r1), 3, (0, 255, 0), -1)

            #     prediction_record = {'point': [px_to_mm(c1, spacing[0]), px_to_mm(r1, spacing[0]), 0.5009999871253967], 'probability': confidence}
            #     output_dict['points'].append(prediction_record)
            #     if bulk_mask is not None:
            #         if bulk_mask_tile[r][c] != 1:
            #             continue
            #         annotations.append((c1,r1))
            #     else:
            #         annotations.append((c1,r1))

    # annotations = to_wsd(annotations)
    # write_point_set(annotations,
    #         f'tempoutput/detoutput/{slide_file.split(".")[0]}'+'.xml',
    #         label_name='lymphocytes',
    #         label_color='blue')

    # if nms:
    #     print('Performing slide level NMS...')
    #     output_dict = non_max_suppression_by_distance(output_dict, radius=3)
    # output_path = f'output/detoutput/detected-lymphocytes.json'
    # with open(output_path, 'w') as outfile:
    #     json.dump(output_dict, outfile, indent=4)

    plt.imsave(
        f'output/detoutput/{slide_file.split(".")[0]}_ensemble4_25.png',
        detection_writer,
    )
    cv2.imwrite(
        f'output/detoutput/{slide_file.split(".")[0]}_overlay_ensemble4_25.png',
        cv2.cvtColor(image_overlay, cv2.COLOR_RGB2BGR),
    )
    return


def set_tf_gpu_config():
    """Hard-coded GPU limit to balance between tensorflow and Pytorch"""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.set_logical_device_configuration(
                gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=6024)]
            )
            logical_gpus = tf.config.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)


def tf_be_silent():
    """Surpress exessive TF warnings"""
    try:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        tf.get_logger().setLevel("ERROR")
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    except Exception as ex:
        print("failed to silence tf warnings:", str(ex))


class TIGERSegDet(object):
    def __init__(
        self,
        input_folder="./input/",
        mask_folder="./input/images/",
        output_folder="./output/",
        config_folder="./configs/",
        model_folder="./models/",
    ):
        self.input_folder = input_folder + "/"
        self.input_folder_masks = mask_folder + "/"
        self.output_folder = output_folder + "/"

    def process(self):
        """INIT"""
        # set_tf_gpu_config()
        tf_be_silent()
        level = 0
        tile_size = 1024

        """Segmentation inference"""
        slide_file = [x for x in os.listdir(self.input_folder) if x.endswith(".tif")][0]
        tissue_mask_slide_file = [
            x for x in os.listdir(self.input_folder_masks) if x.endswith(".tif")
        ][0]

        # open images
        # image = VirtualWSIReader(os.path.join(self.input_folder, slide_file))
        # tissue_mask = VirtualWSIReader(os.path.join(self.input_folder_masks, tissue_mask_slide_file))
        image = OpenSlideWSIReader(os.path.join(self.input_folder, slide_file))
        tissue_mask = OpenSlideWSIReader(
            os.path.join(self.input_folder_masks, tissue_mask_slide_file)
        )

        # get image info
        dimensions = image.slide_dimensions(
            resolution=0, units="level"
        )  # image.openslide_wsi.dimensions
        spacing = (
            0.5,
            0.5,
        )  # (10000/float(image.openslide_wsi.properties['tiff.XResolution']), 10000/float(image.openslide_wsi.properties['tiff.YResolution']))

        # segModel = get_model('segmentation')
        segModel1 = get_model("segmentation", f"./weights/1_seg1.h5")
        segModel2 = get_model("segmentation", f"./weights/2_seg4.h5")
        segModel3 = get_model("segmentation", f"./weights/3_seg3.h5")
        segModel = [segModel1, segModel2, segModel3]
        seg_inference(segModel, image, tissue_mask, dimensions, spacing, slide_file)

        print("Collect TILs within stroma bulk region")
        """Detection inference"""
        # detModel = get_model('detection')
        detModel1 = get_model("detection", f"./weights/1_det1.h5")
        detModel2 = get_model("detection", f"./weights/2_det5.h5")
        detModel3 = get_model("detection", f"./weights/3_det2.h5")
        detModel = [detModel1, detModel2, detModel3]
        detection_in_mask(detModel, image, tissue_mask, dimensions, spacing, slide_file)


if __name__ == "__main__":
    Path("output").mkdir(parents=True, exist_ok=True)
    Path("output/segoutput").mkdir(parents=True, exist_ok=True)
    Path("output/detoutput").mkdir(parents=True, exist_ok=True)
    Path("output/bulkoutput").mkdir(parents=True, exist_ok=True)
    TIGERSegDet().process()
