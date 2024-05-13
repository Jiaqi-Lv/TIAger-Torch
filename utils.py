import json
import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import skimage
import torch
from shapely import Point, Polygon
from tiatoolbox.annotation.storage import Annotation, SQLiteStore
from tiatoolbox.wsicore.wsireader import VirtualWSIReader, WSIReader


def mm2_to_px(mm2, mpp):
    return (mm2 * 1e6) / mpp**2


def dist_to_px(dist, mpp):
    dist_px = int(round(dist / mpp))
    return dist_px


def px_to_mm(px, mpp):
    return px * mpp / 1000


def px_to_um2(px, mpp):
    area_um2 = px * (mpp**2)
    return area_um2


def point_to_box(x, y, size):
    """Convert centerpoint to bounding box of fixed size"""
    return np.array([x - size, y - size, x + size, y + size])


def get_centerpoints(box, dist):
    """Returns centerpoints of box"""
    return (box[0] + dist, box[1] + dist)


def non_max_suppression_fast(boxes, overlapThresh):
    """Very efficient NMS function taken from pyimagesearch"""

    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(
            idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0]))
        )
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


def slide_nms(slide_path, cell_points, tile_size):
    """Iterate over WholeSlideAnnotation and perform NMS. For this to properly work, tiles need to be larger than model inference patches."""
    # Open WSI and detection points file
    wsi = WSIReader.open(slide_path)
    store = points_to_annotation_store(cell_points)
    # Get base line WSI size
    shape = wsi.slide_dimensions(resolution=0, units="level")

    center_nms_points = []

    box_size = 8
    # get 2048x2048 patch coordinates without overlap
    for y_pos in range(0, shape[1], tile_size):
        for x_pos in range(0, shape[0], tile_size):
            # Select annotations within 2048x2048 box
            box = [x_pos, y_pos, x_pos + tile_size, y_pos + tile_size]
            patch_points = get_points_within_box(store, box)

            if len(patch_points) < 2:
                continue

            # Convert each point to a 8x8 box
            boxes = np.array([point_to_box(x[0], x[1], box_size) for x in patch_points])
            nms_boxes = non_max_suppression_fast(boxes, 0.7)
            for box in nms_boxes:
                center_nms_points.append(get_centerpoints(box, box_size))
    return center_nms_points


def points_to_annotation_store(points: list):
    """
    Args: points(list): list of (x,y) coordinates
    """
    annotation_store = SQLiteStore()

    for coord in points:
        annotation_store.append(
            Annotation(geometry=Point(coord[0], coord[1]), properties={"class": 1})
        )

    return annotation_store


def points_from_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    coordinates = root[0][0][0]
    points = []
    for child in coordinates:
        coord = [float(child.attrib["X"]), float(child.attrib["Y"])]
        points.append(coord)
    return points


def get_points_within_box(annotation_store, box):
    query_poly = Polygon.from_bounds(box[0], box[1], box[2], box[3])
    anns = annotation_store.query(geometry=query_poly)
    results = []
    for point in anns.items():
        results.append(point[1].coords[0])
    return results


def get_mask_area(mask):
    """
    Get the size of a mask in pixels where the mask is 1.
    Mask resolution = 32 mpp
    """

    counts = np.count_nonzero(mask)
    down = 6
    area = counts * down**2
    return area


def create_til_score(wsi_path, cell_points_path, mask):
    cell_points_format = os.path.splitext(cell_points_path)[1]
    if cell_points_format == ".json":
        with open(cell_points_path, "r") as fp:
            cell_points = json.load(fp)
    else:
        cell_points = points_from_xml(cell_points_path)

    # print(f"{len(cell_points)} before nms")
    nms_points = slide_nms(slide_path=wsi_path, cell_points=cell_points, tile_size=2048)

    cell_counts = len(nms_points)
    # print(f"TIL counts = {cell_counts}")

    til_area = dist_to_px(4, 0.5) ** 2
    tils_area = cell_counts * til_area

    stroma_area = get_mask_area(mask)
    # print(f"stroma area = {stroma_area}")
    tilscore = int((100 / int(stroma_area)) * int(tils_area))
    tilscore = min(100, tilscore)
    tilscore = max(0, tilscore)
    return tilscore
    # print(f"tilscore = {tilscore}")


def imagenet_normalise(img: torch.tensor) -> torch.tensor:
    """Normalises input image to ImageNet mean and std"""

    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    for i in range(3):
        img[:, i, :, :] = (img[:, i, :, :] - mean[i]) / std[i]
    return img


def get_seg_models():
    segModel1 = "/home/u1910100/cloud_workspace/GitHub/TIAger-Torch/runs/tissue/fold_1/model_59.pth"
    segModel2 = "/home/u1910100/cloud_workspace/GitHub/TIAger-Torch/runs/tissue/fold_3/model_41.pth"
    segModel3 = "/home/u1910100/cloud_workspace/GitHub/TIAger-Torch/runs/tissue/fold_4/model_35.pth"
    segModel = [segModel1, segModel2, segModel3]

    models: list[torch.nn.Module] = []
    for model_path in segModel:
        model = smp.Unet(
            encoder_name="efficientnet-b0",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=None,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=3,  # model output channels (number of classes in your dataset)
        )

        model.load_state_dict(torch.load(model_path))

        model.to("cuda")
        model.eval()
        models.append(model)
    return models


def get_det_models():
    detModel1 = "/home/u1910100/cloud_workspace/GitHub/TIAger-Torch/runs/cell/fold_1/model_55.pth"
    detModel2 = "/home/u1910100/cloud_workspace/GitHub/TIAger-Torch/runs/cell/fold_2/model_40.pth"
    detModel3 = "/home/u1910100/cloud_workspace/GitHub/TIAger-Torch/runs/cell/fold_3/model_30.pth"
    detModel = [detModel1, detModel2, detModel3]

    models: list[torch.nn.Module] = []
    for model_path in detModel:
        model = smp.Unet(
            encoder_name="efficientnet-b0",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=None,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,  # model output channels (number of classes in your dataset)
        )

        model.load_state_dict(torch.load(model_path))

        model.to("cuda")
        model.eval()
        models.append(model)
    return models


def get_tumor_bulk(tumor_seg_mask):
    binary_tumor_mask = tumor_seg_mask.astype(bool)
    mpp = 32
    min_size_px = mm2_to_px(1.0, mpp)
    kernel_diameter = dist_to_px(1000, mpp)

    wsi_patch = skimage.morphology.remove_small_objects(
        binary_tumor_mask, min_size=mm2_to_px(0.005, mpp), connectivity=2
    )
    wsi_patch = wsi_patch.astype(np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_diameter, kernel_diameter)
    )
    closing = cv2.morphologyEx(wsi_patch, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    wsi_patch = opening
    wsi_patch = wsi_patch.astype(bool)
    wsi_patch = skimage.morphology.remove_small_objects(
        wsi_patch, min_size=min_size_px, connectivity=2
    )

    labels = skimage.measure.label(wsi_patch)
    try:
        tumor_bulk = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    except ValueError:
        tumor_bulk = closing

    return tumor_bulk


def get_tumor_stroma_mask(bulk_tumor_mask, stroma_mask):
    tumor_stroma_mask = bulk_tumor_mask * stroma_mask
    mpp = 32
    kernel_diameter = dist_to_px(1000, mpp)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_diameter, kernel_diameter)
    )
    closing = cv2.morphologyEx(tumor_stroma_mask, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

    if np.count_nonzero(opening) > 0:
        tumor_stroma_mask = opening
    else:
        tumor_stroma_mask = closing

    return tumor_stroma_mask


def is_l1(mask):
    count = np.count_nonzero(mask)
    return count < 50000
