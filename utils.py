import json
import math
import os
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import shapely
import shapely.geometry as geometry
import skimage
import torch
from PIL import Image, ImageDraw
from scipy.spatial import Delaunay
from shapely import Point, Polygon
from shapely.ops import polygonize, unary_union
from tiatoolbox.annotation.storage import Annotation, SQLiteStore
from tiatoolbox.tools.patchextraction import SlidingWindowPatchExtractor
from tiatoolbox.wsicore.wsireader import WSIReader

sys.path.append("/opt/ASAP/bin")
from wholeslidedata.interoperability.asap.imagewriter import \
    WholeSlideMonochromeMaskWriter

from config import ChallengeConfig, DefaultConfig

cell_model_dir = ChallengeConfig.cell_model_dir
tissue_mdoel_dir = ChallengeConfig.tissue_model_dir


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

    box_size = 5
    # get 2048x2048 patch coordinates without overlap
    for y_pos in range(0, shape[1], tile_size):
        for x_pos in range(0, shape[0], tile_size):
            # Select annotations within 2048x2048 box
            box = [x_pos, y_pos, x_pos + tile_size, y_pos + tile_size]
            patch_points = get_points_within_box(store, box)

            if len(patch_points) < 2:
                continue

            # Convert each point to a 5x5 box
            boxes = np.array([point_to_box(x[0], x[1], box_size) for x in patch_points])
            nms_boxes = non_max_suppression_fast(boxes, 0.5)
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

    nms_points = slide_nms(slide_path=wsi_path, cell_points=cell_points, tile_size=2048)

    cell_counts = len(nms_points)
    # print(f"TIL counts = {cell_counts}")

    # til_area = dist_to_px(4, 0.5) ** 2
    til_area = dist_to_px(2.3, 0.5) ** 2
    tils_area = cell_counts * til_area

    stroma_area = get_mask_area(mask)
    tilscore = int((100 / int(stroma_area)) * int(tils_area))
    tilscore = min(99, tilscore)
    tilscore = max(1, tilscore)
    return tilscore


def imagenet_normalise(img: torch.tensor) -> torch.tensor:
    """Normalises input image to ImageNet mean and std"""

    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    for i in range(3):
        img[:, i, :, :] = (img[:, i, :, :] - mean[i]) / std[i]
    return img


def get_seg_models():
    segModel1 = os.path.join(tissue_mdoel_dir, "tissue_1.pth")
    segModel2 = os.path.join(tissue_mdoel_dir, "tissue_3.pth")
    segModel3 = os.path.join(tissue_mdoel_dir, "tissue_4.pth")
    segModel = [segModel1, segModel2, segModel3]

    models: list[torch.nn.Module] = []
    for model_path in segModel:
        model = smp.Unet(
            encoder_name="efficientnet-b0",
            encoder_weights=None,
            in_channels=3,
            classes=3,
        )

        model.load_state_dict(torch.load(model_path))

        model.to("cuda")
        model.eval()
        models.append(model)
    return models


def get_det_models():
    detModel1 = os.path.join(cell_model_dir, "cell_1.pth")
    detModel2 = os.path.join(cell_model_dir, "cell_2.pth")
    detModel3 = os.path.join(cell_model_dir, "cell_3.pth")
    detModel = [detModel1, detModel2, detModel3]

    models: list[torch.nn.Module] = []
    for model_path in detModel:
        model = smp.Unet(
            encoder_name="efficientnet-b0",
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )

        model.load_state_dict(torch.load(model_path))

        model.to("cuda")
        model.eval()
        models.append(model)
    return models


def alpha_shape(points, alpha):
    def add_edge(edges, edge_points, coords, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            return
        edges.add((i, j))
        edge_points.append([coords[i], coords[j]])

    coords = [(i[0], i[1]) if type(i) or tuple else i for i in points]
    tri = Delaunay(coords)
    edges = set()
    edge_points = []

    # loop over triangles:
    # ia, ib, ic = indices of corner points of the
    # triangle
    for ia, ib, ic in tri.simplices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]
        # Lengths of sides of triangle
        a = math.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = math.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = math.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        # Semiperimeter of triangle
        s = (a + b + c) / 2.0
        # Area of triangle by Heron's formula
        area = math.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        # Here's the radius filter.
        # print circum_r
        if circum_r < 1 / alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)
    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return unary_union(triangles), edge_points


def calc_ratio(patch):
    ratio_patch = patch.copy()
    ratio_patch[ratio_patch > 1] = 1
    counts = np.unique(ratio_patch, return_counts=True)
    try:
        return (100 / counts[1][0]) * counts[1][1]
    except IndexError as ie:
        print(ie)
        print("Could not calculate ratio, using 0")
        return 0


def get_bulk(tumor_seg_mask):
    ratio = calc_ratio(tumor_seg_mask)
    print(ratio)
    mpp = 32
    min_size = 1.5

    if ratio > 50.0:
        kernel_diameter = dist_to_px(1000, mpp)
        min_size_px = mm2_to_px(min_size, mpp)
    else:
        min_size_px = mm2_to_px(1.0, mpp)
        kernel_diameter = dist_to_px(500, mpp)

    binary_tumor_mask = tumor_seg_mask.astype(bool)

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
    wsi_patch_indexes = skimage.morphology.remove_small_objects(
        wsi_patch, min_size=min_size_px, connectivity=2
    )
    wsi_patch = wsi_patch.astype(np.uint8)
    wsi_patch[wsi_patch_indexes == False] = 0
    wsi_patch[wsi_patch_indexes == True] = 1

    points = np.argwhere(wsi_patch == 1)
    if len(points) == 0:
        print(f"no hull found")
        return wsi_patch

    alpha = 0.07
    concave_hull, _ = alpha_shape(points, alpha)
    if isinstance(concave_hull, shapely.geometry.polygon.Polygon) or isinstance(
        concave_hull, shapely.geometry.GeometryCollection
    ):
        polygons = [concave_hull]
    else:
        polygons = list(concave_hull.geoms)

    buffersize = dist_to_px(250, mpp)
    polygons = [
        geometry.Polygon(list(x.buffer(buffersize).exterior.coords)) for x in polygons
    ]

    coordinates = []
    for polygon in polygons:
        if polygon.area < min_size_px:
            continue

        coordinates.append(
            [(int(x[1]), int(x[0])) for x in polygon.boundary.coords[:-1]]
        )
    print(f"tumor bulk counts {len(coordinates)}")

    dimensions = tumor_seg_mask.shape
    img = Image.new("L", (dimensions[1], dimensions[0]), 0)
    for poly in coordinates:
        ImageDraw.Draw(img).polygon(poly, outline=1, fill=1)
    tumor_bulk = np.array(img, dtype=np.uint8)
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


def convert_tissue_masks_for_l1(mask_path, tumor_mask_path, stroma_mask_path):
    mask_reader = WSIReader.open(mask_path)
    mask = mask_reader.slide_thumbnail(resolution=5, units="power")[:, :, 0]

    # Masks are at 5x resolution
    tumor_mask = np.load(tumor_mask_path)
    stroma_mask = np.load(stroma_mask_path)

    combined_mask = np.zeros_like(tumor_mask)
    combined_mask[np.where(tumor_mask == 1)] = 1
    combined_mask[np.where(stroma_mask == 1)] = 2

    combined_mask = combined_mask * mask

    mask_shape = combined_mask.shape

    patch_size = 256
    patch_extractor = SlidingWindowPatchExtractor(
        input_img=combined_mask, patch_size=(patch_size, patch_size)
    )

    tif_save_path = os.path.join(ChallengeConfig.seg_out_dir, f"segmentation.tif")
    writer = WholeSlideMonochromeMaskWriter()
    writer.write(
        path=tif_save_path,
        spacing=0.5,
        dimensions=(mask_shape[1] * 4, mask_shape[0] * 4),
        tile_shape=(patch_size * 4, patch_size * 4),
    )
    for i, patch in enumerate(patch_extractor):
        # mask =
        mask = cv2.resize(
            patch[:, :, 0],
            (patch_size * 4, patch_size * 4),
            interpolation=cv2.INTER_NEAREST,
        ).astype("uint8")
        x_start, y_start = (
            patch_extractor.coordinate_list[i][0],
            patch_extractor.coordinate_list[i][1],
        )
        writer.write_tile(tile=mask, coordinates=(int(x_start) * 4, int(y_start) * 4))
    writer.save()


def check_coord_in_mask(x, y, mask):
    """Checks if a given coordinate is inside the tissue mask
    Coordinate (x, y)
    Binary tissue mask at 5x
    """
    if mask is None:
        return True
    return mask[int(np.round(y)), int(np.round(x))] == 1
