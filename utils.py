import numpy as np


# helper function for patch extracyion
def patchBoundsByOverlap(img, patch_size, overlap=0, out_bound="valid"):
    """This functions returns bounding boxes to extract patches with fized overlaps from image.

    Patch extraction is done uniformely on the image surface, by sampling
    patch_size x patch_size patches from the original image. Paches would have
    overlap wiith each other.
    inputs:
        img: input image to extract original image size
        patch_size: desired batch size
        overlap: amount of overlap between images
        out_bounds: the scnario of handling patches near the image boundaries
                    - if 'valid' (default), the last vertical/horizontal patch
                    start position would be changed to fit the image size. Therefore,
                    amount of overlap would be different for that specific patch.
                    - if None, no measure is taken in the bound calculation and probably
                    the image should be be padded to expand sizes before patch extraction.
                In either waym all patches should have size of patch_size.
    outputs:
        patch_boxes: a list of patch positions in the format of
                     [row_start, row_end, col_start, col_end]

    """
    out_bound = out_bound.lower()
    assert out_bound in {
        "valid",
        None,
    }, 'out_bound parameter must be either "padded" or None'
    if type(patch_size) is tuple:
        patch_size_c = patch_size[1]
        patch_size_r = patch_size[0]
    elif type(patch_size) is int:
        patch_size_c = patch_size
        patch_size_r = patch_size
    else:
        raise ("invalid type patch_size argumant")

    if type(overlap) is tuple:
        overlap_c = overlap[1]
        overlap_r = overlap[0]
    elif type(overlap) is int:
        overlap_c = overlap
        overlap_r = overlap
    else:
        raise ("invalid type for overlap argumant")

    img_rows, img_cols = img.shape[0:2]
    # calculating number of patches per image rows and cols
    num_patch_cols = (
        np.int(np.ceil((img_cols - patch_size_c) / (patch_size_c - overlap_c))) + 1
    )  # num patch columns
    num_patch_rows = (
        np.int(np.ceil((img_rows - patch_size_r) / (patch_size_r - overlap_r))) + 1
    )  # num patch rows

    patch_boxes = []
    for m in range(num_patch_cols):
        c_start = m * patch_size_c - m * overlap_c
        c_end = (m + 1) * patch_size_c - m * overlap_c
        if c_end > img_cols and out_bound == "valid":  # correct for the last patch
            c_diff = c_end - img_cols
            c_start -= c_diff
            c_end = img_cols
        for n in range(num_patch_rows):
            r_start = n * patch_size_r - n * overlap_r
            r_end = (n + 1) * patch_size_r - n * overlap_r
            if r_end > img_rows and out_bound == "valid":  # correct for the last patch
                r_diff = r_end - img_rows
                r_start -= r_diff
                r_end = img_rows
            patch_boxes.append([r_start, r_end, c_start, c_end])

    return patch_boxes
