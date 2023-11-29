"""
Box and crop out eggs in channel images; stores the result
in a directory of choice.
Can be set to box only a certain amount of eggs. 
"""
DEBUG_MODE = False

import numpy as np
import cv2
import os
from typing import Tuple

def enclosing_rect(idim: Tuple[int, int], center: Tuple[int, int], radius: int, margins: int) -> Tuple[int, int, int, int]:
    """
    Returns a 4-element tuple.
    The first two elements locate the upper-left corner of the square.
    The last two elements locate the bottom-right corner of the square.
    Coordinates are rectified to the boundaries of the image so as to not reference
    out-of-bounds pixels.
    """
    return (
        max(0, center[0]-radius-margins), 
        max(0, center[1]-radius-margins),
        min(center[0]+radius+margins, idim[0]),
        min(center[1]+radius+margins, idim[1])
        )

def box_image(in_fp: str, out_dir: str, curr_count: int=-1, max_count: int=-1, debug: bool=False) -> int:
    """
    Returns the number of eggs found in the image.
    """
    assert os.path.isdir(out_dir)
    # hash the input filename to give each egg image a unique idenifier
    hash_fp = abs(hash(in_fp)) % (10 ** 8)
    # load the image at `fp` in grayscale
    original = cv2.cvtColor(cv2.imread(in_fp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
    # run canny edge detection
    edges = cv2.Canny(original, 20, 200)
    # apply gaussian blur
    blurred_edges = cv2.GaussianBlur(edges, (7,7), cv2.BORDER_DEFAULT)
    # run hough circle transform, with an emphasis on no false positives
    detected_circles = cv2.HoughCircles(blurred_edges, 
                        cv2.HOUGH_GRADIENT, 0.5, 20, param1 = 10,
                    param2 = 30, minRadius = 25, maxRadius = 36)
    if detected_circles is None:
        return 0
    
    # load the original image in color
    original_color = cv2.imread(in_fp, cv2.IMREAD_COLOR)  
    # draw and display detected circles if debug mode is on
    if debug:
        for pt in detected_circles[0, :]:
            a, b, r = int(pt[0]), int(pt[1]), int(pt[2])
            cv2.circle(original_color, (a, b), r, (0, 255, 0), 2)
            cv2.circle(original_color, (a, b), 1, (0, 0, 255), 3)
        cv2.imshow("Result Image", original_color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # quit before saving anything
        raise
    
    for egg_number, circ in enumerate(detected_circles[0, :]):
        if max_count != -1 and curr_count + egg_number*6 + 1 > max_count:
            break
        cx, cy, rad = int(circ[0]), int(circ[1]), int(circ[2]) 
        og_dimensions = (original_color.shape[1], original_color.shape[0])
        leftx, upy, rightx, downy = enclosing_rect(og_dimensions, (cx, cy), rad, 9)
        cropped_egg = original_color[upy:downy, leftx:rightx]
        # code to join mirror pairs together
        mirror = og_dimensions[0] - 20
        cropped_egg_mirror_img = original_color[upy:downy, (mirror-rightx):(mirror-leftx)]
        full_pair = cv2.hconcat([cropped_egg, cropped_egg_mirror_img])
        # we should also flip the order in which they appear to teach the model symmetry
        full_pair_flip = cv2.hconcat([cropped_egg_mirror_img, cropped_egg])
        egg_path = os.path.join(out_dir, "%d_egg_%d.png" % (hash_fp, egg_number))
        egg_path_flip = os.path.join(out_dir, "%d_egg_%d_flip.png" % (hash_fp, egg_number))
        # and do horizontal and vertical flips on both
        egg_path_H = os.path.join(out_dir, "%d_egg_%d_H.png" % (hash_fp, egg_number))
        full_pair_H = cv2.flip(full_pair, 0)
        egg_path_flip_H = os.path.join(out_dir, "%d_egg_%d_flip_H.png" % (hash_fp, egg_number)) 
        full_pair_flip_H = cv2.flip(full_pair_flip, 0)
        egg_path_V = os.path.join(out_dir, "%d_egg_%d_H.png" % (hash_fp, egg_number))
        full_pair_V = cv2.flip(full_pair, 1)
        egg_path_flip_V = os.path.join(out_dir, "%d_egg_%d_flip_V.png" % (hash_fp, egg_number)) 
        full_pair_flip_V = cv2.flip(full_pair_flip, 1)
        # write everything
        cv2.imwrite(egg_path, full_pair)
        cv2.imwrite(egg_path_flip, full_pair_flip)
        cv2.imwrite(egg_path_H, full_pair_H)
        cv2.imwrite(egg_path_flip_H, full_pair_flip_H)
        cv2.imwrite(egg_path_V, full_pair_V)
        cv2.imwrite(egg_path_flip_V, full_pair_flip_V) 

    return egg_number*6

import argparse
import glob
import sys

parser = argparse.ArgumentParser(description="") # TODO: write more detailed description

parser.add_argument("directory")
parser.add_argument("-o", "--output")
parser.add_argument("-m", "--maximum")

args = parser.parse_args()
in_drc = args.directory
out_drc = args.output
max_boxes = int(args.maximum) if args.maximum is not None else -1

if not os.path.exists(in_drc):
    raise IOError("%s is not a valid directory or filepath." % args.directory)
if not os.path.exists(out_drc):
    raise IOError("%s is not a valid directory or filepath." % args.directory)

all_tiffs = glob.glob(os.path.join(in_drc, "*.tif"))
curr_count_eggs = 0
tiff_no = 0
for tiff in all_tiffs:
    tiff_no += 1
    curr_count_eggs += box_image(tiff, out_drc, curr_count_eggs, max_boxes)
    if max_boxes != -1 and curr_count_eggs >= max_boxes:
        break

sys.stdout.write("Successfully boxed %d channel images into %d egg images" % (tiff_no, curr_count_eggs))