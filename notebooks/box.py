"""
Box and crop out eggs in channel images; stores the result
in a directory of choice.
Can be set to box only a certain amount of eggs. 
"""
DEBUG_MODE = False

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

def box_image(in_fp: str, out_dir: str, debug: bool=False) -> int:
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
        cx, cy, rad = int(circ[0]), int(circ[1]), int(circ[2]) 
        og_dimensions = (original_color.shape[1], original_color.shape[0])
        leftx, upy, rightx, downy = enclosing_rect(og_dimensions, (cx, cy), rad, 8)
        cropped_egg = original_color[upy:downy, leftx:rightx]
        egg_path = os.path.join(out_dir, "%d_egg_%d.png" % (hash_fp, egg_number))
        cv2.imwrite(egg_path, cropped_egg)

import argparse
import glob

parser = argparse.ArgumentParser(description="") # TODO: write more detailed description

parser.add_argument("directory")
parser.add_argument("-o", "--output")
parser.add_argument("-m", "--maximum")

args = parser.parse_args()
in_drc = args.directory
out_drc = args.output
max_boxes = args.maximum

if not os.path.exists(args.directory):
    raise IOError("%s is not a valid directory or filepath." % args.directory)


