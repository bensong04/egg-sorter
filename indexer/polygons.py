### DEMO ###

import cv2
import numpy as np
from math import sqrt
import glob
from copy import copy as copy

hep_scale = 0.595 # area of the heptagon in mm^2
# px * sqrt(mm^2 / px^2) <== CONVERSION BETWEEN PX TO MM
hep_mm_l = 1.0
tri_mm_l = 1.5

def get_contours(f: str):
    img = cv2.imread(f, cv2.IMREAD_COLOR)
    height = img.shape[0]

    alpha = 1
    beta = 0

    cropdist = 150
    tlcol = copy(img[0:cropdist, 0:cropdist])
    blcol = copy(img[height-cropdist:height, 0:cropdist])
    cut_image = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    topleft = cut_image[0:cropdist, 0:cropdist]
    bottomleft = cut_image[height-cropdist:height, 0:cropdist] # Y, X ORDERING: CHANGE CHANNELLESS CROPPING TO REFLECT THIS

    kernel = (5,)*2
    topleft = cv2.GaussianBlur(topleft, kernel, cv2.BORDER_DEFAULT)
    bottomleft = cv2.GaussianBlur(bottomleft, kernel, cv2.BORDER_DEFAULT)

    down = 30
    up = 70
    edgetop = cv2.Canny(topleft, down, up)
    edgebottom = cv2.Canny(bottomleft, down, up)

    mat = np.ones((4, 4), np.uint8)
    #edgetop = cv2.erode(edgetop, mat, iterations=1)
    edgetop = cv2.dilate(edgetop, mat, iterations=1)
    #edgebottom = cv2.erode(edgebottom, mat, iterations=1)
    edgebottom = cv2.dilate(edgebottom, mat, iterations=1)

    t_contours, _ = cv2.findContours(edgetop, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    b_contours, _ = cv2.findContours(edgebottom, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    mtc = max(t_contours, key = cv2.contourArea)
    mbc = max(b_contours, key = cv2.contourArea)
    cv2.drawContours(image=tlcol, contours=mtc, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.drawContours(image=blcol, contours=mbc, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

    sl_top_hep = sqrt(cv2.contourArea(mtc) / 0.595)  # in px, TOP SIDE ONLY
    sl_btm_tri = 2*sqrt(cv2.contourArea(mbc)/sqrt(3)) # in px

    top_r = round(hep_mm_l/sl_top_hep, 5)
    btm_r = round(tri_mm_l/sl_btm_tri, 5)

    tlcol = cv2.resize(tlcol, (750, 750), interpolation= cv2.INTER_LINEAR)
    blcol = cv2.resize(blcol, (750, 750), interpolation= cv2.INTER_LINEAR)

    cv2.putText(tlcol,'side length: %f'%round(sl_top_hep, 4), (10,100), 1, 4, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(tlcol,'mm/px: %f'%top_r, (10,600), 1, 4, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(blcol,'side length: %f'%round(sl_btm_tri, 4),(10,100), 1, 4, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(blcol,'mm/px: %f'%btm_r,(10,600), 1, 4, (255, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Result Image", tlcol)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow("Result Image", blcol)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

for f in glob.glob("tests/eggs_batch4/*.jpg"):
    get_contours(f)

# IDEA: get the area and convert that to side length