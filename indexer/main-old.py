import cv2
import numpy as np
import scipy.optimize as spo
from matplotlib import pyplot as plt
from math import sin, tanh, sqrt
import glob
from os.path import dirname, join as joinpath
from copy import copy as copy
from itertools import count

CONTOURS_ON = False

# The following are measures of the smaller indicators. There are two sets of indicators to guarantee precision.
hep_scale = 0.595 # area of the heptagon in mm^2
# px * sqrt(mm^2 / px^2) <== CONVERSION BETWEEN PX TO MM
hep_mm_l = 1.0 # side length of the top, flat, long edge of the heptagon marker, in mm
tri_mm_l = 1.5 # side length of the equilateral triangle marker, in mm

# The larger indicators are scaled versions of the smaller indicators.
scale_factor = 2 # Larger indicator / smaller indicator, in mm/mm

# Adding contrast and brightness by applying an affine transformation to the original image.
alpha = 0.3
beta = 0

# To locate the indicators, we focus on the regions where they are, cropping out extraneous regions of the image.
cropdist = 150

# Top and bottom of the chip have crushed, stacked eggs, so these regions must be cut out before channelless fitting.
ycut = 200

def up(p, n):
    for i in range(n):
        p = dirname(p)
    return p

fls = glob.glob(joinpath(up(__file__, 2), "notebooks/20231019/*/*/*/*.tif"), recursive=True)

# Channelless fitting: near the middle region, egg centers naturally trace out the path of the egg channel, which itself can
# be modelled as a sinusoidal function.
sin_vec = np.vectorize(sin)
def vert_sine(y, freq, ampl, phase, off):
    return ampl*sin_vec(2*np.pi*freq*y + phase) + off
def vert_sine_fixed_freq(y, ampl, phase, off):
    return ampl*sin_vec(2*np.pi*1/309*y + phase) + off

# The following tanh function is a more accurate model of the egg channel, but traditional methods of optimization have trouble
# properly fitting this function to the data. Therefore, it remains unused.
boxy = lambda x: tanh(5*sin(x))

for i in range(30):
    f = fls[i] # joinpath(up(__file__, 1), "0072.tif") # fls[2] # TO BE PROPERLY IMPLEMENTED LATER
    print(f)
    if f == "/Users/ben/Documents/stanford/egg-sorter/notebooks/20231019/pm/thaw/bad/0072.tif":
        continue
    img = cv2.imread(f, cv2.IMREAD_COLOR)
    height = img.shape[0]

    ##############################################################################################################
    # We first compare px measurements of indicators against their known mm dimensions to determine px/mm ratio. #
    ##############################################################################################################
    if CONTOURS_ON:
        cut_image = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        topleft = cut_image[0:cropdist, 0:cropdist]
        bottomleft = cut_image[height-cropdist:height, 0:cropdist]

        # Apply Gaussian blur to filter out noise
        plg_kernel = (5,)*2 # Gaussian blur kernel specifically for finding contours within the image.
        topleft = cv2.GaussianBlur(topleft, plg_kernel, cv2.BORDER_DEFAULT)
        bottomleft = cv2.GaussianBlur(bottomleft, plg_kernel, cv2.BORDER_DEFAULT)

        # Apply Canny edge detection
        down = 30
        up = 70
        edgetop = cv2.Canny(topleft, down, up)
        edgebottom = cv2.Canny(bottomleft, down, up)

        # Dilation with 4x4 kernel to join contours
        mat = np.ones((4, 4), np.uint8)
        edgetop = cv2.dilate(edgetop, mat, iterations=1)
        edgebottom = cv2.dilate(edgebottom, mat, iterations=1)

        # Find contours...
        t_contours, _ = cv2.findContours(edgetop, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        b_contours, _ = cv2.findContours(edgebottom, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # and discard smaller contours, as they are unlikely to trace the indicator.
        mtc = max(t_contours, key = cv2.contourArea)
        mbc = max(b_contours, key = cv2.contourArea)

        # Find the indicator side lengths...
        sl_top_hep = sqrt(cv2.contourArea(mtc) / 0.595)  # in px, TOP SIDE ONLY
        sl_btm_tri = 2*sqrt(cv2.contourArea(mbc)/sqrt(3)) # in px
        # and compare against known mm dimensions, to determine mm/px ratios.
        top_r = round(hep_mm_l/sl_top_hep, 5)
        btm_r = round(tri_mm_l/sl_btm_tri, 5)

        # In the future, maybe use some sort of linear interpolation depending on egg center location in image.
        # This would be robust against certain angling changes in the chip, in particular the case when bottom of chip
        # is closer to camera plane than top of chip.
        canonical = (top_r + btm_r)/2 

    #############################################################################
    # Now that we have the mm/px ratio, focus on detecting eggs in the channel. #
    #############################################################################

    transformed = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    transformed = transformed[ycut:img.shape[0], 0:img.shape[1]//3]
    gray = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to filter out noise
    plg_kernel = (3,)*2 # Gaussian blur kernel specifically for finding contours within the image.
    blurgray = cv2.GaussianBlur(gray, plg_kernel, cv2.BORDER_DEFAULT)

    detected_circles = cv2.HoughCircles(blurgray, 
                    cv2.HOUGH_GRADIENT, 1.0, 35, param1 = 10,
                param2 = 25, minRadius = 25, maxRadius = 38)

    y_vals = []
    x_dep_vals = []

    for pt in detected_circles[0, :]:
        a, b, r = int(pt[0]), int(pt[1]), int(pt[2])
        if b in range(200, 800):
            y_vals.append(b)
            x_dep_vals.append(a)

        cv2.circle(img, (a, b+ycut), r, (0, 255, 0), 1)

        cv2.circle(img, (a, b+ycut), 1, (0, 0, 255), 3)

    guess = [1/220, 200, 0.2, 350]
    sin_fit_params = spo.curve_fit(vert_sine, y_vals, x_dep_vals, guess)[0]
    sin_fit_f = lambda y: vert_sine(y, sin_fit_params[0], sin_fit_params[1], sin_fit_params[2], sin_fit_params[3])
    sin_fit_guess = lambda y: vert_sine(y, guess[0], guess[1], guess[2], guess[3])

    def sq_dist_from_point_on_sine(y, a, b):
        return (sin_fit_f(int(y)) - a)**2 + (int(y) - b)**2

    egg_centers = []
    for center_x, center_y, _ in detected_circles[0, :]:
        # Closest parameter of the egg center, used for ordering eggs.
        min_pt = int(spo.minimize(sq_dist_from_point_on_sine, center_y, (center_x, center_y), method='Powell').x[0])
        egg_centers.append((min_pt+ycut, (center_x, center_y+ycut)))

    sorted_centers = [tup[1] for tup in sorted(egg_centers, key=lambda c: c[0])]

    for first_center, second_center, i in zip(sorted_centers, sorted_centers[1:], count()):
        center_x1 = first_center[0]
        center_y1 = first_center[1]
        center_x2 = second_center[0]
        center_y2 = second_center[1]

        dist = sqrt((center_y2-center_y1)**2 + (center_x2-center_x1)**2) * (canonical if CONTOURS_ON else 1) # px to mm

        mp_x = int((center_x1 + center_x2)/2)
        mp_y = int((center_y1 + center_y2)/2)

        cv2.line(img, (int(center_x1), int(center_y1)), (int(center_x2), int(center_y2)), (0, 255, 0), 1)
        if (i % 2 == 0):
            cv2.line(img, (mp_x, mp_y), (mp_x + 8, mp_y + 8), (255, 255, 255), 1)
            cv2.putText(img,('%g mm' if CONTOURS_ON else '%g px') %round(dist, 2), (mp_x+10, mp_y+10), 1, 0.9, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow("Result Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()