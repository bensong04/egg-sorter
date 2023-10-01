### DEMO ###

import cv2
import numpy as np
import scipy.optimize as spo
from matplotlib import pyplot as plt
from math import sin, tanh
import glob
from os.path import dirname, join as joinpath

def up(p, n):
    for i in range(n):
        p = dirname(p)
    return p

fls = glob.glob(joinpath(up(__file__, 2), "tests/*/*.jpg"), recursive=True)

boxy = lambda x: tanh(5*sin(x))
sin_vec = np.vectorize(sin)
def vert_sine(y, freq, ampl, phase, off):
    return ampl*sin_vec(2*np.pi*freq*y + phase) + off
def vert_sine_fixed_freq(y, ampl, phase, off):
    return ampl*sin_vec(2*np.pi*1/309*y + phase) + off


alpha = 0.3
beta = 0 

def disp(f):
    ycut = 200
    img = cv2.imread(f, cv2.IMREAD_COLOR)
    gray = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    gray = gray[ycut:img.shape[0], 0:img.shape[1]//2]
    gray2 = img[ycut:img.shape[0], 0:img.shape[1]//2]
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(gray2, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray2, 20, 200)

    gray = cv2.GaussianBlur(edges, (7,7), cv2.BORDER_DEFAULT)

    detected_circles = cv2.HoughCircles(gray, 
                    cv2.HOUGH_GRADIENT, 0.5, 20, param1 = 3,
                param2 = 20, minRadius = 25, maxRadius = 36)

    lines = cv2.HoughLinesP(edges, 0.0001, np.pi/2, threshold=4, minLineLength=70)

    y_vals = []
    x_dep_vals = []

    for pt in detected_circles[0, :]:
        a, b, r = int(pt[0]), int(pt[1]), int(pt[2])
        if b in range(200, 800):
            y_vals.append(b)
            x_dep_vals.append(a)

        cv2.circle(img, (a, b+ycut), r, (0, 255, 0), 2)

        cv2.circle(img, (a, b+ycut), 1, (0, 0, 255), 3)

    guess = [1/220, 200, 0.2, 350]
    sin_fit_params = spo.curve_fit(vert_sine, y_vals, x_dep_vals, guess)[0]
    sin_fit_f = lambda y: vert_sine(y, sin_fit_params[0], sin_fit_params[1], sin_fit_params[2], sin_fit_params[3])
    sin_fit_guess = lambda y: vert_sine(y, guess[0], guess[1], guess[2], guess[3])
    last_pt = ()
    for i in np.linspace(0, img.shape[0], 1000):
        if last_pt == ():
            last_pt = (round(sin_fit_f(i)), round(i)+ycut)
        else:
            cur_pt = (round(sin_fit_f(i)), round(i)+ycut)
            cv2.line(img, last_pt, cur_pt, (255, 0, 0), 2)
            last_pt = cur_pt
    # GUESS #
    '''
    last_pt = ()
    for i in np.linspace(0, img.shape[0], 1000):
        if last_pt == ():
            last_pt = (round(sin_fit_guess(i)), round(i)+ycut)
        else:
            cur_pt = (round(sin_fit_guess(i)), round(i)+ycut)
            cv2.line(img, last_pt, cur_pt, (255, 255, 0), 2)
            last_pt = cur_pt
    '''
    # GUESS #
    def sq_dist_from_point_on_sine(y, a, b):
        return (sin_fit_f(int(y)) - a)**2 + (int(y) - b)**2
    for center_x, center_y, _ in detected_circles[0, :]:
        min_pt = int(spo.minimize(sq_dist_from_point_on_sine, center_y, (center_x, center_y), method='Powell').x[0])
        
        cv2.line(img, (int(center_x), int(center_y)+ycut), (int(sin_fit_f(min_pt)), min_pt+ycut), (0, 255, 255), 2)

    cv2.imshow("Result Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

for f in fls:
    disp(f)
disp("tests/quality.jpg")
