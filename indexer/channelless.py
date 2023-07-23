import cv2
import numpy as np
import scipy.optimize as spo
from matplotlib import pyplot as plt
from math import sin
from jenkspy import JenksNaturalBreaks

no_lines = 4

jnk = JenksNaturalBreaks(no_lines)
sin_vec = np.vectorize(sin)
def vert_sine(y, freq, ampl, phase, off):
    return ampl*sin_vec(2*np.pi*freq*y + phase) + off
def vert_sine_fixed_freq(y, ampl, phase, off):
    return ampl*sin_vec(2*np.pi*1/309*y + phase) + off
img = cv2.imread('IMG_0014.jpeg', cv2.IMREAD_COLOR) # road.png is the filename
alpha = 0.3 # Contrast control (1.0-3.0)
beta = 0 # Brightness control (0-100)
gray = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

gray = gray[:, 0:img.shape[1]//2]
gray2 = img[:, 0:img.shape[1]//2]
gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(gray2, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray2, 50, 200)

gray = cv2.GaussianBlur(gray, (31,31), cv2.BORDER_DEFAULT)

# Apply Hough transform on the blurred image.
detected_circles = cv2.HoughCircles(gray, 
                   cv2.HOUGH_GRADIENT, 0.5, 20, param1 = 10,
               param2 = 20, minRadius = 25, maxRadius = 36)

lines = cv2.HoughLinesP(edges, 0.0001, np.pi/2, threshold=4, minLineLength=70)

y_vals = []
y_vals_lines = []
x_dep_vals = []
for line in lines:
    _, y1, _, y2 = line[0]
    if y1 != y2:
        continue
    y_vals_lines.append(y1)

jnk.fit(y_vals_lines)
y_vals_lines = [sum(l.tolist())//len(l.tolist()) if type(l.tolist()) is list else l.tolist() for l in jnk.groups_]
for y in y_vals_lines:
    cv2.line(img, (250, y), (550, y), (0, 0, 255), 2)
'''for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(img, (x1,y1), (x2,y2), (0, 0, 255), 2)'''

for pt in detected_circles[0, :]:
    a, b, r = int(pt[0]), int(pt[1]), int(pt[2])
    y_vals.append(b)
    x_dep_vals.append(a)

    # Draw the circumference of the circle.
    cv2.circle(img, (a, b), r, (0, 255, 0), 2)

    # Draw a small circle (of radius 1) to show the center. 

    cv2.circle(img, (a, b), 1, (0, 0, 255), 3)

sin_fit_params = spo.curve_fit(vert_sine, y_vals, x_dep_vals, [1/309, 200, 0.3, 50])[0]
#sin_fit_params = spo.curve_fit(vert_sine_fixed_freq, y_vals, x_dep_vals, [500, 0.3, 50])[0]
sin_fit_f = lambda y: vert_sine(y, sin_fit_params[0], sin_fit_params[1], sin_fit_params[2], sin_fit_params[3])
print(sin_fit_params)
#sin_fit_f = lambda y: vert_sine_fixed_freq(y, sin_fit_params[0], sin_fit_params[1], sin_fit_params[2])

last_pt = ()
for i in np.linspace(0, img.shape[0], 1000):
    if last_pt == ():
        last_pt = (round(sin_fit_f(i)), round(i))
    else:
        cur_pt = (round(sin_fit_f(i)), round(i))
        cv2.line(img, last_pt, cur_pt, (255, 0, 0), 2)
        last_pt = cur_pt

def sq_dist_from_point_on_sine(y, a, b):
    return (sin_fit_f(int(y)) - a)**2 + (int(y) - b)**2

for center_x, center_y, _ in detected_circles[0, :]:
    min_pt = int(spo.minimize(sq_dist_from_point_on_sine, center_y, (center_x, center_y), method='nelder-mead').x[0])
    print(min_pt)
    print(center_x, center_y)
    
    cv2.line(img, (int(center_x), int(center_y)), (int(sin_fit_f(min_pt)), min_pt), (0, 255, 255), 2)

# Show result
cv2.imshow("Result Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
