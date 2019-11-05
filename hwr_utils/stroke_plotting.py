from pathlib import Path
import matplotlib.pyplot as plt
from hwr_utils.stroke_plotting import *
import numpy as np
import os
import cv2
import random

def plot_stroke_points(x,y, start_points, square=False):
    x_middle_strokes = x[np.where(start_points == 0)]
    y_middle_strokes = y[np.where(start_points == 0)]
    x_start_strokes = x[np.where(start_points == 1)]
    y_start_strokes = y[np.where(start_points == 1)]

    plt.scatter(x_middle_strokes, y_middle_strokes, s=4)
    plt.scatter(x_start_strokes, y_start_strokes, s=4)

    max_y = np.max(y)
    head_length = .01*max_y
    head_width = .01*max_y
    for i, ((x1, y1), (x2, y2)) in enumerate(zip(zip(x, y),zip(x[1:], y[1:]))):
        if start_points[1:][i]:
            continue
        xdiff = (x2 - x1)
        ydiff = (y2 - y1)
        dx = xdiff / 2 #+ x1
        dy = ydiff / 2 #+ y1
        plt.arrow(x1, y1, dx, dy, color="blue", length_includes_head = True, head_length = head_length, head_width=head_width) # head_width = 1.4,

def render_points_on_image(gts, img_path, strokes=None, save_path=None, x_to_y=None):
    gts = np.array(gts)

    if not x_to_y is None:
        gts[0] *= x_to_y

    x = gts[0]
    y = gts[1]
    start_points = gts[2]

    # prep_figure()
    if strokes:
        draw_strokes(normalize_stroke_list(strokes), x_to_y=x_to_y)
    elif img_path:
        img_path = Path(img_path)
        img = cv2.imread(img_path.as_posix(), cv2.IMREAD_GRAYSCALE)
        img = img[::-1, :]
        img = cv2.resize(img, (60, 60))
        plt.imshow(img, cmap="gray", origin='lower')
        #plt.gca().invert_yaxis()
        # move all points positive, fit to square, apply padding, scale up
        x -= min(x)
        y -= min(y)
        x /= max(x)
        y /= max(y)
        x += pad_dpi["padding"]
        y += pad_dpi["padding"]
        x *= 54.7 # HACK fiddled with these constants
        y *= 55

        # (old) Rescale points
        #factor = 2 * pad_dpi["padding"] + 1
        #x = (x + 1) / 2 * 60 / factor + pad_dpi["padding"] * 60 / 2
        #y = (y + 1) / 2 * 60 / factor + pad_dpi["padding"] * 60 / 2

    plot_stroke_points(x,y,start_points)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


pad_dpi = {"padding":.05, "dpi":71}

def prep_figure(dpi=71, size=(5,1)):
    plt.figure(figsize=size, dpi=dpi)
    plt.axis('off')
    plt.axis('square')

def draw_strokes(stroke_list, x_to_y=1, line_width=None, save_path=""):
    # plt.NullFormatter()
    if line_width is None:
        line_width = max(random.gauss(1, .5), .4)
    if x_to_y != 1:
        for stroke in stroke_list:
            stroke["x"] = [item * x_to_y for item in stroke["x"]]

    if save_path:
        prep_figure(pad_dpi["dpi"], size=(int(x_to_y+.99),1))

    for stroke in stroke_list:
        plt.plot(stroke["x"], stroke["y"], linewidth=line_width, color="black")

    y_min = min([min(x["y"]) for x in stroke_list])
    y_max = max([max(x["y"]) for x in stroke_list])
    x_min = min([min(x["x"]) for x in stroke_list])
    x_max = max([max(x["x"]) for x in stroke_list])

    plt.ylim([y_min, y_max])
    plt.xlim([x_min, x_max])
    # print(y_min, y_max, x_min, x_max)
    if save_path:
        plt.savefig(save_path, pad_inches=pad_dpi["padding"], bbox_inches='tight')
        plt.close()

def normalize_stroke_list(stroke_list, maintain_ratio=False):
    """ Max/min rescale to -1,1 range

    Args:
        my_array:

    Returns:

    """
    normalize = lambda _array,_max,_min: (((np.array(_array)-_min)/(_max-_min)-.5)*2).tolist()
    x_max = np.max([max(x["x"]) for x in stroke_list])
    x_min = np.min([min(x["x"]) for x in stroke_list])
    y_min = np.min([min(x["y"]) for x in stroke_list])
    y_max = np.max([max(x["y"]) for x in stroke_list])

    ## THIS DOES NOT MAINTAIN CENTERING!
    if maintain_ratio:
         xrange = x_max-x_min
         yrange = y_max-y_min
         x_max = xrange * yrange/xrange + x_min


    new_stroke_list = []
    for item in stroke_list:
        #print(item["x"])
        new_stroke_list.append({"x":normalize(item["x"].copy(), x_max, x_min), "y":normalize(item["y"].copy(), y_max, y_min)})

    return new_stroke_list

