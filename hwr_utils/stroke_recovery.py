from online_coordinate_parser import *
from pathlib import Path
import json
import cv2
import matplotlib.pyplot as plt

pad_dpi = {"padding":.05, "dpi":71}

def prep_figure(dpi=71):
    plt.figure(figsize=(1, 1), dpi=dpi)
    plt.axis('off')
    plt.axis('square')

def draw_strokes(stroke_list, x_to_y=1, line_width=None, save_path=""):
    # plt.NullFormatter()
    if save_path:
        prep_figure(pad_dpi["dpi"])
    if line_width is None:
        linewidth = max(random.gauss(1, .5), .4)

    if x_to_y != 1:
        for stroke in stroke_list:
            stroke["x"] = [item * x_to_y for item in stroke["x"]]

    for stroke in stroke_list:
        plt.plot(stroke["x"], stroke["y"], linewidth=linewidth, color="black")

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
        plt.imshow(img, cmap="gray")

        # Rescale points
        factor = 2 * pad_dpi["padding"] + 1
        x = (x + 1) / 2 * 60 / factor + pad_dpi["padding"] * 60 / 2
        y = (y + 1) / 2 * 60 / factor + pad_dpi["padding"] * 60 / 2

    x_middle_strokes = x[np.where(start_points == 0)]
    y_middle_strokes = y[np.where(start_points == 0)]
    x_start_strokes = x[np.where(start_points == 1)]
    y_start_strokes = y[np.where(start_points == 1)]

    plt.scatter(x_middle_strokes, y_middle_strokes, s=4)
    plt.scatter(x_start_strokes, y_start_strokes, s=4)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

