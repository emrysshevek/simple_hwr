import os
import torch
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import random
from pathlib import Path
import os
import torch
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import random
from pathlib import Path
from scipy import interpolate
from sklearn.preprocessing import normalize


## Other loss functions and variations
# Distance from actual point
# Distance to nearest point on image
# Minimal penalty for first/last stroke points
# Make it probabilistic? I.e. there's some probability it starts at the top or bottom
# Compare rendered images! How to make it differentiable?
# Handle empty time differently - have computer guess "in between" stroke value? could work
# Stroke level penalties - perfect inverse/backward patterns have minimal penalty

def get_strokes(path):
    """

    Args:
        path: XML path to stroke file

    Returns:
        list of dicts: each dict contains a stroke, keys: x,y, time
    """
    root = ET.parse(path).getroot()
    strokes = root[1]

    # print("Strokes", len(strokes))

    stroke_list = []
    min_time = float(strokes[0][0].attrib["time"])
    last_time = 0
    stroke_delay = 0  # time between strokes
    start_end_strokes = []

    for stroke in strokes:
        x_coords = []
        y_coords = []
        time_list = []

        for i, point in enumerate(stroke):
            # print("Points", len(strokes))
            x, y, time = point.attrib["x"], point.attrib["y"], point.attrib["time"]
            x_coords.append(int(x))
            y_coords.append(-int(y))

            if i == 0:  # no time passes between strokes!
                min_time += float(time) - min_time - last_time - .001
                start_end_strokes.append((last_time, float(time) - min_time))

            next_time = float(time) - min_time

            if time_list and next_time == time_list[-1]:
                next_time += .001
                assert next_time > time_list[-1]

            # No repeated times
            if time_list and next_time <= time_list[-1]:
                next_time = time_list[-1] + .001

            time_list.append(next_time)
        last_time = time_list[-1]
        stroke_list.append({"x": x_coords, "y": y_coords, "time": time_list})
    return stroke_list, start_end_strokes


def convert_strokes(stroke_list):
    """ Convert the stroke dict to 3 lists

    Args:
        stroke_list (list): list of dicts, each dict contains a stroke, keys: x,y, time
    Returns:
        tuple of array-likes: x coordinates, y coordinates, times

    """
    x, y, time = [], [], []
    [x.extend(key["x"]) for key in stroke_list]
    [y.extend(key["y"]) for key in stroke_list]
    [time.extend(key["time"]) for key in stroke_list]
    return np.array(x), np.array(y), np.array(time)


def process(time):
    total_time = np.max(time) - np.min(time)

def normalize(my_array):
    """ Max/min rescale to -1,1 range

    Args:
        my_array:

    Returns:

    """
    return ((my_array-np.min(my_array))/(np.max(my_array)-np.min(my_array))-.5)*2

def get_gts(path, instances = 50):
    """ Take in xml with strokes, output ordered target coordinates
        Parameterizes x & y coordinates as functions of t
        Any t can be selected; strokes are collapsed so there's minimal time between strokes
        If


    Args:
        path (str): path to XML
        instances (int): number of desired coordinates

    Returns:
        x-array, y-array
    """
    stroke_list, start_end_strokes = get_strokes(path)
    x,y,time = convert_strokes(stroke_list)

    # find dead timezones
    # make x and y independently a function of t
    time_continuum = np.linspace(np.min(time), np.max(time), instances)
    x_func = interpolate.interp1d(time, x)
    y_func = interpolate.interp1d(time, y)

    for i,t in enumerate(time_continuum):
        for lower, upper in start_end_strokes:
            if t < lower:
                break
            if t > lower and t < upper:
                t = lower if abs(t-lower)<abs(t-upper) else upper
                time_continuum[i] = t
                break
    return normalize(x_func(time_continuum)), normalize(y_func(time_continuum))

def l1_loss(preds, targs):
    loss = torch.sum(torch.abs(preds-targs))
    return loss

def test_cnn():
    import torch
    from models.basic import BidirectionalRNN, CNN
    import torch.nn as nn

    cnn = CNN(nc=1)
    pool = nn.MaxPool2d(3, (4, 1), padding=1)
    batch = 7
    y = torch.rand(batch, 1, 60, 1024)
    a, b = cnn(y, intermediate_level=13)
    new = cnn.post_process(pool(b))

    final = torch.cat([a, new], dim=2)
    print(a.size())
    print(final.size())

    for x in range(1000,1100):
        y = torch.rand(2, 1, 60, x)
        a,b = cnn(y, intermediate_level=13)

        print(a.size(), b.size())
        new = cnn.post_process(pool(b)).size()
        print(new)
        assert new == a.size()

def test_loss():
    import torch
    batch = 3
    y = torch.rand(batch, 1, 60, 1024)
    x = y
    z = l1_loss(y,x)
    print(z)

def test_stroke_parse():
    ## Add terminating stroke!!

    path = Path("/media/taylor/Data/Linux/Github/simple_hwr/data/prepare_online_data/lines-xml/a01-000u-06.xml")
    r = convert_strokes(get_strokes(path))
    datapoints = len(r)
    return r

if __name__=="__main__":
    x = test_stroke_parse()
    print(x)