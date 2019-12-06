import os
import torch
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import random
import math
from pathlib import Path
import os
import torch
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import random
from pathlib import Path
from scipy import interpolate
#from sklearn.preprocessing import normalize
from pathlib import Path
import json
from easydict import EasyDict as edict

from hwr_utils.stroke_plotting import *


## Other loss functions and variations
# Distance from actual point
# Distance to nearest point on image
# Minimal penalty for first/last stroke points
# Make it probabilistic? I.e. there's some probability it starts at the top or bottom
# Compare rendered images! How to make it differentiable?
# Handle empty time differently - have computer guess "in between" stroke value? could work
# Stroke level penalties - perfect inverse/backward patterns have minimal penalty
# Need an end of stroke and end of sequence token - small penalty if end of stroke is close

## Other ideas
# Feed in current "step" to RNN, scaled to same scale as width

# Add more instances -- otherwise make it so the first instance is at the start of the letter


def read_stroke_xml(path, start_stroke=None, end_stroke=None):
    """
    Args:
        path: XML path to stroke file OR XML parsing of it

    Returns:
        list of lists of dicts: each dict contains a stroke, keys: x,y, time
    """
    if isinstance(path, Path):
        root = ET.parse(path).getroot()
        all_strokes = root[1]
    else:
        all_strokes = path

    stroke_list = []
    strokes = all_strokes[start_stroke:end_stroke]
    start_times = []

    for stroke in strokes:
        x_coords = []
        y_coords = []
        time_list = []

        for i, point in enumerate(stroke):

            x, y, time = point.attrib["x"], point.attrib["y"], float(point.attrib["time"])
            if not stroke_list and i == 0:
                first_time = time

            time -= first_time
            x_coords.append(int(x))
            y_coords.append(-int(y))
            time_list.append(round(time, 3))
            if i == 0:
                start_times.append(time)

        stroke_list.append({"x": x_coords, "y": y_coords, "time": time_list})

    return stroke_list, start_times

def create_functions_from_strokes(stroke_dict):
    if not isinstance(stroke_dict, edict):
        stroke_dict = edict(stroke_dict)
    x_func = interpolate.interp1d(stroke_dict.t, stroke_dict.x)
    y_func = interpolate.interp1d(stroke_dict.t, stroke_dict.y)
    return x_func, y_func

def prep_stroke_dict(strokes, time_interval=None, scale_time_distance=True):
    """ Takes in a "raw" stroke dictionary for one image

        time_interval (float): duration of upstroke events; None=original duration
        Returns:
             x(t), y(t), stroke_up_down(t), start_times
                 Note that the last start time is the end of the last stroke
    """
    x_list = []
    y_list = []
    t_list = []
    start_strokes = []
    start_times = []
    epsilon = 1e-8

    # Epsilon is the amount of time before or after a stroke for the interpolation
    # Time between strokes must be greater than epsilon, or interpolated points between strokes will result
    if time_interval is None or time_interval < epsilon:
        time_interval = epsilon * 3

    distance = 0

    # euclidean distance metric
    distance_metric = lambda x, y: ((x[:-1] - x[1:]) ** 2 + (y[:-1] - y[1:]) ** 2) ** (1 / 2)

    # Loop through each stroke
    for i, stroke_dict in enumerate(strokes):
        xs = np.asarray(stroke_dict["x"])
        ys = np.asarray(stroke_dict["y"])
        distance += np.sum(distance_metric(xs, ys))

        x_list += stroke_dict["x"]
        y_list += stroke_dict["y"]
        start_strokes += [1] + [0] * (len(stroke_dict["x"])-1)

        # Set duration for "upstroke" events
        if not time_interval is None and i > 0:
            next_start_time = stroke_dict["time"][0]
            last_end_time = t_list[-1]
            t_offset = time_interval + last_end_time - next_start_time
            t_list_add = [t + t_offset for t in stroke_dict["time"]]
        else:
            t_list_add = stroke_dict["time"]

        t_list += t_list_add
        start_times += [t_list_add[0]]

    # Add the last time to the start times
    start_times += [t_list_add[-1]]
    start_strokes += [0]

    ## Normalize
    y_list = np.asarray(y_list)
    x_list = np.asarray(x_list)

    assert t_list[0] == start_times[0] # first time should be first start time
    t_list = np.asarray(t_list) - t_list[0]
    start_strokes = np.asarray(start_strokes)
    start_times = np.asarray(start_times) - start_times[0] # zero start

    y_list, scale_param = normalize(y_list)
    x_list, scale_param = normalize(x_list, scale_param)

    distance = distance / scale_param

    if scale_time_distance:
        time_factor = distance / t_list[-1]
        t_list = t_list * time_factor
        start_times = start_times * time_factor

    # Have interpolation not move after last point
    x_list = np.append(x_list, x_list[-1])
    y_list = np.append(y_list, y_list[-1])
    t_list = np.append(t_list, t_list[-1] + 20)
    x_to_y = np.max(x_list) / np.max(y_list)

    # Start strokes (binary list) will now be 1 short!

    output = edict({"x":x_list, "y":y_list, "t":t_list, "start_times":start_times, "x_to_y":x_to_y, "start_strokes":start_strokes, "raw":strokes, "tmin":start_times[0], "tmax":start_times[-1], "trange":start_times[-1]-start_times[0]})
    return output

## DOES THIS WORK? SHOULD BE THE SAME AS BATCH_TORCH, NEED TO TEST
def relativefy_batch(batch, reverse=False):
    """ A tensor: Batch, Width, Vocab

    Args:
        batch:
        reverse:

    Returns:

    """
    for i,b in enumerate(batch):
        #print(batch.size(), batch)
        #print(batch[i,:,0])
        #print(i, b)
        relativefy(b[:, 0], reverse=reverse)
        batch[i] = relativefy(b[:,0], reverse=reverse)
    return batch

def relativefy_batch_torch(batch, reverse=False):
    """ A tensor: Batch, Width, Vocab
    """
    if reverse:
        return torch.cumsum(batch,dim=1)
    else:
        r = torch.zeros(batch.shape)
        r[:,1:] = batch[:,1:]-batch[:, :-1]
        return r


def relativefy(x, reverse=False):
    """
    Args:
        x:
        reverse:

    Returns:

    """
    if isinstance(x, np.ndarray):
        relativefy_numpy(x, reverse)
    elif isinstance(x, torch.Tensor):
        relativefy_torch(x, reverse)
    else:
        raise Exception(f"Unexpected type {type(x)}")

def relativefy_numpy(x, reverse=False):
    """ Make the x-coordinate relative to the previous one
        First coordinate is relative to 0
    Args:
        x (array-like): Just an array of x's coords!

    Returns:

    """
    if reverse:
        return np.cumsum(x,axis=0)
    else:
        return np.insert(x[1:]-x[:-1], 0, x[0])

def relativefy_torch(x, reverse=False):
    """ Make the x-coordinate relative to the previous one
        First coordinate is relative to 0
    Args:
        x:

    Returns:

    """
    if reverse:
        return torch.cumsum(x,dim=0)
    else:
        r = torch.zeros(x.shape)
        r[1:] = x[1:]-x[:-1]
        return r


def get_all_substrokes(stroke_dict, length=3):
    """

    Args:
        stroke_dict: ['x', 'y', 't', 'start_times', 'x_to_y', 'start_strokes', 'raw', 'tmin', 'tmax', 'trange']
        length:

    Returns:

    """
    if length is None:
        yield stroke_dict
        return

    start_args = np.where(stroke_dict.start_strokes==1)[0] # returns an "array" of the list, just take first index
    start_args = np.append(start_args, None) # last start arg should be the end of the sequence

    # If fewer strokes, just return the whole thing
    if start_args.shape[0] <= length:
        return stroke_dict

    for stroke_number in range(start_args.shape[0]-length): # remember, last start_stroke is really the end stroke
        start_idx = start_args[stroke_number]
        end_idx = start_args[stroke_number+length]

        t = stroke_dict.t[start_idx:end_idx].copy()
        x = stroke_dict.x[start_idx:end_idx].copy()
        y = stroke_dict.y[start_idx:end_idx].copy()
        raw = stroke_dict.raw[stroke_number:stroke_number+length]
        start_strokes = stroke_dict.start_strokes[start_idx:end_idx]
        start_times =   stroke_dict.start_times[stroke_number:stroke_number+length+1].copy()

        y, scale_param = normalize(y)
        x, scale_param = normalize(x, scale_param)
        x_to_y = np.max(x) / np.max(y)

        start_time = t[0]
        t -= start_time
        start_times -= start_time
        output = edict({"x": x,
                     "y": y,
                     "t": t,
                     "start_times": start_times,
                     "start_strokes": start_strokes,
                     "x_to_y":x_to_y,
                     "raw":raw})
        assert start_times[0]==t[0]
        yield output


def normalize(x_list, scale_param=None):
    x_list -= np.min(x_list)

    if scale_param is None:
        scale_param = np.max(x_list)

    x_list = x_list / scale_param
    return x_list, scale_param

def sample(function_x, function_y, starts, number_of_samples=64, noise=None, plot=False):
    last_time = starts[-1]
    interval = last_time / number_of_samples
    std_dev = interval / 4
    time = np.linspace(0, last_time, number_of_samples)

    if noise:
        momentum = .8
        if noise == "random":
            noises = np.random.normal(0, std_dev, time.shape)
        elif noise == "lagged":
            noises = []
            offset = 0
            noise = 0
            std_dev_decay = 1 - (.9) ** min(number_of_samples, 100)
            # Decay std_dev
            for i in range(0, number_of_samples):
                remaining = number_of_samples - i
                noise = np.random.normal(-offset / remaining + noise * momentum, std_dev)  # add momentum term
                offset += noise
                noises.append(noise + offset)
                if remaining < 100:
                    std_dev *= std_dev_decay
            noises = np.asarray(noises)

        if plot:
            plt.plot(time, noises)
            plt.show()
        time += noises
        time.sort(kind='mergesort')  # not actually a mergesort, but fast on nearly sorted data
        time = np.maximum(time, 0)
        time = np.minimum(time, last_time)

    ## Get start strokes
    start_stroke_idx = [0]  # first one is a start
    for start in starts[1:]:
        already_checked = start_stroke_idx[-1]
        start_stroke_idx.append(np.argmax(time[already_checked:] >= start) + already_checked)

    # print(function_x, function_y, time, start_stroke_idx)
    # time[start_stroke_idx] - start times
    is_start_stroke = np.zeros(time.shape)
    is_start_stroke[start_stroke_idx[:-1]] = 1 # the last "start stroke time" is not the last element, not a start stroke

    #print(time)
    return function_x(time), function_y(time), is_start_stroke

if __name__=="__main__":
    os.chdir("../data")
    with open("online_coordinate_data/3_stroke_16_v2/train_online_coords.json") as f:
        output_dict = json.load(f)

    instance = output_dict[11]
    render_points_on_image(instance['gt'], img_path=instance['image_path'], x_to_y=instance["x_to_y"])
