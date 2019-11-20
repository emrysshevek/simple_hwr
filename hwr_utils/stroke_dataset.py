import re
import json
import multiprocessing
import torch
from torch.utils.data import Dataset
from hwr_utils.utils import print

import os
import cv2
import numpy as np
from tqdm import tqdm

from hwr_utils import stroke_recovery
from hwr_utils.utils import unpickle_it
from pathlib import Path
PADDING_CONSTANT = 0

script_path = Path(os.path.realpath(__file__))
project_root = script_path.parent.parent

class StrokeRecoveryDataset(Dataset):
    def __init__(self,
                 data_paths,
                 img_height=32,
                 num_of_channels=3,
                 root= project_root / "data",
                 max_images_to_load=None,
                 x_relative_positions=True,
                 cnn=None,
                 logger=None, **kwargs):

        # Make it an iterable
        if isinstance(data_paths, str) or isinstance(data_paths, Path):
            data_paths = [data_paths]

        self.collate = collate_stroke
        self.root = Path(root)
        self.num_of_channels = num_of_channels
        self.img_height = img_height
        self.interval = .05
        self.noise = None
        self.x_relative_positions = x_relative_positions
        self.cnn = cnn
        ### LOAD THE DATA LAST!!
        self.data = self.load_data(root, max_images_to_load, data_paths)

    def resample_one(self, item):
        """
        Args:
            item: Dictionary with a "raw" dictionary item
        Returns:
            Adds/modifies the "gt" key

        """
        output = stroke_recovery.prep_stroke_dict(item["raw"])  # returns dict with {x,y,t,start_times,x_to_y (aspect ratio), start_strokes (binary list), raw strokes}
        x_func, y_func = stroke_recovery.create_functions_from_strokes(output)
        number_of_samples = item["number_of_samples"] if "number_of_samples" in item.keys() else int(output.trange / self.interval)
        gt = create_gts(x_func, y_func, start_times=output.start_times, number_of_samples=number_of_samples, noise=self.noise, relative_x_positions=self.x_relative_positions)
        item["gt"] = gt  # {"x":x, "y":y, "is_start_time":is_start_stroke, "full": gt}
        item["x_func"] = x_func
        item["y_func"] = y_func
        item["start_times"] = output.start_times
        return item

    def resample_data(self, data_list, parallel=True):

        if parallel:
            poolcount = max(1, multiprocessing.cpu_count()-3)
            pool = multiprocessing.Pool(processes=poolcount)
            all_results = list(pool.imap_unordered(self.resample_one, tqdm(data_list)))  # iterates through everything all at once
            pool.close()
        else:
            all_results = []
            for item in data_list:
                all_results.append(self.resample_one(item))

        return all_results

    def load_data(self, root, images_to_load, data_paths):
        data = []
        for data_path in data_paths:
            data_path = str(data_path)
            with open(os.path.join(root, data_path)) as fp:
                new_data = json.load(fp)
                if isinstance(new_data, dict):
                    new_data = [item for key, item in new_data.items()]

                data.extend(new_data)
        # Calculate how many points are needed
        if self.cnn:
            add_output_size_to_data(data, self.cnn)
            self.cnn=True # remove CUDA-object from class for multiprocessing to work!!

        if images_to_load:
            print("Original dataloader size", len(data))
            data = data[:images_to_load]
        print("Dataloader size", len(data))

        if "gt" not in data[0].keys():
            data = self.resample_data(data, parallel=True)
        print("Done resampling", len(data))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = self.root / item['image_path']
        if self.num_of_channels == 3:
            img = cv2.imread(image_path.as_posix())
        elif self.num_of_channels == 1:  # read grayscale
            img = cv2.imread(image_path.as_posix(), 0)
        else:
            raise Exception("Unexpected number of channels")
        if img is None:
            print("Warning: image is None:", self.root / item['image_path'])
            return None

        percent = float(self.img_height) / img.shape[0]

        # Add channel dimension, since resize and warp only keep non-trivial channel axis
        if self.num_of_channels == 1:
            img = img[:, :, np.newaxis]

        img = img.astype(np.float32)
        img = img / 128.0 - 1.0

        ## DEFAULT GT ARRAY
        # X, Y, FLAG_BEGIN_STROKE, FLAG_END_STROKE, FLAG_EOS - VOCAB x length
        gt = np.asarray(item["gt"]) # LENGTH, VOCAB #.transpose([1,0])
        #assert gt[-1,2] != 1 # last stroke point shouldn't usually be a start stroke
        # print(gt.shape)
        # print(gt)
        # (32, 5)

        return {
            "line_img": img,
            "gt": gt,
            "path": image_path,
            "x_func": item["x_func"],
            "y_func": item["y_func"],
            "start_times": item["start_times"],
            "x_relative": self.x_relative_positions
        }

# def pad(list_of_numpy_arrays, lengths=None, variable_length_axis=1):
#     # Get original dimensions
#     batch_size = len(list_of_numpy_arrays)
#     dims = list(list_of_numpy_arrays[0].shape)
#
#     # Get the variable lengths
#     if lengths is None:
#         lengths = [l.shape[variable_length_axis] for l in list_of_numpy_arrays]  # only iteration
#     maxlen = max(lengths)
#     dims[variable_length_axis] = maxlen
#
#     # Reshape
#     list_of_numpy_arrays = [np.asarray(l).reshape(-1) for l in list_of_numpy_arrays]
#
#     # Create output array and mask
#     output_dims = batch_size, np.product(dims)
#     other_dims = int(np.product(dims)/maxlen)
#     arr = np.zeros(output_dims) # BATCH, VOCAB, LENGTH
#     mask = np.arange(maxlen) < np.array(lengths)[:, None]  # BATCH, MAX LENGTH
#     mask = np.tile(mask, other_dims).reshape(output_dims)
#     arr[mask] = np.concatenate(list_of_numpy_arrays)  # fast 1d assignment
#     return arr.reshape(batch_size, *dims)

def create_gts_from_raw_dict(item, interval, noise, relative_x_positions=False):
    """
    Args:
        item: Dictionary with a "raw" item
    Returns:

    """
    output = stroke_recovery.prep_stroke_dict(item) # returns dict with {x,y,t,start_times,x_to_y (aspect ratio), start_strokes (binary list), raw strokes}
    x_func, y_func = stroke_recovery.create_functions_from_strokes(output)
    number_of_samples = int(output.trange/interval)
    return create_gts(x_func, y_func, output.start_times, number_of_samples=number_of_samples, noise=noise, relative_x_positions=relative_x_positions)

def create_gts(x_func, y_func, start_times, number_of_samples, noise=None, relative_x_positions=False):
    """ Return LENGTH X VOCAB

    Args:
        x_func:
        y_func:
        start_times:
        number_of_samples:
        noise:

    Returns:

    """
    x, y, is_start_stroke = stroke_recovery.sample(x_func, y_func, start_times,
                                                   number_of_samples=number_of_samples, noise=noise)

    if relative_x_positions:
        x = stroke_recovery.relativefy(x)

    # Create GT matrix
    end_of_sequence_flag = np.zeros(x.shape[0])
    end_of_sequence_flag[-1] = 1
    gt = np.array([x, y, is_start_stroke, end_of_sequence_flag]).transpose([1,0])
    return gt


def put_at(start, stop, axis=1):
    if axis < 0:
        return (Ellipsis, ) + (slice(start, stop),) + (slice(None),) * abs(-1-axis)
    else:
        return (slice(None),) * (axis) + (slice(start,stop),)


def calculate_output_size(data, cnn):
    """ For each possible width, calculate the CNN output width
    Args:
        data:

    Returns:

    """
    all_possible_widths = set()
    for i in data:
        all_possible_widths.add(i)

    width_to_output_mapping={}
    for i in all_possible_widths:
        t = torch.zeros(1, 1, 32, i)
        shape = cnn(t).shape
        width_to_output_mapping[i] = shape[-1]
    return width_to_output_mapping

def add_output_size_to_data(data, cnn):
    """ Calculate how wide the GTs should be based on the output width of the CNN
    Args:
        data:

    Returns:

    """
    #cnn.to("cpu")
    width_to_output_mapping = {}
    for instance in data:
        width = instance["shape"][1] # H,W,Channels
        if width not in width_to_output_mapping:
            t = torch.zeros(1, 1, 32, width).to("cuda")
            shape = cnn(t).shape
            width_to_output_mapping[width] = shape[0]
        instance["number_of_samples"]=width_to_output_mapping[width]
    #cnn.to("cuda")

## Hard coded -- ~20% faster
# def pad3(batch, variable_width_dim=1):
#     dims = list(batch[0].shape)
#     max_length = max([b.shape[variable_width_dim] for b in batch])
#     dims[variable_width_dim] = max_length
#
#     input_batch = np.full((len(batch), *dims), PADDING_CONSTANT).astype(np.float32)
#
#     for i in range(len(batch)):
#         b_img = batch[i]
#         img_length = b_img.shape[variable_width_dim]
#         input_batch[i][ :, :img_length] = b_img
#     return input_batch

## Variable, foot loop based
def pad(batch, variable_width_dim=1):
    """ Outer dimension asumed to be batch, variable width dimension excluding batch

    Put at could kind of be moved outside of the loop
    Args:
        batch:
        variable_width_dim:

    Returns:

    """
    dims = list(batch[0].shape)
    max_length = max([b.shape[variable_width_dim] for b in batch])
    dims[variable_width_dim] = max_length
    input_batch = np.full((len(batch), *dims), PADDING_CONSTANT).astype(np.float32)

    for i in range(len(batch)):
        b_img = batch[i]
        img_length = b_img.shape[variable_width_dim]
        input_batch[i][put_at(0, img_length, axis=variable_width_dim)] = b_img
    return input_batch

def test_padding(pad_list, func):
    start = timer()
    for m in pad_list:
        x = func(m)
    # print(x.shape)
    # print(x[-1,-1])
    end = timer()
    print(end - start)  # Time in seconds, e.g. 5.38091952400282
    return x #[0,0]

def collate_stroke(batch, device="cpu"):
    """ Pad ground truths with 0's
        Report lengths to get accurate average loss

    Args:
        batch:
        device:

    Returns:

    """

    batch = [b for b in batch if b is not None]
    #These all should be the same size or error
    if len(set([b['line_img'].shape[0] for b in batch])) > 1:
        print("Problem with collating!!! See hw_dataset.py")
        print(batch)
    assert len(set([b['line_img'].shape[0] for b in batch])) == 1
    assert len(set([b['line_img'].shape[2] for b in batch])) == 1

    batch_size = len(batch)
    dim0 = batch[0]['line_img'].shape[0] # height
    dim1 = max([b['line_img'].shape[1] for b in batch]) # width
    dim2 = batch[0]['line_img'].shape[2] # channel

    all_labels = []
    label_lengths = []

    # Make input square (variable vidwth

    input_batch = np.full((batch_size, dim0, dim1, dim2), PADDING_CONSTANT).astype(np.float32)
    max_label = max([b['gt'].shape[0] for b in batch]) # width
    labels = np.full((batch_size, max_label, 4), PADDING_CONSTANT).astype(np.float32)

    for i in range(len(batch)):
        b_img = batch[i]['line_img']
        input_batch[i,:,: b_img.shape[1],:] = b_img

        l = batch[i]['gt']
        #all_labels.append(l)
        label_lengths.append(len(l))
        ## ALL LABELS - list of length batch size; arrays LENGTH, VOCAB SIZE
        labels[i,:len(l), :] = l
        all_labels.append(torch.from_numpy(l.astype(np.float32)).to(device))

    #print("ALL", all_labels.shape)
    label_lengths = np.asarray(label_lengths)

    line_imgs = input_batch.transpose([0,3,1,2]) # batch, channel, h, w
    line_imgs = torch.from_numpy(line_imgs).to(device)

    labels = torch.from_numpy(labels.astype(np.float32)).to(device)
    label_lengths = torch.from_numpy(label_lengths.astype(np.int32)).to(device)

    return {
        "line_imgs": line_imgs,
        "gt": labels, # Numpy Array, with padding
        "gt_list": all_labels, # List of numpy arrays
        "x_relative": [batch[0]["x_relative"]]*batch_size,
        "label_lengths": label_lengths,
        "paths": [b["path"] for b in batch],
        "x_func": [b["x_func"] for b in batch],
        "y_func": [b["y_func"] for b in batch],
        "start_times": [b["start_times"] for b in batch]
    }


if __name__=="__main__":
    # x = [np.array([[1,2,3,4],[4,5,3,5]]),np.array([[1,2,3],[4,5,3]]),np.array([[1,2],[4,5]])]
    # print(pad(x))
    from timeit import default_timer as timer

    vocab = 4
    iterations = 100
    batch = 32
    min_length = 32
    max_length = 64

    the_list = []

    for i in range(0,iterations): # iterations
        sub_list = []
        for m in range(0,batch): # batch size
            length = np.random.randint(min_length, max_length)
            sub_list.append(np.random.rand(vocab, length))
        the_list.append(sub_list)

    #test_padding(the_list, pad)
    x = test_padding(the_list, pad)
    # y = test_padding(the_list, pad2)
    # assert np.allclose(x,y)