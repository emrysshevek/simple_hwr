import re
import json
import multiprocessing
import torch
from torch.utils.data import Dataset

import os
import cv2
import numpy as np
from tqdm import tqdm

from hwr_utils import stroke_recovery
from hwr_utils.utils import unpickle_it
PADDING_CONSTANT = 0

class StrokeRecoveryDataset(Dataset):
    def __init__(self,
                 data_paths,
                 img_height=32,
                 num_of_channels=3,
                 root="./data",
                 max_images_to_load=None,
                 logger=None, **kwargs):

        self.collate = collate_stroke
        self.root = root
        self.num_of_channels = num_of_channels
        self.img_height = img_height
        self.interval = .05
        self.noise = None
        self.data = self.load_data(root, max_images_to_load, data_paths)

    def resample_one(self, item):
        output = stroke_recovery.prep_stroke_dict(item["raw"]) # returns dict with {x,y,t,start_times,x_to_y (aspect ratio), start_strokes (binary list), raw strokes}
        x_func, y_func = stroke_recovery.create_functions_from_strokes(output)
        #t = np.linspace(output.tmin, output.tmax, output.trange/output.interval)
        number_of_samples = int(output.trange/self.interval)
        x, y, is_start_stroke = stroke_recovery.sample(x_func, y_func, output.start_times, number_of_samples=number_of_samples, noise=self.noise)

        # Create GT matrix
        end_of_sequence_flag = np.zeros(x.shape[0])
        end_of_sequence_flag[-1] = 1
        gt = np.array([x, y, is_start_stroke, end_of_sequence_flag])
        item["gt"] = gt # {"x":x, "y":y, "is_start_time":is_start_stroke, "full": gt}
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
        image_path = os.path.join(self.root, item['image_path'])
        if self.num_of_channels == 3:
            img = cv2.imread(image_path)
        elif self.num_of_channels == 1:  # read grayscale
            img = cv2.imread(image_path, 0)
        else:
            raise Exception("Unexpected number of channels")
        if img is None:
            print("Warning: image is None:", os.path.join(self.root, item['image_path']))
            return None

        percent = float(self.img_height) / img.shape[0]

        # Add channel dimension, since resize and warp only keep non-trivial channel axis
        if self.num_of_channels == 1:
            img = img[:, :, np.newaxis]

        img = img.astype(np.float32)
        img = img / 128.0 - 1.0

        ## DEFAULT GT ARRAY
        # X, Y, FLAG_BEGIN_STROKE, FLAG_END_STROKE, FLAG_EOS - VOCAB x length
        gt = np.asarray(item["gt"]).transpose([1,0]) # LENGTH, VOCAB
        #assert gt[-1,2] != 1 # last stroke point shouldn't usually be a start stroke
        # print(gt.shape)
        # print(gt)
        # (32, 5)

        return {
            "line_img": img,
            "gt": gt,
            "path": image_path
        }

def collate_stroke_old(batch, device="cpu"):
    batch = [b for b in batch if b is not None]
    #These all should be the same size or error
    if len(set([b['line_img'].shape[0] for b in batch])) > 1:
        print("Problem with collating!!! See hw_dataset.py")
        print(batch)
    assert len(set([b['line_img'].shape[0] for b in batch])) == 1
    assert len(set([b['line_img'].shape[2] for b in batch])) == 1

    dim0 = batch[0]['line_img'].shape[0] # height
    dim1 = max([b['line_img'].shape[1] for b in batch]) # width
    dim2 = batch[0]['line_img'].shape[2] # channel

    all_labels = []
    label_lengths = []

    # Make input square (variable vidwth
    input_batch = np.full((len(batch), dim0, dim1, dim2), PADDING_CONSTANT).astype(np.float32)

    for i in range(len(batch)):
        b_img = batch[i]['line_img']
        input_batch[i,:,:b_img.shape[1],:] = b_img

        l = batch[i]['gt']
        all_labels.append(l)
        label_lengths.append(len(l))

    #all_labels = np.concatenate(all_labels)
    all_labels = np.array(all_labels)
    #print("ALL", all_labels.shape)
    label_lengths = np.array(label_lengths)

    line_imgs = input_batch.transpose([0,3,1,2]) # batch, channel, h, w
    line_imgs = torch.from_numpy(line_imgs).to(device)

    labels = torch.from_numpy(all_labels.astype(np.float32)).to(device)
    label_lengths = torch.from_numpy(label_lengths.astype(np.int32)).to(device)

    return {
        "line_imgs": line_imgs,
        "gt": labels,
        "label_lengths": label_lengths,
        "paths": [b["path"] for b in batch],
    }

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
        #label_lengths.append(len(l))
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
        "label_lengths": label_lengths,
        "paths": [b["path"] for b in batch],
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

def put_at(start, stop, axis=1):
    if axis < 0:
        return (Ellipsis, ) + (slice(start, stop),) + (slice(None),) * abs(-1-axis)
    else:
        return (slice(None),) * (axis) + (slice(start,stop),)

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
    y = test_padding(the_list, pad2)
    assert np.allclose(x,y)