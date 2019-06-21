import re
import json

import torch
from torch.utils.data import Dataset
from torch.autograd import Variable

from collections import defaultdict
import os
import cv2
import numpy as np

import random
import string_utils

import grid_distortion
from utils import unpickle_it
PADDING_CONSTANT = 0
ONLINE_JSON_PATH = ''

def collate(batch):
    batch = [b for b in batch if b is not None]
    #These all should be the same size or error
    if len(set([b['line_img'].shape[0] for b in batch])) > 1:
        print(batch)
    assert len(set([b['line_img'].shape[0] for b in batch])) == 1
    assert len(set([b['line_img'].shape[2] for b in batch])) == 1

    dim0 = batch[0]['line_img'].shape[0]
    dim1 = max([b['line_img'].shape[1] for b in batch])
    dim2 = batch[0]['line_img'].shape[2]

    all_labels = []
    label_lengths = []

    input_batch = np.full((len(batch), dim0, dim1, dim2), PADDING_CONSTANT).astype(np.float32)
    for i in range(len(batch)):
        b_img = batch[i]['line_img']
        input_batch[i,:,:b_img.shape[1],:] = b_img

        l = batch[i]['gt_label']
        all_labels.append(l)
        label_lengths.append(len(l))

    all_labels = np.concatenate(all_labels)
    label_lengths = np.array(label_lengths)

    line_imgs = input_batch.transpose([0,3,1,2])
    line_imgs = torch.from_numpy(line_imgs)
    labels = torch.from_numpy(all_labels.astype(np.int32))
    label_lengths = torch.from_numpy(label_lengths.astype(np.int32))

    return {
        "line_imgs": line_imgs,
        "labels": labels,
        "label_lengths": label_lengths,
        "gt": [b['gt'] for b in batch],
        "writer_id": torch.FloatTensor([b['writer_id'] for b in batch])
    }

class HwDataset(Dataset):
    def __init__(self, data_paths, char_to_idx, img_height=32, num_of_channels=3, root="./data", warp=False, writer_ids_pickle="./data/prepare_IAM_Lines/writer_IDs.pickle"):
        data = []
        for data_path in data_paths:
            with open(os.path.join(root, data_path)) as fp:
                data.extend(json.load(fp))

        data, self.classes_count = self.add_writer_ids(data, writer_ids_pickle)
        self.root = root
        self.img_height = img_height
        self.char_to_idx = char_to_idx
        self.data = data
        self.warp = warp
        self.num_of_channels = num_of_channels

    def add_writer_ids(self, data, writer_id_path):
        """
        Args:
            data (json type thing): hw-dataset_
            writer_id_path (str): Path to pickle dictionary of form {ID: [file1,file2...] ... }

        Returns:
            tuple: updated data with ID, number of classes
        """
        d = unpickle_it(writer_id_path)
        inverted_dict = dict([[v, k] for k, vs in d.items() for v in vs ])

        for i,item in enumerate(data):
            # Get writer ID from file
            p,child = os.path.split(item["image_path"])
            child = re.search("([a-z0-9]+-[a-z0-9]+)", child).group(1)
            item["writer_id"] = inverted_dict[child]
            data[i] = item
        return data, int(max(d.keys()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        image_path = os.path.join(self.root, item['image_path'])
        if self.num_of_channels == 3:
            img = cv2.imread(image_path)
        elif self.num_of_channels == 1: # read grayscale
            img = cv2.imread(image_path, 0)
        else:
            raise Exception("Unexpected number of channels")
        if img is None:
            print("Warning: image is None:", os.path.join(self.root, item['image_path']))
            return None

        percent = float(self.img_height) / img.shape[0]

        img = cv2.resize(img, (0,0), fx=percent, fy=percent, interpolation = cv2.INTER_CUBIC)

        # Add channel dimension, since resize only keeps non-trivial channel axis
        if self.num_of_channels==1:
            img=img[:,:, np.newaxis]

        if self.warp:
            img = grid_distortion.warp_image(img) 

        img = img.astype(np.float32)
        img = img / 128.0 - 1.0

        gt = item['gt'] # actual text
        gt_label = string_utils.str2label(gt, self.char_to_idx) # character indices of text
        return {
            "line_img": img,
            "gt_label": gt_label,
            "gt": gt,
            "writer_id": int(item['writer_id'])
        }
