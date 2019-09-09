import re
import json

import torch
from torch.utils.data import Dataset

import os
import cv2
import numpy as np

from utils import grid_distortion, string_utils
from utils.utils import unpickle_it
from utils.character_set import PAD_IDX

ONLINE_JSON_PATH = ''


def seq2seq_collate(batch, sos, eos, pad, device='cpu'):
    batch = [b for b in batch if b is not None]
    # These all should be the same size or error
    if len(set([b['line_img'].shape[0] for b in batch])) > 1:
        print(batch)
    assert len(set([b['line_img'].shape[0] for b in batch])) == 1
    assert len(set([b['line_img'].shape[2] for b in batch])) == 1

    dim0 = batch[0]['line_img'].shape[0]
    dim1 = max([b['line_img'].shape[1] for b in batch])
    dim2 = batch[0]['line_img'].shape[2]

    all_labels = []
    label_lengths = []
    img_pad_value = 0
    input_batch = np.full((len(batch), dim0, dim1, dim2), img_pad_value).astype(np.float32)
    for i in range(len(batch)):
        b_img = batch[i]['line_img']
        input_batch[i, :, :b_img.shape[1], :] = b_img

        l = batch[i]['gt_label']
        all_labels.append(np.pad(l, 1, mode='constant', constant_values=(sos, eos)))
        label_lengths.append(len(l))

    max_label_len = max(len(label) for label in all_labels)
    all_labels = [np.pad(label, (0, max_label_len-len(label)), mode='constant', constant_values=pad) for label in all_labels]
    all_labels = np.stack(all_labels)
    label_lengths = np.array(label_lengths)

    line_imgs = input_batch.transpose([0, 3, 1, 2])
    line_imgs = torch.from_numpy(line_imgs).to(device)
    labels = torch.from_numpy(all_labels.astype(np.int32)).to(device)
    label_lengths = torch.from_numpy(label_lengths.astype(np.int32)).to(device)
    online = torch.from_numpy(np.array([1 if b['online'] else 0 for b in batch])).float().to(device)

    return {
        "line_imgs": line_imgs,
        "labels": labels,
        "label_lengths": label_lengths,
        "gt": [b['gt'] for b in batch],
        "writer_id": torch.FloatTensor([b['writer_id'] for b in batch]),
        "actual_writer_id": torch.FloatTensor([b['actual_writer_id'] for b in batch]),
        "paths": [b["path"] for b in batch],
        "online": online
    }

def collate(batch, device="cpu"):
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
    pad_value = 0
    input_batch = np.full((len(batch), dim0, dim1, dim2), pad_value).astype(np.float32)
    for i in range(len(batch)):
        b_img = batch[i]['line_img']
        input_batch[i,:,:b_img.shape[1],:] = b_img

        l = batch[i]['gt_label']
        all_labels.append(l)
        label_lengths.append(len(l))

    all_labels = np.concatenate(all_labels)
    label_lengths = np.array(label_lengths)

    line_imgs = input_batch.transpose([0,3,1,2])
    line_imgs = torch.from_numpy(line_imgs).to(device)
    labels = torch.from_numpy(all_labels.astype(np.int32)).to(device)
    label_lengths = torch.from_numpy(label_lengths.astype(np.int32)).to(device)
    online = torch.from_numpy(np.array([1 if b['online'] else 0 for b in batch])).float().to(device)

    return {
        "line_imgs": line_imgs,
        "labels": labels,
        "label_lengths": label_lengths,
        "gt": [b['gt'] for b in batch],
        "writer_id": torch.FloatTensor([b['writer_id'] for b in batch]),
        "actual_writer_id": torch.FloatTensor([b['actual_writer_id'] for b in batch]),
        "paths": [b["path"] for b in batch],
        "online": online
    }

class HwDataset(Dataset):
    def __init__(self, data_paths, char_to_idx, img_height=32, num_of_channels=3, root="./data", warp=False, writer_id_paths=("prepare_IAM_Lines/writer_IDs.pickle",)):
        data = []
        for data_path in data_paths:
            with open(os.path.join(root, data_path)) as fp:
                data.extend(json.load(fp))

        ## Read in all writer IDs
        writer_id_dict = {}
        for writer_id_file in writer_id_paths:
            path = os.path.join(root, writer_id_file)
            # Add files to create super dicitonary
            d = unpickle_it(path)
            writer_id_dict = {**writer_id_dict, **d}

        data, self.classes_count = self.add_writer_ids(data, writer_id_dict)

        self.root = root
        self.img_height = img_height
        self.char_to_idx = char_to_idx
        self.data = data
        self.warp = warp
        self.num_of_channels = num_of_channels

    def add_writer_ids(self, data, writer_dict):
        """

        Args:
            data (json type thing): hw-dataset_
            writer_id_path (str): Path to pickle dictionary of form {Writer_ID: [file_path1,file_path2...] ... }

        Returns:
            tuple: updated data with ID, number of classes
        """
        actual_ids = dict([[v, k] for k, vs in writer_dict.items() for v in vs ]) # {Partial path : Writer ID}
        writer_ids = dict([[v, k] for k, (_, vs) in enumerate(writer_dict.items()) for v in vs])  # {Partial path : IDX}

        for i,item in enumerate(data):
            # Get writer ID from file
            p,child = os.path.split(item["image_path"])
            child = re.search("([a-z0-9]+-[0-9]+)", child).group(1)[0:7] # take the first 7 characters
            item["actual_writer_id"] = actual_ids[child]
            item["writer_id"] = writer_ids[child]
            data[i] = item

        return data, len(set(writer_dict.keys())) # returns dictionary and number of writers

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

        #if random.randint(0, 1):
        #    img = cv2.resize(img, (0,0), fx=percent, fy=percent, interpolation = cv2.INTER_CUBIC)
        #else:
        img = cv2.resize(img, (0, 0), fx=percent, fy=percent, interpolation=cv2.INTER_CUBIC)

        if self.warp:
            img = grid_distortion.warp_image(img)

        # Add channel dimension, since resize and warp only keep non-trivial channel axis
        if self.num_of_channels==1:
            img=img[:,:, np.newaxis]

        img = img.astype(np.float32)
        img = img / 128.0 - 1.0


        gt = item['gt'] # actual text
        gt_label = string_utils.str2label(gt, self.char_to_idx) # character indices of text
        #online = item.get('online', False)
        # THIS IS A HACK, FIX THIS (below)
        online = int(item['actual_writer_id']) > 700
        
        return {
            "line_img": img,
            "gt_label": gt_label,
            "gt": gt,
            "actual_writer_id": int(item['actual_writer_id']),
            "writer_id": int(item['writer_id']),
            "path": image_path,
            "online": online
        }
