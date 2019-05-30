from __future__ import print_function
from builtins import range

import json
import character_set
import sys
import hw_dataset
from hw_dataset import HwDataset
import crnn
import os
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from warpctc_pytorch import CTCLoss
from train import *
import error_rates
import string_utils

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    config_path = sys.argv[1]
    model_path = sys.argv[2]
    print(config_path, model_path)

    with open(config_path) as f:
        config = json.load(f)

    data_config = config['data']
    augment_config = config.get('augmentation')
    network_config = config['network']

    idx_to_char, char_to_idx = character_set.load_char_set(data_config['character_set_path'])
    if augment_config is not None:
        online_idx_to_char, online_char_to_idx = character_set.load_char_set(augment_config['character_set_path'])
        idx_to_char.update(online_idx_to_char)
        char_to_idx.update(online_char_to_idx)

    train_dataloader, test_dataloader = make_dataloaders(data_config, augment_config, network_config, char_to_idx,
                                                         config['warp'])
    hw = crnn.create_model({
        'cnn_out_size': config['network']['cnn_out_size'],
        'num_of_channels': 3,
        'num_of_outputs': len(idx_to_char) + 1
    })

    if torch.cuda.is_available():
        hw.cuda()
        dtype = torch.cuda.FloatTensor
        print("Using GPU")
    else:
        dtype = torch.FloatTensor
        print("No GPU detected")

    hw.load_state_dict(torch.load(model_path))

    test_loss = test(hw, test_dataloader, idx_to_char, dtype)
    print(test_loss)


if __name__ == "__main__":
    main()