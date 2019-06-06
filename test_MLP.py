from __future__ import print_function
from builtins import range

from utils import is_iterable
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

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from warpctc_pytorch import CTCLoss
import error_rates
import string_utils
from torch.nn import CrossEntropyLoss

import matplotlib
matplotlib.use('Agg')

mlp = crnn.MLP(input_size, classifier_output_dimension, hidden_layers, dropout=.8)

if torch.cuda.is_available():
    mlp.cuda()
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
criterion = CrossEntropyLoss()
