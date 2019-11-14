from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from models.basic import CNN, BidirectionalRNN
from torch import nn
from models.stroke_recovery_loss import StrokeLoss
import torch
from models.CoordConv import CoordConv
from crnn import TrainerStrokeRecovery
from hwr_utils.stroke_dataset import StrokeRecoveryDataset
from hwr_utils.stroke_recovery import *
from hwr_utils import utils
from torch.optim import lr_scheduler
from timeit import default_timer as timer
from train_stroke_recovery import graph

folder = Path("online_coordinate_data/3_stroke_vSmall")
folder = Path("online_coordinate_data/8_stroke_vSmall_16")


x_relative_positions = True
test_size = 2000
train_size = None
batch_size=32

test_dataset=StrokeRecoveryDataset([folder / "test_online_coords.json"],
                        img_height = 60,
                        num_of_channels = 1.,
                        max_images_to_load = test_size,
                        root=r"../data",
                        x_relative_positions=x_relative_positions
                        )

test_dataloader = DataLoader(test_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=3,
                              collate_fn=test_dataset.collate,
                              pin_memory=False)

device="cuda"
example = next(iter(test_dataloader)) # BATCH, WIDTH, VOCAB
output = Path("./TEST_GRAPHING")

for i in output.rglob("*"):
    i.unlink()

graph(example, save_folder=output, x_relative_positions=x_relative_positions)