from pathlib import Path
import numpy as np
from hwr_utils import visualize
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
from hwr_utils.utils import print

from train_stroke_recovery import

def main():
    global epoch, device, trainer, batch_size, output, loss_obj, x_relative_positions, config, LOGGER
    torch.cuda.empty_cache()

    config = utils.load_config("./configs/stroke_config/baseline.yaml", hwr=False)
    LOGGER = config.logger
    test_size = config.test_size
    train_size = config.train_size
    batch_size = config.batch_size
    x_relative_positions = config.x_relative_positions
    if x_relative_positions == "both":
        raise Exception("Not implemented")
    vocab_size = config.vocab_size

    device=torch.device("cuda")
    #device=torch.device("cpu")

    #output = utils.increment_path(name="Run", base_path=Path("./results/stroke_recovery"))
    output = Path(config.results_dir)
    output.mkdir(parents=True, exist_ok=True)
    loss_obj = StrokeLoss(loss_fn=config.loss_fn)
    config.loss_obj = loss_obj
    # folder = Path("online_coordinate_data/3_stroke_32_v2")
    # folder = Path("online_coordinate_data/3_stroke_vSmall")
    # folder = Path("online_coordinate_data/3_stroke_vFull")
    # folder = Path("online_coordinate_data/8_stroke_vFull")
    # folder = Path("online_coordinate_data/8_stroke_vSmall_16")
    folder = Path(config.dataset_folder)


    model = StrokeRecoveryModel(vocab_size=vocab_size, device=device, first_conv_op=config.coordconv, first_conv_opts=config.coordconv_opts).to(device)
    cnn = model.cnn # if set to a cnn object, then it will resize the GTs to be the same size as the CNN output
    print("Current dataset: ", folder)
