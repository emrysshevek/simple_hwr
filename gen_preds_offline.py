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
from hwr_utils.stroke_dataset import StrokeRecoveryDataset, BasicDataset
from hwr_utils.stroke_recovery import *
from hwr_utils import utils
from torch.optim import lr_scheduler
from timeit import default_timer as timer
from hwr_utils.utils import debugger
from train_stroke_recovery import StrokeRecoveryModel, parse_args, graph
from hwr_utils.hwr_logger import logger

#@debugger
def main(config_path):
    global epoch, device, trainer, batch_size, output, loss_obj, x_relative_positions, config, LOGGER
    torch.cuda.empty_cache()

    config = utils.load_config(config_path, hwr=False)
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
    folder = Path(config.dataset_folder)
    folder = Path("/media/data/GitHub/simple_hwr/data/prepare_IAM_Lines/lines/")
    model = StrokeRecoveryModel(vocab_size=vocab_size, device=device, cnn_type=config.cnn_type, first_conv_op=config.coordconv, first_conv_opts=config.coordconv_opts).to(device)
    config.model = model
    config.load_path = "/media/data/GitHub/simple_hwr/results/stroke_config/20191120_170400-baseline-GOOD,MAX/baseline_model.pt"
    ## LOAD THE WEIGHTS
    utils.load_model(config)
    model = model.to(device)

    logger.info(("Current dataset: ", folder))
    # Dataset - just expecting a folder
    eval_dataset=BasicDataset(root=folder)

    eval_loader=DataLoader(eval_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=6,
                                  collate_fn=eval_dataset.collate, # this should be set to collate_stroke_eval
                                  pin_memory=False)

    eval_only(eval_loader, model)

def eval_only(dataloader, model):
    for i, item in enumerate(dataloader):
        preds = TrainerStrokeRecovery.eval(item["line_imgs"], model)
        preds_to_graph = [p.permute([1, 0]) for p in preds]
        graph(item, preds=preds_to_graph, _type="eval", x_relative_positions=x_relative_positions, epoch="current", config=config)


if __name__=="__main__":
    opts = parse_args()
    main(config_path=opts.config)
