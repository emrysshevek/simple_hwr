from pathlib import Path
import numpy as np
from gitinfo import get_git_info
from hwr_utils import visualize
from torch.utils.data import DataLoader
from models.basic import CNN, BidirectionalRNN
from models.attention import MultiLayerSelfAttention
from torch import nn
from models.stroke_recovery_loss import StrokeLoss
from models.stroke_recovery_model import StrokeRecoveryModel
import torch
from models.CoordConv import CoordConv
from crnn import TrainerStrokeRecovery
from hwr_utils.stroke_dataset import StrokeRecoveryDataset
from hwr_utils.stroke_recovery import *
from hwr_utils import utils
from torch.optim import lr_scheduler
from timeit import default_timer as timer
import argparse
import logging
from hwr_utils.hwr_logger import logger

torch.cuda.empty_cache()
utils.kill_gpu_hogs()

## Change CWD to the folder containing this script
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
os.chdir(ROOT_DIR)

## Variations:
# Relative position
# CoordConv - 0 center, X-as-rectanlge
# L1 loss, DTW
# Dataset size


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="./configs/stroke_config/baseline.yaml", help='Path to the config file.')
    # parser.add_argument('--name', type=str, default="", help='Optional - special name for this run')
    opts = parser.parse_args()
    return opts


def run_epoch(dataloader, report_freq=500):
    loss_list = []

    # for i in range(0, 16):
    #     line_imgs = torch.rand(batch, 1, 60, 60)
    #     targs = torch.rand(batch, 16, 5)
    instances = 0
    start_time = timer()
    logger.info(("Epoch: ", epoch))

    for i, item in enumerate(dataloader):
        current_batch_size = item["line_imgs"].shape[0]
        instances += current_batch_size
        loss, preds, *_ = trainer.train(item, train=True)

        loss_list += [loss]
        config.stats["Actual_Loss_Function_train"].accumulate(loss)

        if config.counter.updates % report_freq == 0 and i > 0:
            utils.reset_all_stats(config, keyword="_train")
            logger.info(("update: ", config.counter.updates, "combined loss: ", config.stats["Actual_Loss_Function_train"].get_last()))

    end_time = timer()
    logger.info(("Epoch duration:", end_time-start_time))

    #preds_to_graph = preds.permute([0, 2, 1])
    preds_to_graph = [p.permute([1, 0]) for p in preds]
    graph(item, config=config, preds=preds_to_graph, _type="train", x_relative_positions=config.x_relative_positions, epoch=epoch)
    config.scheduler.step()
    return np.sum(loss_list) / config.n_train_instances

def test(dataloader):
    for i, item in enumerate(dataloader):
        loss, preds, *_ = trainer.test(item)
        config.stats["Actual_Loss_Function_test"].accumulate(loss)
    preds_to_graph = [p.permute([1, 0]) for p in preds]
    graph(item, config=config, preds=preds_to_graph, _type="test", x_relative_positions=config.x_relative_positions, epoch=epoch)
    utils.reset_all_stats(config, keyword="_test")

    return config.stats["Actual_Loss_Function_test"].get_last()

def graph(batch, config=None, preds=None, _type="test", save_folder=None, x_relative_positions=False, epoch="current"):
    if save_folder is None:
        _epoch = str(epoch)
        save_folder = (config.image_dir / _epoch / _type)
    else:
        save_folder = Path(save_folder)

    save_folder.mkdir(parents=True, exist_ok=True)

    def subgraph(coords, img_path, name, is_gt=True):
        if not is_gt:
            if coords is None:
                return
            coords = utils.to_numpy(coords[i])
            #print("before round", coords[2])
            coords[2:, :] = np.round(coords[2:, :]) # VOCAB SIZE, LENGTH
            #print("after round", coords[2])

            suffix=""
        else:
            suffix="_gt"
            coords = utils.to_numpy(coords).transpose() # LENGTH, VOCAB => VOCAB SIZE, LENGTH

        ## Undo relative positions for X for graphing
        if x_relative_positions:
            coords[0] = relativefy_numpy(coords[0], reverse=True)

        render_points_on_image(gts=coords, img_path=img_path, save_path=save_folder / f"temp{i}_{name}{suffix}.png")

    # Loop through each item in batch
    for i, el in enumerate(batch["paths"]):
        img_path = el
        name=Path(batch["paths"][i]).stem
        if _type != "eval":
            subgraph(batch["gt_list"][i], img_path, name, is_gt=True)
        subgraph(preds, img_path, name, is_gt=False)
        if i > 8:
            break


def main(config_path):
    logger.info(f'Using commit: {get_git_info()["commit"]}')
    logger.info(f"Commit date: {get_git_info()['author_date']}")

    global epoch, device, trainer, batch_size, output, loss_obj, config, LOGGER
    torch.cuda.empty_cache()
    config = utils.load_config(config_path, hwr=False)
    test_size = config.test_size
    train_size = config.train_size
    batch_size = config.batch_size
    vocab_size = config.vocab_size

    device = torch.device("cuda" if torch.cuda.is_available() and config.gpu else "cpu")  # cpu, cuda

    #output = utils.increment_path(name="Run", base_path=Path("./results/stroke_recovery"))
    output = Path(config.results_dir)
    output.mkdir(parents=True, exist_ok=True)
    # folder = Path("online_coordinate_data/3_stroke_32_v2")
    # folder = Path("online_coordinate_data/3_stroke_vSmall")
    # folder = Path("online_coordinate_data/3_stroke_vFull")
    # folder = Path("online_coordinate_data/8_stroke_vFull")
    # folder = Path("online_coordinate_data/8_stroke_vSmall_16")
    folder = Path(config.dataset_folder)

    model = StrokeRecoveryModel(vocab_size=vocab_size, device=device, cnn_type=config.cnn_type, first_conv_op=config.coordconv, first_conv_opts=config.coordconv_opts).to(device)
    cnn = model.get_cnn()  # if set to a cnn object, then it will resize the GTs to be the same size as the CNN output
    logger.info(("Current dataset:", folder))

    ## LOAD DATASET
    train_dataset = StrokeRecoveryDataset([folder / "train_online_coords.json"],
                            img_height = 60,
                            num_of_channels = 1,
                            root=config.data_root,
                            max_images_to_load = train_size,
                            x_relative_positions=config.x_relative_positions,
                            cnn=cnn,
                            device=device
                            )

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=6,
                                  collate_fn=train_dataset.collate,
                                  pin_memory=False)

    config.n_train_instances = len(train_dataloader.dataset)

    test_dataset = StrokeRecoveryDataset([folder / "test_online_coords.json"],
                            img_height = 60,
                            num_of_channels = 1.,
                            root=config.data_root,
                            max_images_to_load=test_size,
                            x_relative_positions=config.x_relative_positions,
                            cnn=cnn,
                            device=device
                            )

    test_dataloader = DataLoader(test_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=3,
                                  collate_fn=train_dataset.collate,
                                  pin_memory=False)

    n_test_points = 0
    for i in test_dataloader:
        n_test_points += sum(i["label_lengths"])
    config.n_test_instances = len(test_dataloader.dataset)
    config.n_test_points = n_test_points
    # example = next(iter(test_dataloader)) # BATCH, WIDTH, VOCAB
    # vocab_size = example["gt"].shape[-1]

    ## Stats
    if config.use_visdom:
        visualize.initialize_visdom(config["full_specs"], config)
    utils.stat_prep_strokes(config)

    # Create loss object
    config.loss_obj = StrokeLoss(loss_names=config.loss_fns, loss_stats=config.stats, counter=config.counter)
    optimizer = torch.optim.Adam(model.parameters(), lr=.0005 * batch_size/32)

    config.scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=.95)
    trainer = TrainerStrokeRecovery(model, optimizer, config=config, loss_criterion=config.loss_obj, device=device)

    config.optimizer = optimizer
    config.trainer = trainer
    config.model = model
    torch.autograd.set_detect_anomaly(True)

    for i in range(0,200):
        epoch = i+1
        #config.counter.epochs = epoch
        config.counter.update(epochs=1)
        loss = run_epoch(train_dataloader, report_freq=config.update_freq)
        logger.info(f"Epoch: {epoch}, Training Loss: {loss}")
        test_loss = test(test_dataloader)
        logger.info(f"Epoch: {epoch}, Test Loss: {test_loss}")
        if config.first_loss_epochs and epoch == config.first_loss_epochs:
            config.loss_obj.set_loss(config.loss_fns2)

        if epoch % config.save_freq == 0: # how often to save
            utils.save_model_stroke(config, bsf=False)

    ## Bezier curve
    # Have network predict whether it has reached the end of a stroke or not
    # If it has not reached the end of a stroke, the starting point = previous end point


if __name__=="__main__":
    opts = parse_args()
    main(config_path=opts.config)
    
    # TO DO:
        # logging
        # Get running on super computer - copy the data!
