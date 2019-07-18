from __future__ import print_function
from builtins import range
import faulthandler
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

### TO DO:
# Add ONLINE flag to regular CRNN
# Download updated JSONs/Processing


# EMAIL SUPERCOMPUTER?
# "right" way to make an embedding
# CycleGAN - threshold
# Deepwriting - clean up generated images?
# Dropout schedule

from warpctc_pytorch import CTCLoss
import error_rates
import string_utils
from torch.nn import CrossEntropyLoss
import traceback

import visualize
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import time
from utils import *
from torch.optim import lr_scheduler
from crnn import Stat

## Notes on usage
# conda activate hw2
# python -m visdom.server -p 8080


faulthandler.enable()

def test(model, dataloader, idx_to_char, dtype, config, with_analysis=False):
    sum_loss = 0.0
    steps = 0.0
    model.eval()
    for x in dataloader:
        line_imgs = Variable(x['line_imgs'].type(dtype), requires_grad=False)
        gt = x['gt'] # actual string ground truth
        online = Variable(x['online'].type(dtype), requires_grad=False).view(1, -1, 1) if config["online_augmentation"] else None
        config["trainer"].test(line_imgs, online, gt)

        # Only do one test
        if config["TESTING"]:
            break

    accumulate_stats(config)
    test_cer = config["stats"][config["designated_test_cer"]].y[-1] # most recent test CER

    LOGGER.debug(config["stats"])
    return test_cer

def plot_images(line_imgs, name):
    # Save images
    f, axarr = plt.subplots(len(line_imgs), 1)
    if len(line_imgs) > 1:
        for j, img in enumerate(line_imgs):
            axarr[j].imshow(img.squeeze().detach().numpy(), cmap='gray')
    else:
        axarr.imshow(line_imgs.squeeze().detach().numpy(), cmap='gray')

    # plt.show()
    path = os.path.join(config["image_dir"], '{}.png'.format(name))
    plt.savefig(path)


def run_epoch(model, dataloader, ctc_criterion, optimizer, dtype, config):
    model.train()
    config["stats"]["epochs"] += [config["current_epoch"]]
    plot_freq = config["plot_freq"]

    for i, x in enumerate(dataloader):
        LOGGER.debug("Training Iteration: {}".format(i))
        line_imgs = Variable(x['line_imgs'].type(dtype), requires_grad=config["improve_image"])
        labels = Variable(x['labels'], requires_grad=False) # numeric indices version of ground truth
        label_lengths = Variable(x['label_lengths'], requires_grad=False)
        gt = x['gt'] # actual string ground truth
        config["global_step"] += 1
        config["global_instances_counter"] += line_imgs.shape[0]
        config["stats"]["instances"] += [config["global_instances_counter"]]

        # Add online/offline binary flag
        online = Variable(x['online'].type(dtype), requires_grad=False).view(1, -1, 1) if config["online_augmentation"] else None

        config["trainer"].train(line_imgs, online, labels, label_lengths, gt, step=config["global_step"])

        if config["improve_image"]:
            plot_images(x['line_imgs'], "{}_original".format(i))
            plot_images(line_imgs, i)
            print(torch.abs(x['line_imgs']-line_imgs).sum())



        # Update visdom every 50 instances
        if config["global_step"] % plot_freq == 0 and config["global_step"] > 0:
            config["stats"]["updates"] += [config["global_step"]]
            config["stats"]["epoch_decimal"] += [config["current_epoch"]+ i * config["batch_size"] * 1.0 / config['n_train_instances']]
            LOGGER.info("updates: {}".format(config["global_step"]))
            accumulate_stats(config)
            visualize.plot_all(config)

        if config["TESTING"] or config["SMALL_TRAINING"]:
            config["stats"]["updates"] += [config["global_step"]]
            config["stats"]["epoch_decimal"] += [config["current_epoch"]+i * config["batch_size"] / config['n_train_instances']]
            LOGGER.info("updates: {}".format(config["global_step"]))
            accumulate_stats(config)
            break

    training_cer = config["stats"][config["designated_training_cer"]].y[-1] # most recent training CER
    LOGGER.debug(config["stats"])
    return training_cer

def make_dataloaders(config):
    train_dataset = HwDataset(config["training_jsons"], config["char_to_idx"], img_height=config["input_height"],
                              num_of_channels=config["num_of_channels"], root=config["training_root"],
                              warp=config["training_warp"], writer_id_paths=config["writer_id_pickles"])


    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=config["training_shuffle"], num_workers=0, collate_fn=hw_dataset.collate, pin_memory=True)

    test_dataset = HwDataset(config["testing_jsons"], config["char_to_idx"], img_height=config["input_height"], num_of_channels=config["num_of_channels"], root=config["testing_root"], warp=config["testing_warp"])
    test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=config["testing_shuffle"], num_workers=0, collate_fn=hw_dataset.collate)

    return train_dataloader, test_dataloader, train_dataset, test_dataset

def load_data(config):
    # Load characters and prep datasets
    config["char_to_idx"], config["idx_to_char"], config["char_freq"] = character_set.make_char_set(config['training_jsons'], root=config["training_root"])

    train_dataloader, test_dataloader, train_dataset, test_dataset = make_dataloaders(config=config)

    config['alphabet_size'] = len(config["idx_to_char"]) + 1 # alphabet size to be recognized
    config['num_of_writers'] = train_dataset.classes_count + 1

    config['n_train_instances'] = len(train_dataloader.dataset)
    log_print("Number of training instances:", config['n_train_instances'])
    log_print("Number of test instances:", len(test_dataloader.dataset), '\n')
    return train_dataloader, test_dataloader, train_dataset, test_dataset

def main():
    global config, LOGGER
    opts = parse_args()
    config = load_config(opts.config)
    LOGGER = config["logger"]
    config["global_step"] = 0
    config["global_instances_counter"] = 0
    
    # Use small batch size when using CPU/testing
    if config["TESTING"]:
        config["batch_size"] = 1

    # Prep data loaders
    train_dataloader, test_dataloader, train_dataset, test_dataset = load_data(config)

    # Prep optimizer
    criterion = CTCLoss()

    # Create classifier
    if config["style_encoder"] == "basic_encoder":
        hw = crnn.create_CRNNClassifier(config)
    elif config["style_encoder"] == "fake_encoder":
        hw = crnn.create_CRNNClassifier(config)
    elif config["style_encoder"] == "2Stage":
        hw = crnn.create_2Stage(config)
        config["embedding_size"]=0

    elif config["style_encoder"] == "2StageNudger":
        hw = crnn.create_CRNN(config)
        config["nudger"] = crnn.create_Nudger(config)
        config["embedding_size"]=0
        config["nudger_optimizer"] = torch.optim.Adam(config["nudger"].parameters(), lr=config['learning_rate'])

    else: # basic HWR
        config["embedding_size"]=0
        hw = crnn.create_CRNN(config)

    # Setup defaults
    config["starting_epoch"] = 1
    config["model"] = hw
    config['lowest_loss'] = float('inf')
    config["train_losses"] = []
    config["test_losses"] = []

    # Create optimizer
    optimizer = torch.optim.Adam(hw.parameters(), lr=config['learning_rate'])
    config["optimizer"] = optimizer

    scheduler = lr_scheduler.StepLR(optimizer, step_size=config["scheduler_step"], gamma=config["scheduler_gamma"])
    config["scheduler"] = scheduler

    ## LOAD FROM OLD MODEL
    if config["load_path"]:
        load_model(config)
        hw = config["model"]
        # DOES NOT LOAD OPTIMIZER, SCHEDULER, ETC?

    # Create trainer
    if config["style_encoder"] == "2StageNudger":
        train_baseline = False if config["load_path"] else True
        config["trainer"] = crnn.TrainerNudger(hw, config["nudger_optimizer"], config, criterion, train_baseline=train_baseline)
    else:
        config["trainer"] = crnn.TrainerBaseline(hw, optimizer, config, criterion)

    # Prep GPU
    if config["TESTING"]: # don't use GPU for testing
        dtype = torch.FloatTensor
        log_print("Testing mode, not using GPU")
        config["cuda"]=False
    elif torch.cuda.is_available():
        hw.cuda()
        if "nudger" in config.keys():
            config["nudger"].cuda()
        dtype = torch.cuda.FloatTensor
        log_print("Using GPU")
        config["cuda"] = True
    else:
        dtype = torch.FloatTensor
        log_print("No GPU detected")
        config["cuda"]=False

    # Alternative Models
    if config["style_encoder"]=="basic_encoder":
        config["secondary_criterion"] = CrossEntropyLoss()
    else: # config["style_encoder"] = False
        config["secondary_criterion"] = None

    # Launch visdom
    if config["use_visdom"]:
        visualize.initialize_visdom(config["full_specs"], config)

    # Stat prep - must be after visdom
    stat_prep(config)

    ## HACKISH IMAGE IMPROVER
    if config["improve_image"]:
        # hw.freeze()
        # Reset optimizer
        config['learning_rate'] = .1
        # Improve test images
        train_dataloader = test_dataloader

    for epoch in range(config["starting_epoch"], config["starting_epoch"]+config["epochs_to_run"]+1):
        LOGGER.info("Epoch: {}".format(epoch))
        config["current_epoch"] = epoch

        # Only test
        if not config["test_only"]:
            training_cer = run_epoch(hw, train_dataloader, criterion, optimizer, dtype, config)

            scheduler.step()

            LOGGER.info("Training CER: {}".format(training_cer))
            config["train_losses"].append(training_cer)

        # CER plot
        test_cer = test(hw, test_dataloader, config["idx_to_char"], dtype, config)
        LOGGER.info("Test CER: {}".format(test_cer))
        config["test_losses"].append(test_cer)

        if config["use_visdom"]:
            config["visdom_manager"].update_plot("Test Error Rate", [epoch], test_cer)

        if config["test_only"]:
            break

        if not config["results_dir"] is None:

            # Save BSF
            if config['lowest_loss'] > test_cer:
                config['lowest_loss'] = test_cer
                save_model(config, bsf=True)

            if epoch % config["save_freq"] == 0:
                save_model(config, bsf=False)

            plt_loss(config)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
        traceback.print_exc()
    finally:
        torch.cuda.empty_cache()

# https://github.com/theevann/visdom-save/blob/master/vis.py
