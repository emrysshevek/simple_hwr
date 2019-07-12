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


wait_for_gpu()
faulthandler.enable()

def test(model, dataloader, idx_to_char, dtype, config, with_analysis=False):
    if with_analysis:
        charAcc = CharAcc(config["char_to_idx"])
        cers = []
        lens = []

    sum_loss = 0.0
    steps = 0.0
    model.eval()
    for x in dataloader:
        with torch.no_grad():
            line_imgs = Variable(x['line_imgs'].type(dtype), requires_grad=False)
            labels = Variable(x['labels'], requires_grad=False)
            label_lengths = Variable(x['label_lengths'], requires_grad=False)
            online = Variable(x['online'].type(dtype), requires_grad=False).view(1, -1, 1) if config[
                "online_augmentation"] else None
        # Returns letter_predictions, writer_predictions
        #print(online, x['online'])
        preds, *_ = model(line_imgs, online)
   
        preds = preds.cpu()
        output_batch = preds.permute(1, 0, 2)
        out = output_batch.data.cpu().numpy()

        for i, gt_line in enumerate(x['gt']):
            logits = out[i, ...]
            pred, raw_pred = string_utils.naive_decode(logits)
            pred_str = string_utils.label2str(pred, idx_to_char, False)
            cer = error_rates.cer(gt_line, pred_str)

            if with_analysis:
                charAcc.char_accuracy(pred_str, gt_line)
                cers.append(cer)
                lens.append(len(gt_line))
                if cer > 0 and config["output_predictions"]:
                    write_out(config["results_dir"], "incorrect.txt", "\n".join((str(cer), pred_str, gt_line, os.path.abspath(x['paths'][i]))))
                if cer > 0.3 and config["output_predictions"]:
                    write_out(config["results_dir"], "very_incorrect.txt", "\n".join((str(cer), pred_str, gt_line, os.path.abspath(x['paths'][i]))))

            sum_loss += cer
            steps += 1

        # Only do one test
        if config["TESTING"]:
            break

    if with_analysis:
        # Plot accuracy by character
        chars = [m[0] for m in sorted(config["char_to_idx"].items(), key=lambda kv: kv[1])]

        #print(charAcc.correct, charAcc.false_negative, charAcc.false_positive)
        plt.bar(chars, charAcc.correct/(charAcc.correct+charAcc.false_negative)) # recall
        plt.title("Recall")
        plt.show()

        plt.bar(chars, charAcc.correct/(charAcc.correct+charAcc.false_positive)) # precision
        plt.title("Precision")
        plt.show()

        # Plot character error rate by line length
        plt.scatter(cers, lens)
        plt.show()

        # Plot 2d histogram of letters
        plt.hist2d(cers, lens, density=True)
        plt.show()

        m = list(zip(lens,cers))
        for x in range(0,100,10):
            to_plot = [p[1] for p in m if p[0] in range(x,x+10)]
            print(to_plot)
            plt.title("{} {}".format(x, x+10))
            plt.hist(to_plot, density=True)
            plt.show()


    test_cer = sum_loss / steps
    return test_cer

def run_epoch(model, dataloader, ctc_criterion, optimizer, dtype, config):
    model.train()
    config["stats"]["epochs"] += [config["current_epoch"]]

    for i, x in enumerate(dataloader):
        LOGGER.debug("Training Iteration: {}".format(i))
        line_imgs = Variable(x['line_imgs'].type(dtype), requires_grad=False)
        labels = Variable(x['labels'], requires_grad=False) # numeric indices version of ground truth
        label_lengths = Variable(x['label_lengths'], requires_grad=False)
        gt = x['gt'] # actual string ground truth

        config["global_counter"] += 1
        plot_freq = 50 if not config["TESTING"] else 1

        # Add online/offline binary flag
        online = Variable(x['online'].type(dtype), requires_grad=False).view(1, -1, 1) if config["online_augmentation"] else None

        if config["style_encoder"]=="2StageNudger":
            crnn.train_nudger(model, optimizer, config, line_imgs, online, labels, label_lengths, gt, ctc_criterion, step=config["global_counter"])
        elif not config["style_encoder"]:
            crnn.train_baseline(model, optimizer, config, line_imgs, online, labels, label_lengths, gt, ctc_criterion, step=config["global_counter"])

        # Plot it with visdom
        if config["global_counter"] % plot_freq == 0 and config["global_counter"] > 0:
            config["stats"]["instances"] += [config["global_counter"]]
            LOGGER.info(config["global_counter"])
            visualize.plot_all(config)

        if config["TESTING"] or config["SMALL_TRAINING"]:
            break

    accumulate_stats(config)
    training_cer = config["stats"]["Training Error Rate"].x[-1] # most recent training CER
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
    config["global_counter"] = 0

    # Use small batch size when using CPU/testing
    if config["TESTING"]:
        config["batch_size"] = 1

    # Prep data loaders
    train_dataloader, test_dataloader, train_dataset, test_dataset = load_data(config)

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

    else: # basic HWR
        config["embedding_size"]=0
        hw = crnn.create_CRNN(config)

    # Prep GPU
    if config["TESTING"]: # don't use GPU for testing
        dtype = torch.FloatTensor
        log_print("Testing mode, not using GPU")
    elif torch.cuda.is_available():
        hw.cuda()
        dtype = torch.cuda.FloatTensor
        log_print("Using GPU")
    else:
        dtype = torch.FloatTensor
        log_print("No GPU detected")

    # Prep optimizer
    optimizer = torch.optim.Adam(hw.parameters(), lr=config['learning_rate'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config["scheduler_step"], gamma=config["scheduler_gamma"])
    criterion = CTCLoss()

    # Alternative Models
    if config["style_encoder"]=="basic_encoder":
        config["secondary_criterion"] = CrossEntropyLoss()
    else: # config["style_encoder"] = False
        config["secondary_criterion"] = None

    # Setup defaults
    config["starting_epoch"] = 1
    config["optimizer"] = optimizer
    config["model"] = hw
    config['lowest_loss'] = float('inf')
    config["train_losses"] = []
    config["test_losses"] = []

    # Launch visdom
    if config["use_visdom"]:
        visualize.initialize_visdom(config["full_specs"], config)

    # Stat prep - must be after visdom
    stat_prep(config)


    ## LOAD FROM OLD MODEL
    if config["load_path"]:
        load_model(config)

    for epoch in range(config["starting_epoch"], config["starting_epoch"]+config["epochs_to_run"]+1):
        log_print("Epoch ", epoch)
        config["current_epoch"] = epoch

        # Only test
        if not config["test_only"]:
            training_cer = run_epoch(hw, train_dataloader, criterion, optimizer, dtype, config)

            scheduler.step()

            log_print("Training CER", training_cer)
            config["train_losses"].append(training_cer)

        # CER plot
        test_cer = test(hw, test_dataloader, config["idx_to_char"], dtype, config)
        log_print("Test CER", test_cer)
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
