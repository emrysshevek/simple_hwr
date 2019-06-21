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

### TO DO:
# Merge in MASON's STUFF

# EMAIL SUPERCOMPUTER?
# "right" way to make an embedding

# CycleGAN - threshold
# Deepwriting - clean up generated images?

# Dropout schedule

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from warpctc_pytorch import CTCLoss
import error_rates
import string_utils
from torch.nn import CrossEntropyLoss

import visualize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
from utils import *
from torch.optim import lr_scheduler

## Notes on usage
# conda activate hw2
# python -m visdom.server -p 8080

wait_for_gpu()


hwr_loss_title = "HWR Loss"
writer_loss_title = "Writer Loss"
total_loss_title = "Total Loss"

GLOBAL_COUNTER = 0

def initialize_visdom_manager(env_name, config):
    global visdom_manager
    visdom_manager = visualize.Plot("Loss", env_name=env_name, config=config)
    visdom_manager.register_plot(hwr_loss_title, "Instances", "Loss")
    visdom_manager.register_plot(writer_loss_title, "Instances", "Loss")
    visdom_manager.register_plot(total_loss_title, "Instances", "Loss")
    visdom_manager.register_plot("Test Error Rate", "Epoch", "Loss", ymax=.2)
    return visdom_manager

def plot_loss(epoch, _hwr_loss, _writer_loss=None, _total_loss=None):
    # Plot writer recognizer loss and total loss
    if _writer_loss is None:
        _writer_loss = 0
    if _total_loss is None:
        _total_loss = _hwr_loss

    # Plot regular losses
    visdom_manager.update_plot(hwr_loss_title, [epoch], _hwr_loss)
    visdom_manager.update_plot(writer_loss_title, [epoch], _writer_loss)
    visdom_manager.update_plot(total_loss_title, [epoch], _total_loss)

def test(model, dataloader, idx_to_char, dtype, config):
    sum_loss = 0.0
    steps = 0.0
    model.eval()
    for x in dataloader:
        with torch.no_grad():
            line_imgs = Variable(x['line_imgs'].type(dtype), requires_grad=False)
            labels = Variable(x['labels'], requires_grad=False)
            label_lengths = Variable(x['label_lengths'], requires_grad=False)

        # Returns letter_predictions, writer_predictions
        preds, *_ = model(line_imgs)

        preds = preds.cpu()
        output_batch = preds.permute(1, 0, 2)
        out = output_batch.data.cpu().numpy()

        for i, gt_line in enumerate(x['gt']):
            logits = out[i, ...]
            pred, raw_pred = string_utils.naive_decode(logits)
            pred_str = string_utils.label2str(pred, idx_to_char, False)
            cer = error_rates.cer(gt_line, pred_str)
            sum_loss += cer
            steps += 1
        if config["TESTING"]:
            break
    test_cer = sum_loss / steps
    return test_cer

def run_epoch(model, dataloader, ctc_criterion, optimizer, idx_to_char, dtype, config, secondary_criterion=None):
    global GLOBAL_COUNTER
    sum_loss = 0.0
    steps = 0.0
    model.train()

    crnn_with_classifier = type(model).__name__ == "CRNNClassifier"

    loss_history_recognizer = []
    loss_history_writer = []
    loss_history_total = loss_history_recognizer if not crnn_with_classifier else []

    for i, x in enumerate(dataloader):
        line_imgs = Variable(x['line_imgs'].type(dtype), requires_grad=False)
        labels = Variable(x['labels'], requires_grad=False)
        label_lengths = Variable(x['label_lengths'], requires_grad=False)
        GLOBAL_COUNTER += 1
        plot_freq = 50 if not config["TESTING"] else 1

        pred_text, *pred_author = [x.cpu() for x in model(line_imgs)]

        # Calculate HWR loss
        preds_size = Variable(torch.IntTensor([pred_text.size(0)] * pred_text.size(1)))

        output_batch = pred_text.permute(1, 0, 2)
        out = output_batch.data.cpu().numpy()

        loss_recognizer = ctc_criterion(pred_text, labels, preds_size, label_lengths)
        total_loss = loss_recognizer
        loss_history_recognizer.append(torch.mean(loss_recognizer.cpu(), 0, keepdim=False).item())

        ## With writer classification
        if not secondary_criterion is None:
            labels_author = Variable(x['writer_id'], requires_grad=False).long()

            # pred_author: batch size * number of authors
            # labels_author: batch size (each element is an author index number)
            loss_author = secondary_criterion(pred_author, labels_author)
            loss_history_writer.append(    torch.mean(loss_author.cpu(),     0, keepdim=False).item())

            # rescale writer loss function
            if loss_history_writer[-1] > .05:
                total_loss = (loss_recognizer + loss_author * loss_history_recognizer[-1]/loss_history_writer[-1])/ 2
            loss_history_total.append(torch.mean(total_loss.cpu(),     0, keepdim=False).item())

        # Plot it with visdom
        if GLOBAL_COUNTER % plot_freq == 0 and GLOBAL_COUNTER > 0:
            print(GLOBAL_COUNTER)

            # Writer loss
            plot_primary_loss = np.mean(loss_history_recognizer[-plot_freq:])

            if not secondary_criterion is None:
                plot_total_loss = np.mean(loss_history_total[-plot_freq:])
                plot_secondary_loss = np.mean(loss_history_writer[-plot_freq:])
            else:
                plot_total_loss = plot_primary_loss
                plot_secondary_loss = 0

            plot_loss(GLOBAL_COUNTER*config["batch_size"], plot_primary_loss, plot_secondary_loss, plot_total_loss)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # if i == 0:
        #    for i in xrange(out.shape[0]):
        #        pred, pred_raw = string_utils.naive_decode(out[i,...])
        #        pred_str = string_utils.label2str(pred_raw, idx_to_char, True)
        #        print(pred_str)

        # Calculate CER
        for j in range(out.shape[0]):
            logits = out[j, ...]
            pred, raw_pred = string_utils.naive_decode(logits)
            pred_str = string_utils.label2str(pred, idx_to_char, False)
            gt_str = x['gt'][j]
            cer = error_rates.cer(gt_str, pred_str)
            sum_loss += cer
            steps += 1

        if config["TESTING"]:
            break
    training_cer = sum_loss / steps

    return training_cer

def make_dataloaders(config):
    train_dataset = HwDataset(config["training_jsons"], config["char_to_idx"], img_height=config["input_height"], num_of_channels=config["num_of_channels"], root=config["training_root"], warp=config["warp"])
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=config["training_shuffle"], num_workers=2, collate_fn=hw_dataset.collate, pin_memory=True)

    test_dataset = HwDataset(config["testing_jsons"], config["char_to_idx"], img_height=config["input_height"], num_of_channels=config["num_of_channels"], root=config["testing_root"])
    test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=config["testing_shuffle"], num_workers=2, collate_fn=hw_dataset.collate)

    return train_dataloader, test_dataloader, train_dataset, test_dataset

def main():
    config = load_config(sys.argv[1])

    # Use small batch size when using CPU/testing
    if config["TESTING"]:
        config["batch_size"] = 1

    # Launch visdom
    initialize_visdom_manager(config["full_specs"], config)

    # Load characters and prep datasets
    config["char_to_idx"], config["idx_to_char"], config["char_freq"] = character_set.make_char_set(config['training_jsons'])

    train_dataloader, test_dataloader, train_dataset, test_dataset = make_dataloaders(config=config)
    #        train_json=config['training_jsons'], train_root=config['training_root'], test_json=config['testing_jsons'], test_root=config['testing_root'], char_to_idx=config["char_to_idx"],
    # img_height=config['input_height'], num_of_channels=config['num_of_channels'], warp=config['warp'], batch_size=config['shuffle_train=config['training_suffle'], shuffle_test=config['testing_suffle']

    config['alphabet_size'] = len(config["idx_to_char"]) + 1 # alphabet size to be recognized
    config['num_of_writers'] = train_dataset.classes_count + 1

    n_train_instances = len(train_dataloader.dataset)
    log_print("Number of training instances:", n_train_instances)
    log_print("Number of test instances:", len(test_dataloader.dataset), '\n')

    # Create classifier
    if config["style_encoder"] == "basic_encoder":
        hw = crnn.create_CRNNClassifier(config)
    elif config["style_encoder"] == "fake_encoder":
        hw = crnn.create_CRNNClassifier(config)
    else:
        hw = crnn.create_CRNN(config)

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

    optimizer = torch.optim.Adam(hw.parameters(), lr=config['learning_rate'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config["scheduler_step"], gamma=config["scheduler_gamma"])

    criterion = CTCLoss()

    # Alternative Models
    secondary_criterion = None
    if config["style_encoder"]=="basic_encoder":
        secondary_criterion = CrossEntropyLoss()
    else: # config["style_encoder"] = False
        secondary_criterion = None

    lowest_loss = float('inf')
    train_losses = []
    test_losses = []
    for epoch in range(1, config["epochs"]+1):
        log_print("Epoch ", epoch)
        training_cer = run_epoch(hw, train_dataloader, criterion, optimizer, config["idx_to_char"], dtype, config, secondary_criterion=secondary_criterion)
        scheduler.step()

        log_print("Training CER", training_cer)
        train_losses.append(training_cer)

        # CER plot
        test_cer = test(hw, test_dataloader, config["idx_to_char"], dtype, config)
        log_print("Test CER", test_cer)
        test_losses.append(test_cer)
        visdom_manager.update_plot("Test Error Rate", [epoch], test_cer)

        if not config["results_dir"] is None:
            # Save visdom graph
            visdom_manager.save_env()

            # Save the best model
            if lowest_loss > test_cer:
                lowest_loss = test_cer
                log_print("Saving Best")
                torch.save(hw.state_dict(), os.path.join(config["results_dir"], config['name'] + "_model.pt"))

            # Save losses/CER
            results = {'train': train_losses, 'test': test_losses}
            with open(os.path.join(config["results_dir"], "losses.json"), 'w') as fh:
                json.dump(results, fh, indent=4)

            ## Plot with matplotlib
            x_axis = [(i+1) * n_train_instances for i in range(epoch)]
            plt.figure()
            plt.plot(x_axis, train_losses, label='train')
            plt.plot(x_axis, test_losses, label='test')
            plt.legend()
            plt.ylim(top=.2)
            plt.ylabel("CER")
            plt.xlabel("Number of Instances")
            plt.title("CER Loss")
            plt.savefig(os.path.join(config["results_dir"], config['name'] + ".png"))
            plt.close()

if __name__ == "__main__":
    main()


# https://github.com/theevann/visdom-save/blob/master/vis.py