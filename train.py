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
# SAVE/LOAD VISDOM
# Make fast testing version
# Make version without embedding in CONFIG
# Merge in MASON's STUFF
# VISDOM GRAPH OVERLAY - integrate with matplotlib

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

wait_for_gpu()

visdom_manager = visualize.Plot("Loss")

hwr_loss = "HWR Loss"
w_loss = "Writer Loss"
t_loss = "Total Loss"
visdom_manager.register_plot(hwr_loss, "Epoch", "Loss")
visdom_manager.register_plot(w_loss, "Epoch", "Loss")
visdom_manager.register_plot(t_loss, "Epoch", "Loss")
visdom_manager.register_plot("Test Error Rate", "Epoch", "Loss")
GLOBAL_COUNTER = 0

# conda activate hw2
# python -m visdom.server -p 8080

def plot_loss(epoch, _hwr_loss, _writer_loss):
    #print(_hwr_loss, _writer_loss)
    visdom_manager.update_plot(hwr_loss, [epoch], _hwr_loss)
    visdom_manager.update_plot(w_loss, [epoch], _writer_loss)
    visdom_manager.update_plot(t_loss, [epoch], np.add(_hwr_loss,_writer_loss))

def test(model, dataloader, idx_to_char, dtype):
    sum_loss = 0.0
    steps = 0.0
    model.eval()
    for x in dataloader:
        with torch.no_grad():
            line_imgs = Variable(x['line_imgs'].type(dtype), requires_grad=False)
            labels = Variable(x['labels'], requires_grad=False)
            label_lengths = Variable(x['label_lengths'], requires_grad=False)

        # Returns letter_predictions, writer_predictions
        preds, _ = model(line_imgs)

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
    test_cer = sum_loss / steps
    return test_cer

def run_epoch(model, dataloader, ctc_criterion, optimizer, idx_to_char, dtype, secondary_criterion=None):
    global GLOBAL_COUNTER
    sum_loss = 0.0
    steps = 0.0
    model.train()

    loss_history_recognizer = []
    loss_history_writer = []

    for i, x in enumerate(dataloader):
        line_imgs = Variable(x['line_imgs'].type(dtype), requires_grad=False)
        labels = Variable(x['labels'], requires_grad=False)
        label_lengths = Variable(x['label_lengths'], requires_grad=False)
        GLOBAL_COUNTER += 1
        plot_freq = 10

        ## With writer classification
        if type(model).__name__ == "CRNNClassifier" or not secondary_criterion is None:
            labels_author = Variable(x['writer_id'], requires_grad=False).long()

            pred_text, pred_author = [x.cpu() for x in model(line_imgs)]
            preds_size = Variable(torch.IntTensor([pred_text.size(0)] * pred_text.size(1)))

            output_batch = pred_text.permute(1, 0, 2)
            out = output_batch.data.cpu().numpy()

            loss_recognizer = ctc_criterion(pred_text, labels, preds_size, label_lengths)

            # pred_author: batch size * number of authors
            # labels_author: batch size (each element is an index)
            loss_author = secondary_criterion(pred_author, labels_author)

            #print(loss_recognizer.cpu(), loss_recognizer.cpu().shape)
            loss_history_recognizer.append(torch.mean(loss_recognizer.cpu(), 0, keepdim=False).item())
            loss_history_writer.append(    torch.mean(loss_author.cpu(),     0, keepdim=False).item())

            # rescale writer loss function
            if loss_history_writer[-1] < .05:
                total_loss = loss_recognizer
            else:
                total_loss = (loss_recognizer + loss_author * loss_history_recognizer[-1]/loss_history_writer[-1])/ 2

            # Plot it with visdom
            if i % plot_freq == 0 and i > 0:
                print(GLOBAL_COUNTER)
                plot_loss(GLOBAL_COUNTER, np.mean(loss_history_recognizer[-plot_freq:]), np.mean(loss_history_writer[-plot_freq:]))


        else:
            preds = model(line_imgs).cpu()
            preds_size = Variable(torch.IntTensor([preds.size(0)] * preds.size(1)))

            output_batch = preds.permute(1, 0, 2)
            out = output_batch.data.cpu().numpy()
            total_loss = ctc_criterion(preds, labels, preds_size, label_lengths)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        # if i == 0:
        #    for i in xrange(out.shape[0]):
        #        pred, pred_raw = string_utils.naive_decode(out[i,...])
        #        pred_str = string_utils.label2str(pred_raw, idx_to_char, True)
        #        print(pred_str)

        for j in range(out.shape[0]):
            logits = out[j, ...]
            pred, raw_pred = string_utils.naive_decode(logits)
            pred_str = string_utils.label2str(pred, idx_to_char, False)
            gt_str = x['gt'][j]
            cer = error_rates.cer(gt_str, pred_str)
            sum_loss += cer
            steps += 1

    training_cer = sum_loss / steps

    return training_cer


def make_dataloaders(train_json, train_root, test_json, test_root, char_to_idx, img_height, warp, shuffle_train=False, shuffle_test=False):

    train_dataset = HwDataset(train_json, char_to_idx, root=train_root, img_height=img_height, warp=warp)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=shuffle_train, num_workers=0, collate_fn=hw_dataset.collate)

    test_dataset = HwDataset(test_json, char_to_idx, root=test_root, img_height=img_height)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=shuffle_test, num_workers=0, collate_fn=hw_dataset.collate)

    return train_dataloader, test_dataloader, train_dataset, test_dataset

def main():
    config = load_config(sys.argv[1])

    # Load characters and prep datasets
    char_to_idx, idx_to_char, char_freq = character_set.make_char_set(config['training_jsons'])
    train_dataloader, test_dataloader, train_dataset, test_dataset = make_dataloaders(
        train_json=config['training_jsons'], train_root=config['training_root'], test_json=config['testing_jsons'], test_root=config['testing_root'], char_to_idx=char_to_idx,
        img_height=config['input_height'], warp=config['warp'], shuffle_train=config['training_suffle'], shuffle_test=config['testing_suffle']
    )
    config['alphabet_size'] = len(idx_to_char) + 1 # alphabet size to be recognized
    config['num_of_writers'] = train_dataset.classes_count + 1

    n_train_instances = len(train_dataloader.dataset)
    log_print("Number of training instances:", n_train_instances)
    log_print("Number of test instances:", len(test_dataloader.dataset), '\n')

    # Create classifier
    hw = crnn.create_CRNNClassifier(config)

    if torch.cuda.is_available():
        hw.cuda()
        dtype = torch.cuda.FloatTensor
        log_print("Using GPU")
    else:
        dtype = torch.FloatTensor
        log_print("No GPU detected")

    optimizer = torch.optim.Adam(hw.parameters(), lr=config['learning_rate'])
    criterion = CTCLoss()
    secondary_criterion = CrossEntropyLoss()

    lowest_loss = float('inf')
    train_losses = []
    test_losses = []
    for epoch in range(1, 101):
        log_print("Epoch ", epoch)
        training_cer = run_epoch(hw, train_dataloader, criterion, optimizer, idx_to_char, dtype, secondary_criterion=secondary_criterion)

        log_print("Training CER", training_cer)
        train_losses.append(training_cer)

        test_cer = test(hw, test_dataloader, idx_to_char, dtype)
        log_print("Test CER", test_cer)
        test_losses.append(test_cer)
        visdom_manager.update_plot("Test Error Rate", [epoch], test_cer)

        if lowest_loss > test_cer:
            lowest_loss = test_cer
            log_print("Saving Best")
            torch.save(hw.state_dict(), os.path.join(config["results_dir"], config['name'] + "_model.pt"))

        results = {'train': train_losses, 'test': test_losses}
        with open(os.path.join(config["results_dir"], "losses.json"), 'w') as fh:
            json.dump(results, fh, indent=4)

        x_axis = [(i+1) * n_train_instances for i in range(epoch)]
        plt.figure()
        plt.plot(x_axis, train_losses, label='train')
        plt.plot(x_axis, test_losses, label='test')
        plt.legend()
        plt.ylabel("CER")
        plt.xlabel("Number of Instances")
        plt.title("CER Loss")
        plt.savefig(os.path.join(config["results_dir"], config['name'] + ".png"))
        plt.close()

if __name__ == "__main__":
    main()


# https://github.com/theevann/visdom-save/blob/master/vis.py