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
import error_rates
import string_utils

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


LOG_PATH = ""
log = True

def log_print(*args, **kwargs):
    new_start = kwargs.pop('new_start', False)
    print(*args, **kwargs)
    if log:
        mode = 'w' if not os.path.isfile(LOG_PATH) or new_start else 'a'
        with open(LOG_PATH, mode) as log_file:
            log_file.write(" ".join([str(word) for word in args[0:]]) + '\n')


def test(model, dataloader, idx_to_char, dtype):
    sum_loss = 0.0
    steps = 0.0
    model.eval()
    for x in dataloader:
        line_imgs = Variable(x['line_imgs'].type(dtype), requires_grad=False, volatile=True)
        labels = Variable(x['labels'], requires_grad=False, volatile=True)
        label_lengths = Variable(x['label_lengths'], requires_grad=False, volatile=True)

        preds = model(line_imgs).cpu()

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


def run_epoch(model, dataloader, criterion, optimizer, idx_to_char, dtype):
    sum_loss = 0.0
    steps = 0.0
    model.train()
    for i, x in enumerate(dataloader):
        line_imgs = Variable(x['line_imgs'].type(dtype), requires_grad=False)
        labels = Variable(x['labels'], requires_grad=False)
        label_lengths = Variable(x['label_lengths'], requires_grad=False)

        preds = model(line_imgs).cpu()
        preds_size = Variable(torch.IntTensor([preds.size(0)] * preds.size(1)))

        output_batch = preds.permute(1, 0, 2)
        out = output_batch.data.cpu().numpy()

        loss = criterion(preds, labels, preds_size, label_lengths)

        optimizer.zero_grad()
        loss.backward()
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


def make_dataloaders(data_config, augment_config, network_config, char_to_idx, warp):

    train_dataset = HwDataset(data_config['training_set_path'], char_to_idx,
                              img_height=network_config['input_height'],
                              data_root=data_config['image_root_directory'],
                              warp=warp,
                              augment_path=augment_config['training_set_path'] if augment_config else None,
                              augment_root=augment_config['image_root_directory'] if augment_config else None)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=0,
                                  collate_fn=hw_dataset.collate)

    test_dataset = HwDataset(data_config['validation_set_path'], char_to_idx, img_height=network_config['input_height'],
                             data_root=data_config['image_root_directory'],
                             augment_path=augment_config['validation_set_path'] if augment_config else None,
                             augment_root=augment_config['image_root_directory'] if augment_config else None)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0,
                                 collate_fn=hw_dataset.collate)

    return train_dataloader, test_dataloader


def main():
    config_path = sys.argv[1]

    with open(config_path) as f:
        config = json.load(f)

    results_dir = os.path.join('results', config['name'])
    if len(results_dir) > 0 and not os.path.exists(results_dir):
        os.makedirs(results_dir)

    global LOG_PATH
    LOG_PATH = os.path.join(results_dir, "log.txt")

    log_print("Using config file", config_path, new_start=True)
    log_print(json.dumps(config, indent=2))
    log_print()

    data_config = config['data']
    augment_config = config.get('augmentation')
    network_config = config['network']

    idx_to_char, char_to_idx = character_set.load_char_set(data_config['character_set_path'])
    if augment_config is not None:
        online_idx_to_char, online_char_to_idx = character_set.load_char_set(augment_config['character_set_path'])
        idx_to_char.update(online_idx_to_char)
        char_to_idx.update(online_char_to_idx)

    # train_dataset = HwDataset(data_config['training_set_path'], char_to_idx,
    #                           img_height=network_config['input_height'],
    #                           data_root=data_config['image_root_directory'],
    #                           warp=config['warp'],
    #                           augment_path=augment_config['training_set_path'] if augment_config else None,
    #                           augment_root=augment_config['image_root_directory'] if augment_config else None)
    # train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=0,
    #                               collate_fn=hw_dataset.collate)
    #
    # test_dataset = HwDataset(data_config['validation_set_path'], char_to_idx, img_height=network_config['input_height'],
    #                          data_root=data_config['image_root_directory'],
    #                          augment_path=augment_config['validation_set_path'] if augment_config else None,
    #                          augment_root=augment_config['image_root_directory'] if augment_config else None)
    # test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0,
    #                              collate_fn=hw_dataset.collate)

    train_dataloader, test_dataloader = make_dataloaders(data_config, augment_config, network_config, char_to_idx, config['warp'])

    n_train_instances = len(train_dataloader.dataset)
    log_print("Number of training instances:", n_train_instances)
    log_print("Number of test instances:", len(test_dataloader.dataset), '\n')

    hw = crnn.create_model({
        'cnn_out_size': config['network']['cnn_out_size'],
        'num_of_channels': 3,
        'num_of_outputs': len(idx_to_char)+1
    })

    if torch.cuda.is_available():
        hw.cuda()
        dtype = torch.cuda.FloatTensor
        log_print("Using GPU")
    else:
        dtype = torch.FloatTensor
        log_print("No GPU detected")

    optimizer = torch.optim.Adam(hw.parameters(), lr=config['network']['learning_rate'])
    criterion = CTCLoss()
    lowest_loss = float('inf')
    train_losses = []
    test_losses = []
    for epoch in range(1, 1001):
        log_print("Epoch ", epoch)
        # sum_loss = 0.0
        # steps = 0.0
        # hw.train()
        # for i, x in enumerate(train_dataloader):
        #     line_imgs = Variable(x['line_imgs'].type(dtype), requires_grad=False)
        #     labels = Variable(x['labels'], requires_grad=False)
        #     label_lengths = Variable(x['label_lengths'], requires_grad=False)
        #
        #     preds = hw(line_imgs).cpu()
        #     preds_size = Variable(torch.IntTensor([preds.size(0)] * preds.size(1)))
        #
        #     output_batch = preds.permute(1, 0, 2)
        #     out = output_batch.data.cpu().numpy()
        #
        #     loss = criterion(preds, labels, preds_size, label_lengths)
        #
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        #     #if i == 0:
        #     #    for i in xrange(out.shape[0]):
        #     #        pred, pred_raw = string_utils.naive_decode(out[i,...])
        #     #        pred_str = string_utils.label2str(pred_raw, idx_to_char, True)
        #     #        print(pred_str)
        #
        #     for j in range(out.shape[0]):
        #         logits = out[j,...]
        #         pred, raw_pred = string_utils.naive_decode(logits)
        #         pred_str = string_utils.label2str(pred, idx_to_char, False)
        #         gt_str = x['gt'][j]
        #         cer = error_rates.cer(gt_str, pred_str)
        #         sum_loss += cer
        #         steps += 1
        # training_cer = sum_loss / steps
        training_cer = run_epoch(hw, train_dataloader, criterion, optimizer, idx_to_char, dtype)
        log_print("Training CER", training_cer)
        train_losses.append(training_cer)

        # sum_loss = 0.0
        # steps = 0.0
        # hw.eval()
        # for x in test_dataloader:
        #     line_imgs = Variable(x['line_imgs'].type(dtype), requires_grad=False, volatile=True)
        #     labels =  Variable(x['labels'], requires_grad=False, volatile=True)
        #     label_lengths = Variable(x['label_lengths'], requires_grad=False, volatile=True)
        #
        #     preds = hw(line_imgs).cpu()
        #
        #     output_batch = preds.permute(1,0,2)
        #     out = output_batch.data.cpu().numpy()
        #
        #     for i, gt_line in enumerate(x['gt']):
        #         logits = out[i,...]
        #         pred, raw_pred = string_utils.naive_decode(logits)
        #         pred_str = string_utils.label2str(pred, idx_to_char, False)
        #         cer = error_rates.cer(gt_line, pred_str)
        #         sum_loss += cer
        #         steps += 1
        # test_cer = sum_loss / steps
        test_cer = test(hw, test_dataloader, idx_to_char, dtype)
        log_print("Test CER", test_cer)
        test_losses.append(test_cer)

        if lowest_loss > test_cer:
            lowest_loss = test_cer
            log_print("Saving Best")

            torch.save(hw.state_dict(), os.path.join(results_dir, config['name'] + "_model.pt"))

        results = {'train': train_losses, 'test': test_losses}
        with open(os.path.join(results_dir, "losses.json"), 'w') as fh:
            json.dump(results, fh, indent=4)

        x_axis = [(i+1) * n_train_instances for i in range(epoch)]
        plt.figure()
        plt.plot(x_axis, train_losses, label='train')
        plt.plot(x_axis, test_losses, label='test')
        plt.legend()
        plt.ylabel("CER")
        plt.xlabel("Number of Instances")
        plt.title("CER Loss")
        plt.savefig(os.path.join(results_dir, config['name'] + ".png"))
        plt.close()


if __name__ == "__main__":
    main()
