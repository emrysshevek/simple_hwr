import json

import numpy as np

import torch
from torch import nn
from torch.autograd import Variable

from utils import string_utils
from utils.utils import calculate_cer


class TrainerSeq2Seq(json.JSONEncoder):
    def __init__(self, model, optimizer, config, criterion):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.criterion = criterion

        self.idx_to_char = config["idx_to_char"]
        self.teach_force_rate = config['teacher_force_rate']
        self.teacher_force_decay = config['teacher_force_decay']
        self.teacher_step_size = config['teacher_step_size']

    def default(self, o):
        return None

    def format_sequences(self, pred, label):
        pred_len = pred.shape[-1]

    def stringify(self, pred_sequences):
        sos, eos, pad = self.config['sos_idx'], self.config['eos_idx'], self.config['pad_idx']
        pred_sequences = pred_sequences.argmax(dim=-1).detach().cpu().numpy()
        cleaned_seqs = []
        for seq in pred_sequences:
            seq = seq.flatten()
            eos_pos = np.argwhere(seq == eos)
            eos_pos = None if len(eos_pos) == 0 else eos_pos.flatten()[0]
            cleaned_seq = seq[:eos_pos]
            cleaned_seq = cleaned_seq[(cleaned_seq != sos) & (cleaned_seq != pad)]
            cleaned_seqs.append(cleaned_seq)
        pred_strs = [string_utils.label2str(seq, self.config['idx_to_char']) for seq in cleaned_seqs]
        return pred_strs

    def train(self, line_imgs, online, labels, label_lengths, gts, retain_graph=False, step=0):
        self.model.train()

        teacher_force_rate = self.teach_force_rate * (self.teacher_force_decay ** (step // self.teacher_step_size))
        print(teacher_force_rate)

        text_sequence = self.model(line_imgs, online, labels, teacher_force_rate).cpu()
        batch_size, seq_len, vocab_size = text_sequence.shape
        pred_strs = self.stringify(text_sequence)

        assert np.array_equal(text_sequence.shape, (*labels.shape, len(self.idx_to_char)))

        self.config["logger"].debug("Calculating Loss: {}".format(step))
        loss = self.criterion(text_sequence.view(-1, vocab_size), labels.view(-1))

        self.optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        self.optimizer.step()

        loss = loss.item()
        # Error Rate
        self.config["stats"]["HWR Training Loss"].accumulate(loss, 1)  # Might need to be divided by batch size?
        err, weight = calculate_cer(pred_strs, gts)
        self.config["stats"]["Training Error Rate"].accumulate(err, weight)

        return loss, err, pred_strs

    def test(self, line_imgs, online, gt, force_training=False, nudger=False):
        """

        Args:
            line_imgs:
            online:
            gt:
            force_training: Run test in .train() as opposed to .eval() mode
            update_stats:

        Returns:

        """

        if force_training:
            self.model.train()
        else:
            self.model.eval()

        pred_seqs = self.model(line_imgs, online)
        pred_strs = self.stringify(pred_seqs)

        # Error Rate
        err, weight = calculate_cer(pred_strs, gt)
        self.config["stats"]["Test Error Rate"].accumulate(err, weight)
        loss = -1  # not calculating test loss here

        return loss, err, pred_strs


class TrainerBaseline(json.JSONEncoder):
    def __init__(self, model, optimizer, config, ctc_criterion):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.ctc_criterion = ctc_criterion
        self.idx_to_char = self.config["idx_to_char"]
        self.train_decoder = string_utils.naive_decode
        self.decoder = config["decoder"]

    def default(self, o):
        return None

    def train(self, line_imgs, online, labels, label_lengths, gt, retain_graph=False, step=0):
        self.model.train()

        pred_tup = self.model(line_imgs, online)
        pred_text, rnn_input, *_ = pred_tup[0].cpu(), pred_tup[1], pred_tup[2:]

        # Calculate HWR loss
        preds_size = Variable(torch.IntTensor([pred_text.size(0)] * pred_text.size(1)))

        output_batch = pred_text.permute(1, 0, 2) # Width,Batch,Vocab -> Batch, Width, Vocab
        pred_strs = self.decoder.decode_training(output_batch)

        # Get losses
        self.config["logger"].debug("Calculating CTC Loss: {}".format(step))
        loss_recognizer = self.ctc_criterion(pred_text, labels, preds_size, label_lengths)

        # Backprop
        self.optimizer.zero_grad()
        loss_recognizer.backward(retain_graph=retain_graph)
        self.optimizer.step()

        loss = torch.mean(loss_recognizer.cpu(), 0, keepdim=False).item()

        # Error Rate
        self.config["stats"]["HWR Training Loss"].accumulate(loss, 1)  # Might need to be divided by batch size?
        err, weight = calculate_cer(pred_strs, gt)
        self.config["stats"]["Training Error Rate"].accumulate(err, weight)

        return loss, err, pred_strs


    def test(self, line_imgs, online, gt, force_training=False, nudger=False):
        """

        Args:
            line_imgs:
            online:
            gt:
            force_training: Run test in .train() as opposed to .eval() mode
            update_stats:

        Returns:

        """

        if force_training:
            self.model.train()
        else:
            self.model.eval()

        pred_tup = self.model(line_imgs, online)
        pred_text, rnn_input, *_ = pred_tup[0].cpu(), pred_tup[1], pred_tup[2:]

        output_batch = pred_text.permute(1, 0, 2)
        pred_strs = self.decoder.decode_test(output_batch)

        # Error Rate
        if nudger:
            return rnn_input
        else:
            err, weight = calculate_cer(pred_strs, gt)
            self.config["stats"]["Test Error Rate"].accumulate(err, weight)
            loss = -1 # not calculating test loss here
            return loss, err, pred_strs


class TrainerNudger(json.JSONEncoder):
    def __init__(self, model, optimizer, config, ctc_criterion, train_baseline=True):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.ctc_criterion = ctc_criterion
        self.idx_to_char = self.config["idx_to_char"]
        self.baseline_trainer = TrainerBaseline(model, config["optimizer"], config, ctc_criterion)
        self.nudger = config["nudger"]
        self.recognizer_rnn = self.model.rnn
        self.train_baseline = train_baseline
        self.decoder = config["decoder"]

    def default(self, o):
        return None

    def train(self, line_imgs, online, labels, label_lengths, gt, retain_graph=False, step=0):
        self.nudger.train()

        # Train baseline at the same time
        if self.train_baseline:
            baseline_loss, baseline_prediction, rnn_input = self.baseline_trainer.train(line_imgs, online, labels, label_lengths, gt, retain_graph=True)
            self.model.my_eval()
        else:
            baseline_prediction, rnn_input = self.baseline_trainer.test(line_imgs, online, gt, force_training=True, update_stats=False)

        pred_text_nudged, nudged_rnn_input, *_ = [x.cpu() for x in self.nudger(rnn_input, self.recognizer_rnn) if not x is None]
        preds_size = Variable(torch.IntTensor([pred_text_nudged.size(0)] * pred_text_nudged.size(1)))
        output_batch = pred_text_nudged.permute(1, 0, 2)
        pred_strs = self.decoder.decode_training(output_batch)

        self.config["logger"].debug("Calculating CTC Loss (nudged): {}".format(step))
        loss_recognizer_nudged = self.ctc_criterion(pred_text_nudged, labels, preds_size, label_lengths)
        loss = torch.mean(loss_recognizer_nudged.cpu(), 0, keepdim=False).item()

        # Backprop
        self.optimizer.zero_grad()
        loss_recognizer_nudged.backward()
        self.optimizer.step()

        ## ASSERT SOMETHING HAS CHANGED

        if self.train_baseline:
            self.model.my_train()

        # Error Rate
        self.config["stats"]["Nudged Training Loss"].accumulate(loss, 1)  # Might need to be divided by batch size?
        err, weight, pred_str = calculate_cer(pred_strs, gt)
        self.config["stats"]["Nudged Training Error Rate"].accumulate(err, weight)

        return loss, err, pred_str

    def test(self, line_imgs, online, gt):
        self.nudger.eval()
        rnn_input = self.baseline_trainer.test(line_imgs, online, gt, nudger=True)

        pred_text_nudged, nudged_rnn_input, *_ = [x.cpu() for x in self.nudger(rnn_input, self.recognizer_rnn) if not x is None]
        # preds_size = Variable(torch.IntTensor([pred_text_nudged.size(0)] * pred_text_nudged.size(1)))
        output_batch = pred_text_nudged.permute(1, 0, 2)
        pred_strs = self.decoder.decode_test(output_batch)
        err, weight = calculate_cer(pred_strs, gt)

        self.config["stats"]["Nudged Test Error Rate"].accumulate(err, weight)
        loss = -1

        return loss, err, pred_strs