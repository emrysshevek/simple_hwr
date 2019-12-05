import torch
import robust_loss_pytorch
import numpy as np
import torch.nn as nn
from pydtw import dtw
from scipy import spatial
from robust_loss_pytorch import AdaptiveLossFunction
#from sdtw import SoftDTW
import torch.multiprocessing as multiprocessing
from hwr_utils.utils import to_numpy
from hwr_utils.stroke_recovery import relativefy
from hwr_utils.stroke_dataset import pad, create_gts
from hwr_utils.utils import print
from scipy.spatial import KDTree
import time

BCELoss = torch.nn.BCELoss()

class StrokeLoss:
    def __init__(self, loss_names="l1", parallel=False, vocab_size=4, loss_stats=None):
        super(StrokeLoss, self).__init__()
        ### Relative preds and relative GTs:
            # Resample GTs to be relative
            # Return unrelative GTs and Preds
            # This doesn't work that well, because it doesn't learn spaces
        ### Absolute everything
        ### Relative preds from network, then convert to absolute before loss
            # Return unrelative preds

        #device = torch.device("cuda")
        self.vocab_size = vocab_size
        self.cosine_similarity = nn.CosineSimilarity(dim=1)
        self.cosine_distance = lambda x, y: 1 - self.cosine_similarity(x, y)
        self.distributions = None
        self.parallel = parallel
        # self.unrelativefy_preds_for_loss = unrelativefy_preds_for_loss # if needed; i.e. calculate loss in absolute terms, but predict/return relative ones
        #
        # #
        # if relative_preds and unrelativefy_preds_for_loss:
        #     self.preprocess = self.unrelativefy
        # else:
        #     self.preprocess = lambda item, preds: preds

        self.poolcount = max(1, multiprocessing.cpu_count()-8)
        self.poolcount = 2
        self.truncate_preds = True
        self.set_loss(loss_names)
        self.stats = loss_stats

    def set_loss(self, loss_names_and_coef):
        loss_fns = []
        coefs = []
        loss_names = []
        if isinstance(loss_names_and_coef, str): # have a bunch of loss functions
            loss_names_and_coef = [loss_names_and_coef]

        for loss_name, coef in loss_names_and_coef:
            coefs.append(coef)
            loss_names.append(loss_name)

            if loss_name.lower() == "l1":
                loss_fn = self.l1
            elif loss_name.lower() == "variable_l1":
                loss_fn = self.variable_l1
            elif loss_name.lower() == "dtw":
                loss_fn = self.dtw
            elif loss_name.lower() == "barron":
                barron_loss_fn = AdaptiveLossFunction(num_dims=vocab_size, float_dtype=np.float32, device='cpu')
                loss_fn = barron_loss_fn.lossfun
            elif loss_name.lower() == "ssl":
                loss_fn = self.ssl
            else:
                raise Exception(f"Unknown loss: {loss_name}")
            loss_fns.append(loss_fn)

        self.coefs = coef
        self.loss_fns = loss_fns
        self.loss_names = loss_names


    @staticmethod
    def resample_gt(preds, targs):
        batch = targs
        device = preds.device
        targs = []
        label_lengths = []
        for i in range(0, preds.shape[0]):
            pred_length = preds[i].shape[0]
            t = create_gts(batch["x_func"][i], batch["y_func"][i], batch["start_times"][i],
                           number_of_samples=pred_length, noise=None,
                           relative_x_positions=batch["x_relative"])  # .transpose([1,0])
            t = torch.from_numpy(t.astype(np.float32)).to(device)
            targs.append(t)
            label_lengths.append(pred_length)
        return targs, label_lengths

    def main_loss(self, preds, targs, label_lengths, suffix="_train"):
        """ Preds: BATCH, TIME, VOCAB SIZE
                    VOCAB: x, y, start stroke, end_of_sequence
        Args:
            preds: Will be in the form [batch, width, alphabet]
            targs: Pass in the whole dictionary, so we can get lengths, etc., whatever we need

        Returns:

        # Adapatively invert stroke targs if first instance is on the wrong end?? sounds sloooow

        """
        # elif isinstance(targs, dict):
        #     label_lengths = targs["label_lengths"]
        #     targs = targs["gt_list"]
        # elif label_lengths is None:
        #     label_lengths = [t.shape[0] for t in targs]
        losses = torch.zeros(len(self.loss_fns))
        total_length = torch.sum(label_lengths.cpu(), 0, keepdim=False).item()
        ## Loop through loss functions
        for i, loss_fn in enumerate(self.loss_fns):
            loss_tensor = loss_fn(preds, targs, label_lengths)
            losses[i] = loss_tensor

            # Update loss stat
            loss = torch.sum(loss_tensor.cpu(), 0, keepdim=False).item() / total_length
            self.stats[self.loss_names[i] + suffix].accumulate(loss)

        combined_loss = torch.sum(losses)
        return combined_loss, loss

    def dtw(self, preds, targs, label_lengths):
        loss = 0
        if self.parallel:
            pool = multiprocessing.Pool(processes=self.poolcount)
            as_and_bs = pool.imap(self.dtw_single, iter(zip(
                to_numpy(preds),
                targs)), chunksize=32)  # iterates through everything all at once
            pool.close()
            #print(as_and_bs[0])
            for i, (a,b) in enumerate(as_and_bs): # loop through BATCH
                loss += abs(preds[i, a, :2] - targs[i][b, :2]).sum()
        else:
            for i in range(len(preds)): # loop through BATCH
                a,b = self.dtw_single((preds[i], targs[i]))
                loss += abs(preds[i][a, :2] - targs[i][b, :2]).sum() # AVERAGE pointwise loss for 1 image
        return loss

    @staticmethod
    def dtw_single(_input):
        """
        Args:
            _input (tuple): targ, pred, label_length

        Returns:

        """
        pred, targ = _input
        pred, targ = to_numpy(pred, astype="float64"), to_numpy(targ, astype="float64")
        x1 = np.ascontiguousarray(pred[:, :2])  # time step, batch, (x,y)
        x2 = np.ascontiguousarray(targ[:, :2])
        dist, cost, a, b = dtw.dtw2d(x1, x2)

        # Cost is weighted by how many GT stroke points, i.e. how long it is
        return a,b

    def barron_loss(self, preds, targs, label_lengths):
        # BATCH, TIME, VOCAB
        vocab_size = preds.shape[-1]
        _preds = preds.reshape(-1, vocab_size)
        _targs = targs.reshape(-1, vocab_size)
        return torch.sum(self.barron_loss_fn((_preds - _targs)))

    @staticmethod
    def variable_l1(preds, targs, label_lengths=None):
        """ Resmaple the targets to match whatever was predicted

        Args:
            preds:
            targs:
            label_lengths:

        Returns:

        """
        targs, label_lengths = StrokeLoss.resample_gt(preds, targs)
        return StrokeLoss.l1(preds, targs, label_lengths)

    @staticmethod
    def l1(preds, targs, label_lengths):
        loss = 0
        for i, pred in enumerate(preds):
          loss += torch.sum(abs(pred[:, :2]-targs[i][:, :2]))
        return loss

    @staticmethod
    def ssl(preds, targs, label_lengths):
        ### TODO: each loss function does the appropriate stat vs gradient scaling

        # Method
            ## Find the point nearest to the actual start stroke
            ## Assume this point should have been the predicted start stroke
            ## Calculate loss for predicting the start strokes!

        # OTHER LOSS
            # Sometimes the model "skips" stroke points because of DTW
            # Calculate the nearest point to every start and end point
            # Have this be an additional loss


        # start_time = time.time()
        # for each of the start strokes in targs
        loss = 0

        for i in range(len(preds)):
            targ_start_strokes = targs[i][torch.nonzero(targs[i][:, 2]).squeeze(1), :2]
            targ_end_strokes = (targ_start_strokes-1)[1:] # get corresponding end strokes - this excludes the final stroke point!!
            k = KDTree(preds[i][:, :2].data)

            # Start indices
            start_indices = k.query(targ_start_strokes)[1]
            pred_start_fitted = torch.zeros(preds[i].shape[0])
            pred_start_fitted[start_indices] = 1

            # End indices
            end_indices = k.query(targ_end_strokes)[1]
            # pred_end_fitted = torch.zeros(preds[i].shape[0])
            # pred_end_fitted[end_indices] = 1

            # Do L1 distance loss for start strokes and nearest stroke point
            loss += abs(preds[i][start_indices, :2] - targ_start_strokes).sum()
            loss += abs(preds[i][end_indices, :2] - targ_end_strokes).sum()

            # Do SOStr classification loss
            loss += BCELoss(pred_start_fitted, targs[i][:, 2])

            # Do EOSeq prediction
            loss += BCELoss(preds[i][:,3], targs[i][:, 3])

            # # Do L1 distance loss
            # loss += abs(preds[i][start_indices, :2] - targ_start_strokes).sum() / len(start_indices)
            # loss += abs(preds[i][end_indices, :2] - targ_end_strokes).sum() / len(end_indices)
            # loss += 0.1 * abs(pred_start_fitted - targs[i][:, 2]).sum()
            # loss += 0.1 * abs(pred_end_fitted - targs[i][:, 3]).sum()

        #print("Time to compute ssl: ", time.time() - start_time)
        return loss
  

    def calculate_nn_distance(self, item, preds):
        """ Can this be done differentiably?

        Args:
            item:
            preds:

        Returns:

        """
        # calculate NN distance
        n_pts = 0
        cum_dist = 0
        gt = item["gt_list"]
        batch_size = len(gt)
        for i in range(batch_size):
            # TODO binarize line images and do dist based on that
            kd = KDTree(preds[i][:, :2].data)
         #   cum_dist = sum(kd.query(gt[i][:, :2])[0])
            n_pts += gt[i].shape[0]

        return (cum_dist / n_pts) * batch_size # THIS WILL BE DIVIDED BY THE NUMBER OF INSTANCES LATER
        #print("cum_dist: ", cum_dist, "n_pts: ", n_pts)
        #print("Distance: ", cum_dist / n_pts)

if __name__ == "__main__":
    from models.basic import CNN, BidirectionalRNN
    from torch import nn

    vocab_size = 4
    batch = 3
    time = 16
    y = torch.rand(batch, 1, 60, 60)
    targs = torch.rand(batch, time, vocab_size)  # BATCH, TIME, VOCAB
    cnn = CNN(nc=1)
    rnn = BidirectionalRNN(nIn=1024, nHidden=128, nOut=vocab_size, dropout=.5, num_layers=2, rnn_constructor=nn.LSTM)
    cnn_output = cnn(y)
    rnn_output = rnn(cnn_output).permute(1, 0, 2)
    print(rnn_output.shape)  # BATCH, TIME, VOCAB
    loss = StrokeLoss()
    loss = loss.main_loss(None, rnn_output, targs)
    print(loss)

