import numpy as np
import torch
# from sdtw import SoftDTW
import torch.multiprocessing as multiprocessing
import torch.nn as nn
from pydtw import dtw
from scipy.spatial import KDTree
from torch import Tensor
from robust_loss_pytorch import AdaptiveLossFunction
import logging
from hwr_utils.stroke_dataset import create_gts
from hwr_utils.utils import to_numpy
from hwr_utils.stroke_recovery import relativefy_torch

BCELoss = torch.nn.BCELoss()
BCEWithLogitsLoss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.ones(1)*5)
SIGMOID = torch.nn.Sigmoid()
# DEVICE???
# x.requires_grad = False

logger = logging.getLogger("root."+__name__)


class CustomLoss(nn.Module):
    def __init__(self, loss_indices, device="cuda", **kwargs):
        super().__init__()
        self.loss_indices = loss_indices
        self.device = "cpu"  # I guess this needs to be CPU? IDK
        self.__dict__.update(**kwargs)
        SIGMOID.to(device)
        if "subcoef" in kwargs:
            subcoef = kwargs["subcoef"]
            if isinstance(subcoef, str):
                subcoef = [float(s) for s in subcoef.split(",")]
            self.subcoef = Tensor(subcoef).to(self.device)
        else:
            # MAY NOT ALWAYS BE 4!!!
            length = len(range(*loss_indices.indices(4))) if isinstance(loss_indices, slice) else len(loss_indices)
            self.subcoef = torch.ones(length).to(self.device)

class DTWLoss(CustomLoss):
    def __init__(self, loss_indices, dtw_mapping_basis=None, **kwargs):
        """

        Args:
            dtw_mapping_basis: The slice on which to match points (e.g. X,Y)
            dtw_loss_slice: The values for which to calculate the loss
                (e.g. after matching on X,Y, calculate the loss based on X,Y,SOS)

        """
        # parse the opts - this will include opts regarding the DTW basis
        # loss_indices - the loss_indices to calculate the actual loss
        super().__init__(loss_indices, **kwargs)
        self.dtw_mapping_basis = loss_indices if dtw_mapping_basis is None else dtw_mapping_basis
        self.lossfun = self.dtw
        self.abs = abs
        self.method = "normal" if not "method" in kwargs else kwargs["method"]


        if "cross_entropy_indices" in kwargs and kwargs["cross_entropy_indices"]:
            self.cross_entropy_indices = kwargs["cross_entropy_indices"]
        else:
            self.cross_entropy_indices = None

        if "barron" in kwargs and kwargs["barron"]:
            logger.info("USING BARRON + DTW!!!")
            self.barron = AdaptiveLossFunction(num_dims=len(loss_indices), float_dtype=np.float32, device='cpu').lossfun
        else:
            self.barron = None

        if "relativefy_cross_entropy_gt" in kwargs and kwargs["relativefy_cross_entropy_gt"]:
            logger.info("Relativefying stroke number + BCE!!!")
            self.relativefy = True
        else:
            self.barron = False

    # not faster
    def parallel_dtw(self, preds, targs, label_lengths, **kwargs):
        loss = 0
        if self.parallel:
            pool = multiprocessing.Pool(processes=self.poolcount)
            as_and_bs = pool.imap(self.dtw_single, iter(zip(
                to_numpy(preds),
                targs)), chunksize=32)  # iterates through everything all at once
            pool.close()
            # print(as_and_bs[0])
            for i, (a, b) in enumerate(as_and_bs):  # loop through BATCH
                loss += abs(preds[i, a, :2] - targs[i][b, :2]).sum()
        return loss

    def dtw(self, preds, targs, label_lengths, **kwargs):
        loss = 0
        for i in range(len(preds)):  # loop through BATCH
            a, b = self.dtw_single((preds[i], targs[i]))

            # LEN X VOCAB
            if self.method=="normal":
                pred = preds[i][a, :][:, self.loss_indices]
                targ = targs[i][b, :][:, self.loss_indices]
            # elif self.method=="align_to_gt":
            #     pred = preds[i][a, :][:, self.loss_indices]
            #     targ = targs[i][:, self.loss_indices]
            # elif self.method=="align_to_pred":
            #     pred = preds[i][:, self.loss_indices]
            #     targ = targs[i][b, :][:, self.loss_indices]
            # elif self.method=="both":
            #     pred = torch.cat(preds[i][:, self.loss_indices], preds[i][a, :][:, self.loss_indices])
            #     targ = torch.cat(targs[i][b, :][:, self.loss_indices], targs[i][:, self.loss_indices])
            else:
                raise NotImplemented

            ## !!! DELETE THIS
            if self.barron:
                loss += (self.barron(pred - targ) * self.subcoef).sum()  # AVERAGE pointwise loss for 1 image
            elif True:
                loss += (abs(pred - targ) * self.subcoef).sum()  # AVERAGE pointwise loss for 1 image
            else:
                # ONLY WHEN USING SOS!!!
                start_strokes_factor = (targs[i][b, 2] * 4 + 1).unsqueeze(1).repeat(1, len(self.loss_indices))
                loss += (start_strokes_factor * abs(pred - targ) * self.subcoef).sum()  # AVERAGE pointwise loss for 1 image

            if self.cross_entropy_indices:
                pred = preds[i][a, :][:, self.cross_entropy_indices]
                targ = targs[i][b, :][:, self.cross_entropy_indices]

                if self.relativefy:
                    targ = relativefy_torch(targ)

                loss += BCEWithLogitsLoss(pred, targ).sum() * .1  # AVERAGE pointwise loss for 1 image

        return loss  # , to_value(loss)

    def dtw_single(self, _input):
        """ THIS DOES NOT USE SUBCOEF
        Args:
            _input (tuple): pred, targ, label_length

        Returns:

        """
        pred, targ = _input
        pred, targ = to_numpy(pred[:, self.dtw_mapping_basis], astype="float64"), \
                     to_numpy(targ[:, self.dtw_mapping_basis], astype="float64")
        dist, cost, a, b = self._dtw(pred, targ)

        # Cost is weighted by how many GT stroke points, i.e. how long it is
        return a, b

    @staticmethod
    # ORIGINAL
    def _dtw(pred, targ):
        # Cost is weighted by how many GT stroke points, i.e. how long it is
        x1 = np.ascontiguousarray(pred)  # time step, batch, (x,y)
        x2 = np.ascontiguousarray(targ)
        return dtw.dtw2d(x1, x2) # dist, cost, a, b

    # @staticmethod
    # # FASTER
    # def _dtw(pred, targ):
    #     # Cost is weighted by how many GT stroke points, i.e. how long it is
    #     x1 = np.ascontiguousarray(pred)  # time step, batch, (x,y)
    #     x2 = np.ascontiguousarray(targ)
    #     return dtw.dtw2d(x1, x2) # dist, cost, a, b
    #
    # import dtaidistance.dtw
    # from dtaidistance.dtw_ndim

class L1(CustomLoss):
    """ Use opts to specify "variable_L1" (resample to get the same number of GTs/preds)
    """

    def __init__(self, loss_indices, **kwargs):
        """
        """
        # parse the opts - this will include opts regarding the DTW basis
        # loss_indices - the loss_indices to calculate the actual loss
        super().__init__(loss_indices, **kwargs)
        self.lossfun = self.l1

    @staticmethod
    # BATCH x LEN x VOCAB
    def l1_swapper(preds, targs, label_lengths, **kwargs):
        loss = 0
        for i, pred in enumerate(preds):
            targ = targs[i].transpose(1,0) # swap width and vocab -> VOCAB x WIDTH
            pred = pred.transpose(1,0)
            diff  = torch.sum(torch.abs(pred.reshape(-1, 2) - targ.reshape(-1, 2)), axis=1)
            diff2 = torch.sum(torch.abs(pred.reshape(-1, 2) - torch.flip(targ.reshape(-1, 2), dims=(1,))), axis=1)
            loss += torch.sum(torch.min(diff, diff2)) # does not support subcoef
        return loss  # , to_value(loss)

    @staticmethod
    def variable_l1(preds, targs, label_lengths, **kwargs):
        """ Resmaple the targets to match whatever was predicted, i.e. so they have the same number (targs/preds)

        Args:
            preds:
            targs:
            label_lengths:

        Returns:

        """
        targs, label_lengths = resample_gt(preds, targs)
        loss = L1.loss(preds, targs, label_lengths)  # already takes average loss
        return loss  # , to_value(loss)

    def l1(self, preds, targs, label_lengths, **kwargs):
        loss = 0
        for i, pred in enumerate(preds):
            loss += torch.sum(abs(pred[:, self.loss_indices] - targs[i][:, self.loss_indices]) * self.subcoef)
        return loss  # , to_value(loss)

class L2(CustomLoss):
    """ Use opts to specify "variable_L1" (resample to get the same number of GTs/preds)
    """

    def __init__(self, loss_indices, **kwargs):
        """
        """
        # parse the opts - this will include opts regarding the DTW basis
        # loss_indices - the loss_indices to calculate the actual loss
        super().__init__(loss_indices, **kwargs)
        self.lossfun = self.l2

    @staticmethod
    def variable_l2(preds, targs, label_lengths, **kwargs):
        """ Resample the targets to match whatever was predicted, i.e. so they have the same number (targs/preds)

        Args:
            preds:
            targs:
            label_lengths:

        Returns:

        """
        targs, label_lengths = resample_gt(preds, targs)
        loss = L2.loss(preds, targs, label_lengths)  # already takes average loss
        return loss  # , to_value(loss)

    def l2(self, preds, targs, label_lengths, **kwargs):
        loss = 0
        for i, pred in enumerate(preds):  # loop through batch
            loss += torch.sum((pred[:, self.loss_indices] - targs[i][:, self.loss_indices]) ** 2 * self.subcoef) ** (
                        1 / 2)
        return loss  # , to_value(loss)


class CrossEntropy(nn.Module):
    """ Use opts to specify "variable_L1" (resample to get the same number of GTs/preds
    """

    def __init__(self, loss_indices, **kwargs):
        """
        """
        # parse the opts - this will include opts regarding the DTW basis
        # loss_indices - the loss_indices to calculate the actual loss
        super().__init__()
        self.__dict__.update(kwargs)
        self.loss_indices = loss_indices
        self.lossfun = self.cross_entropy

        self._loss = BCELoss
        if "activation" in kwargs.keys():
            if kwargs["activation"] == "sigmoid":
                self._loss = BCEWithLogitsLoss
                # torch.nn.Sigmoid().to(device)


    def cross_entropy(self, preds, targs, label_lengths, **kwargs):
        loss = 0
        for i, pred in enumerate(preds):  # loop through batches, since they are not the same size
            targ = targs[i]
            loss += self._loss(pred[:, self.loss_indices], targ[:, self.loss_indices])
        return loss  # , to_value(loss)


class SSL(nn.Module):
    def __init__(self, loss_indices, **kwargs):
        """
        """
        # parse the opts - this will include opts regarding the DTW basis
        # loss_indices - the loss_indices to calculate the actual loss
        super().__init__()
        self.loss_indices = 2
        self.nn_indices = slice(0, 2)
        self.lossfun = self.ssl

    def ssl(self, preds, targs, label_lengths):
        ### TODO: L1 distance and SOS/EOS are really two different losses, but they both depend on identifying the start points

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
        loss_tensor = 0

        # Preds are BATCH x LEN x VOCAB
        for i in range(len(preds)):  # loop through batches
            # Get the coords of all start strokes
            targ_start_strokes = targs[i][torch.nonzero(targs[i][:, self.loss_indices]).squeeze(1), self.nn_indices]  #
            # targ_end_strokes = (targ_start_strokes-1)[1:] # get corresponding end strokes - this excludes the final stroke point!!
            k = KDTree(preds[i][:, self.nn_indices].data)

            # Start loss_indices; get the preds nearest the actual start points
            start_indices = k.query(targ_start_strokes)[1]
            pred_gt_fitted = torch.zeros(preds[i].shape[0])
            pred_gt_fitted[start_indices] = 1

            # End loss_indices
            # end_indices = k.query(targ_end_strokes)[1]
            # pred_end_fitted = torch.zeros(preds[i].shape[0])
            # pred_end_fitted[end_indices] = 1

            # Do L1 distance loss for start strokes and nearest stroke point
            # loss_tensor += abs(preds[i][start_indices, :2] - targ_start_strokes).sum()
            # loss_tensor += abs(preds[i][end_indices, :2] - targ_end_strokes).sum()

            # Do SOStr classification loss
            # print("preds", pred_gt_fitted)
            loss_tensor += BCELoss(preds[i][:, self.loss_indices], pred_gt_fitted)

            # print(targ_start_strokes, start_indices)
            # input()
            # print(pred_gt_fitted)
            # input()
            # print(targs[i][:,2])
            # print(loss_tensor)
            # input()

            # Do EOSeq prediction - not totally fair, again, we should evaluate it based on the nearest point to the last prediction
            # loss_tensor += BCELoss(preds[i][:, 3], targs[i][:, 3])

            # # Do L1 distance loss
            # loss += abs(preds[i][start_indices, :2] - targ_start_strokes).sum() / len(start_indices)
            # loss += abs(preds[i][end_indices, :2] - targ_end_strokes).sum() / len(end_indices)
            # loss += 0.1 * abs(pred_gt_fitted - targs[i][:, 2]).sum()
            # loss += 0.1 * abs(pred_end_fitted - targs[i][:, 3]).sum()
        loss = to_value(loss_tensor)
        # print("Time to compute ssl: ", time.time() - start_time)
        return loss_tensor  # , loss


def resample_gt(preds, targs, gt_format):
    batch = targs
    device = preds.device
    targs = []
    label_lengths = []
    for i in range(0, preds.shape[0]):
        pred_length = preds[i].shape[0]
        t = create_gts(batch["x_func"][i], batch["y_func"][i], batch["start_times"][i],
                       number_of_samples=pred_length, noise=None,
                       gt_format=gt_format)  # .transpose([1,0])
        t = torch.from_numpy(t.astype(np.float32)).to(device)
        targs.append(t)
        label_lengths.append(pred_length)
    return targs, label_lengths


def to_value(loss_tensor):
    return torch.sum(loss_tensor.cpu(), 0, keepdim=False).item()


def tensor_sum(tensor):
    return torch.sum(tensor.cpu(), 0, keepdim=False).item()

