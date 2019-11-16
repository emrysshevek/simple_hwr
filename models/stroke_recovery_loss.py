import torch
import robust_loss_pytorch
import numpy as np
import torch.nn as nn
from pydtw import dtw
from scipy import spatial
from robust_loss_pytorch import AdaptiveLossFunction
from sdtw import SoftDTW
import torch.multiprocessing as multiprocessing
from hwr_utils.utils import to_numpy
from hwr_utils.stroke_recovery import relativefy
from hwr_utils.stroke_dataset import pad, create_gts

class StrokeLoss:
    def __init__(self, parallel=False, vocab_size=4):
        super(StrokeLoss, self).__init__()
        #device = torch.device("cuda")
        self.vocab_size = vocab_size
        self.barron_loss_fn = AdaptiveLossFunction(num_dims=vocab_size, float_dtype=np.float32, device='cpu').lossfun
        self.cosine_similarity = nn.CosineSimilarity(dim=1)
        self.cosine_distance = lambda x, y: 1 - self.cosine_similarity(x, y)
        self.distributions = None
        self.parallel = parallel

        self.poolcount = max(1, multiprocessing.cpu_count()-8)
        self.poolcount = 2
        self.truncate_preds = True

    def main_loss(self, loss_fn, preds, targs, label_lengths):
        """ Preds: BATCH, TIME, VOCAB SIZE
                    VOCAB: x, y, start stroke, end_of_sequence
        Args:
            preds: Will be in the form [batch, width, alphabet]
            targs: Pass in the whole dictionary, so we can get lengths, etc., whatever we need

        Returns:

        # Adapatively invert stroke targs if first instance is on the wrong end?? sounds sloooow

        """
        if loss_fn is None or True: #TEMP
            loss_fn = self.l1 # StrokeLoss.dtw

        ## RESAMPLE THE GTs to match
        if loss_fn==self.variable_l1:
            loss_fn = self.l1
            batch = targs
            device = preds.device
            targs = []
            label_lengths = []
            for i in range(0, preds.shape[0]):
                pred_length = preds[i].shape[0]
                t = create_gts(batch["x_func"][i], batch["y_func"][i], batch["start_times"][i],
                                  number_of_samples=pred_length, noise=None, relative_x_positions=batch["x_relative"]) #.transpose([1,0])
                t = torch.from_numpy(t.astype(np.float32)).to(device)
                targs.append(t)
                label_lengths.append(pred_length)
        elif isinstance(targs, dict):
            label_lengths = targs["label_lengths"]
            targs = targs["gt_list"]
        elif label_lengths is None:
            label_lengths = [t.shape[0] for t in targs]

        loss = loss_fn(preds, targs, label_lengths)
        return loss

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
                loss += abs(preds[i, a, :] - targs[i][b, :]).sum() / label_lengths[i]
        else:
            for i in range(len(preds)): # loop through BATCH
                a,b = self.dtw_single((preds[i], targs[i]))
                loss += abs(preds[i, a, :] - targs[i][b, :]).sum() / label_lengths[i]
        return loss

    def truncate_preds_func(func):
        """ The images have a bunch of padded space. We don't care about what is predicted after a certain point.
            So if we're using preds that have been resampled based on image width, truncate the prediction to match

        Returns:

        """

        # REMOVE PRINT OUTS
        # DON'T RESAMPLE? HM...
        # Check parallel
        def wrapper(*args, **kwargs):
            if False: # FIX THIS - only if self.truncate
                # BATCH, TIME, VOCAB
                kwargs["preds"] = preds = args[0] if len(args)>0 else kwargs["preds"]
                kwargs["targs"] = targs = args[1] if len(args)>1 else kwargs["targs"]
                kwargs["label_lengths"] = label_lengths = args[2] if len(args) > 2 else kwargs["label_lengths"]
                kwargs["preds"]= [preds[i][:label_lengths[i], :] for i in range(0,len(label_lengths))]
                kwargs["targs"]= [targs[i][:label_lengths[i], :] for i in range(0,len(label_lengths))]
            return func(*args, **kwargs)
        return wrapper

    @staticmethod
    @truncate_preds_func
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
        return torch.sum(self.barron_loss_fn((_preds - _targs)))/np.sum(label_lengths)

    @staticmethod
    @truncate_preds_func
    def variable_l1(preds, batch, label_lengths):
        pass
    #     loss = 0
    #     print(batch)
    #     print(preds)
    #     Stop
    #     for i, pred in enumerate(preds):
    #         pred_length = preds.shape[1]
    #         targ = create_gts(batch["x_func"][i], batch["y_func"][i], batch["start_times"][i], number_of_samples=pred_length, noise=None, relative_x_positions=batch["x_relative"])
    #         loss += torch.sum(abs(pred-targ))/label_lengths[i]
    #     return loss

    @staticmethod
    @truncate_preds_func
    def l1(preds, targs, label_lengths):
        loss = 0
        for i, pred in enumerate(preds):
            loss += torch.sum(abs(pred-targs[i]))/label_lengths[i]
        return loss

'''
    OLD attempts at loss
    def angle_loss(self, preds, targs):
        #end_strokes = (targs[:, :, 3] != 1)[:, :-1]
        pred_diffs = preds[:, 1:, :2] - preds[:, :-1, :2]
        targ_diffs = targs[:, 1:, :2] - targs[:, :-1, :2]


        # set diffs to 0 at end_stroke indices
        #pred_diffs[:, :, 0] *= end_strokes
        #pred_diffs[:, :, 1] *= end_strokes
        #targ_diffs[:, :, 0] *= end_strokes
        #targ_diffs[:, :, 1] *= end_strokes

        #print(pred_diffs.shape, targ_diffs.shape)


        # get the indices of the slopes that shouldn't be connected and remove that slope
        loss = self.cosine_distance(pred_diffs, targ_diffs)
        return loss.sum() * 25 # constant to make it start out roughly equal to the location loss

    def log_prob_loss(self, preds, targs):
        pred_diffs = preds[:, 1:, :2] - preds[:, :-1, :2]
        targ_diffs = targs[:, 1:, :2] - targs[:, :-1, :2]
        targ_slopes = targ_diffs[:, :, 0] / targ_diffs[:, :, 1]
        loss = 0
        for i in range(len(targ_diffs)):
            for j in range(len(targ_diffs[i])):
                # Get variances according to diffs
                bivariate = torch.distributions.multivariate_normal.MultivariateNormal(targs[i][j][:2], torch.eye(2))
                loss_part = bivariate.log_prob(preds[i][j][:2])
                loss += loss_part
        return -loss
    '''

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

