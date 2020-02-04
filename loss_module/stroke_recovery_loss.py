import torch
import robust_loss_pytorch
import numpy as np
import torch.nn as nn
from torch import Tensor
from pydtw import dtw
from scipy import spatial
from robust_loss_pytorch import AdaptiveLossFunction
#from sdtw import SoftDTW
import torch.multiprocessing as multiprocessing
from hwr_utils.utils import to_numpy, Counter
from hwr_utils.stroke_recovery import relativefy
from hwr_utils.stroke_dataset import pad, create_gts
from scipy.spatial import KDTree
import time
from loss_module.losses import *

class StrokeLoss:
    def __init__(self, parallel=False, vocab_size=4, loss_stats=None, counter=None, device="cuda", **kwargs):
        super(StrokeLoss, self).__init__()
        ### Relative preds and relative GTs:
            # Resample GTs to be relative
            # Return unrelative GTs and Preds
            # This doesn't work that well, because it doesn't learn spaces
        ### Absolute everything
        ### Relative preds from network, then convert to absolute before loss
            # Return unrelative preds

        self.device = device
        self.vocab_size = vocab_size
        self.cosine_similarity = nn.CosineSimilarity(dim=1)
        self.cosine_distance = lambda x, y: 1 - self.cosine_similarity(x, y)
        self.distributions = None
        self.parallel = parallel
        self.counter = Counter() if counter is None else counter
        self.poolcount = max(1, multiprocessing.cpu_count()-8)
        self.poolcount = 2
        self.truncate_preds = True
        self.dtw = None
        self.stats = loss_stats
        self.loss_names = []
        self.coefs = []
        self.loss_fns = []
        self.loss_names_and_coef = None


    def build_losses(self, loss_fn_definition):
        if loss_fn_definition is None:
            return
        loss_fns = []
        coefs = []
        loss_names = []
        self.loss_fn_definition = loss_fn_definition
        print(loss_fn_definition)
        for loss in loss_fn_definition:
            # IF THE EXACT SAME LOSS ALREADY EXISTS, DON'T BUILD A NEW ONE
            if loss["name"] in self.loss_names:
                idx = self.loss_names.index(loss["name"])
                loss_fn = self.loss_fns[idx]
            elif loss["name"].lower().startswith("l1"):
                loss_fn = L1(**loss, device=self.device).lossfun
            elif loss["name"].lower().startswith("dtw"):
                loss_fn = DTWLoss(**loss, device=self.device).lossfun
            elif loss["name"].lower().startswith("barron"):
                loss_fn = AdaptiveLossFunction(num_dims=vocab_size, float_dtype=np.float32, device='cpu').lossfun
            elif loss["name"].lower().startswith("ssl"):
                loss_fn = SSL(**loss, device=self.device).lossfun
            elif loss["name"].lower().startswith("cross_entropy"):
                loss_fn = CrossEntropy(**loss,device=self.device).lossfun
            else:
                raise Exception(f"Unknown loss: {loss['name']}")

            loss_fns.append(loss_fn)
            loss_names.append(loss["name"])
            coefs.append(loss["coef"])

        self.coefs = Tensor(coefs)
        self.loss_fns = loss_fns
        self.loss_names = loss_names

    def main_loss(self, preds, targs, label_lengths, suffix):
        """ Preds: BATCH, TIME, VOCAB SIZE
                    VOCAB: x, y, start stroke, end_of_sequence
        Args:
            preds: Will be in the form [batch, width, alphabet]
            targs: Pass in the whole dictionary, so we can get lengths, etc., whatever we need

            suffix (str): _train or _test
        Returns:

        # Adapatively invert stroke targs if first instance is on the wrong end?? sounds sloooow

        """
        # elif isinstance(targs, dict):
        #     label_lengths = targs["label_lengths"]
        #     targs = targs["gt_list"]
        # elif label_lengths is None:
        #     label_lengths = [t.shape[0] for t in targs]
        losses = torch.zeros(len(self.loss_fns))
        batch_size = len(preds)
        total_points = tensor_sum(label_lengths)

        ## Loop through loss functions
        for i, loss_fn in enumerate(self.loss_fns):
            loss_tensor = loss_fn(preds, targs, label_lengths)
            loss = to_value(loss_tensor)
            #print(loss, loss_fn.__name__)
            assert loss > 0
            losses[i] = loss_tensor
            # Update loss stat
            self.stats[self.loss_names[i] + suffix].accumulate(loss)

        if suffix == "_train":
            self.counter.update(training_pred_count=total_points)
            #print(total_points)
        elif suffix == "_test":
            self.counter.update(test_pred_count=total_points)

        combined_loss = torch.sum(losses * self.coefs) # only for the actual gradient loss so that the loss doesn't change with bigger batch sizes;, not the reported one since it will be divided by instances later
        combined_loss_value = to_value(combined_loss)
        return combined_loss, combined_loss_value/batch_size # does the total loss makes most sense at the EXAMPLE level?

    def barron_loss(self, preds, targs, label_lengths, **kwargs):
        # BATCH, TIME, VOCAB
        vocab_size = preds.shape[-1]
        _preds = preds.reshape(-1, vocab_size)
        _targs = targs.reshape(-1, vocab_size)
        loss = torch.sum(self.barron_loss_fn((_preds - _targs)))
        return loss #, to_value(loss)

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

