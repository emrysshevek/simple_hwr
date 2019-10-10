import torch
import robust_loss_pytorch
import numpy as np
import torch.nn as nn
from robust_loss_pytorch import AdaptiveLossFunction


class StrokeLoss:
    def __init__(self, loss_type="robust"):
        super(StrokeLoss, self).__init__()
        #device = torch.device("cuda")
        if loss_type == "robust":
            self.bonus_loss = AdaptiveLossFunction(num_dims=5, float_dtype=np.float32, device='cpu').lossfun
        else:
            self.bonus_loss = None

    def main_loss(self, preds, targs, label_lengths=16):
        """ Preds: [x], [y], [start stroke], [end stroke], [end of sequence]

        Args:
            preds: Will be in the form [batch, width, alphabet]
            targs:

        Returns:

        # Adapatively invert stroke targs if first instance is on the wrong end?? sounds sloooow

        """

        # print(preds.shape, targs.shape)
        # print(preds.dtype, targs.dtype)
        if self.bonus_loss:
            _preds = preds.reshape(-1,5)
            _targs = targs.reshape(-1,5)
            # print(_preds.shape, _targs.shape)
            return torch.sum(self.bonus_loss((_preds-_targs)))
        else:
            return abs(preds-targs).sum()

if __name__ == "__main__":
    from models.basic import CNN, BidirectionalRNN
    from torch import nn
    batch = 3
    y = torch.rand(batch, 1, 60, 60)
    targs = torch.rand(batch, 16, 5)
    cnn = CNN(nc=1)
    rnn = BidirectionalRNN(nIn=1024, nHidden=128, nOut=5, dropout=.5, num_layers=2, rnn_constructor=nn.LSTM)
    cnn_output = cnn(y)
    rnn_output = rnn(cnn_output).permute(1, 0, 2)
    print(rnn_output.shape)
    loss = loss_fnc(rnn_output, targs)
    print(loss)
