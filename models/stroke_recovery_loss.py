import torch
import robust_loss_pytorch
import numpy as np
import torch.nn as nn
from pydtw import dtw
from scipy import spatial
from robust_loss_pytorch import AdaptiveLossFunction
from sdtw import SoftDTW

class StrokeLoss:
    def __init__(self, loss_type="robust"):
        super(StrokeLoss, self).__init__()
        #device = torch.device("cuda")
        if loss_type == "robust":
            self.bonus_loss = AdaptiveLossFunction(num_dims=5, float_dtype=np.float32, device='cpu').lossfun
        else:
            self.bonus_loss = None

        self.cosine_similarity = nn.CosineSimilarity(dim=1)
        self.cosine_distance = lambda x, y: 1 - self.cosine_similarity(x, y)
        self.distributions = None

    def main_loss(self, preds, targs, label_lengths=16):
        """ Preds: [x], [y], [start stroke], [end stroke], [end of sequence]

        Args:
            preds: Will be in the form [batch, width, alphabet]
            targs:

        Returns:

        # Adapatively invert stroke targs if first instance is on the wrong end?? sounds sloooow

        """

        if self.bonus_loss:
            _preds = preds.reshape(-1,5)
            _targs = targs.reshape(-1,5)
            return torch.sum(self.bonus_loss((_preds-_targs)))
        else:
            #location_loss = abs(preds - targs).sum()
            #logp_loss = self.log_prob_loss(preds, targs)
            #angle_loss = self.angle_loss(preds, targs)
            #print(f"{location_loss:.1f} {angle_loss:.1f} {logp_loss:.1f}")
            #return logp_loss * 0.8 + angle_loss * 0.2
            #return logp_loss * 0.6 + angle_loss * 0.4 + abs(preds[:, :, 2:] - targs[:, :, 2:]).sum()

            loss = 0
            for i in range(len(preds)): # loop through timesteps
                x1 = np.ascontiguousarray(preds[i, :, :2].detach().numpy()).astype("float64") # time step, batch, (x,y)
                x2 = np.ascontiguousarray(targs[i, :, :2].detach().numpy()).astype("float64")
                dist, cost, a, b = dtw.dtw2d(x1, x2)
                loss += abs(preds[i, a, :] - targs[i, b, :]).sum()
            print(loss)
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



    def soft_dtw(self):
        pass

        # Time series 1: numpy array, shape = [m, d] where m = length and d = dim
        # Time series 2: numpy array, shape = [n, d] where n = length and d = dim

        # D can also be an arbitrary distance matrix: numpy array, shape [m, n]
        D = SquaredEuclidean(X, Y)
        sdtw = SoftDTW(D, gamma=1.0)
        # soft-DTW discrepancy, approaches DTW as gamma -> 0
        value = sdtw.compute()
        # gradient w.r.t. D, shape = [m, n], which is also the expected alignment matrix
        E = sdtw.grad()
        # gradient w.r.t. X, shape = [m, d]
        G = D.jacobian_product(E)


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
    print(rnn_output.shape) # BATCH, TIME, VOCAB
    loss = StrokeLoss(loss_type="")
    loss = loss.main_loss(rnn_output, targs)
    print(loss)



