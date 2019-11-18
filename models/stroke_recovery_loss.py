import torch
import robust_loss_pytorch
import numpy as np
import torch.nn as nn
from pydtw import dtw
from scipy import spatial
from robust_loss_pytorch import AdaptiveLossFunction
#from sdtw import SoftDTW
import bezier
import torch.multiprocessing as multiprocessing
from hwr_utils.utils import to_numpy
from hwr_utils.stroke_dataset import pad

class StrokeLoss:
    def __init__(self, loss_type="robust", parallel=False):
        super(StrokeLoss, self).__init__()
        #device = torch.device("cuda")
        if loss_type == "robust":
            self.bonus_loss = AdaptiveLossFunction(num_dims=5, float_dtype=np.float32, device='cpu').lossfun
        else:
            self.bonus_loss = None

        self.cosine_similarity = nn.CosineSimilarity(dim=1)
        self.cosine_distance = lambda x, y: 1 - self.cosine_similarity(x, y)
        self.distributions = None
        self.parallel = parallel
        self.poolcount = max(1, multiprocessing.cpu_count()-8)
        self.poolcount = 2

    def main_loss(self, preds, targs, label_lengths=None, vocab_size=4):
        """ Preds: [x], [y], [start stroke], [end stroke], [end of sequence]

        Args:
            preds: Will be in the form [batch, width, alphabet]
            targs: Pass in the whole dictionary, so we can get lengths, etc., whatever we need

        Returns:

        # Adapatively invert stroke targs if first instance is on the wrong end?? sounds sloooow

        """
        if isinstance(targs, dict):
            label_lengths = targs["label_lengths"]
            targs = targs["gt_list"]
        else:
            label_lengths = [1] * len(targs)

        if self.bonus_loss:
            _preds = preds.reshape(-1,vocab_size)
            _targs = targs.reshape(-1,vocab_size)
            return torch.sum(self.bonus_loss((_preds-_targs)))
        else:
            loss = self.loop(preds, targs, label_lengths)
            return loss

    def loop(self, preds, targs, label_lengths):
        loss = 0
        #preds2 = [to_numpy(x) for x in preds.detach()]
        #pool = self.pool
        if self.parallel:
            pool = multiprocessing.Pool(processes=self.poolcount)
            as_and_bs = pool.imap(StrokeLoss.dtw, iter(zip(
                to_numpy(preds),
                targs)), chunksize=32)  # iterates through everything all at once
            pool.close()
            #print(as_and_bs[0])
            for i, (a,b) in enumerate(as_and_bs): # loop through BATCH
                loss += abs(preds[i, a, :] - targs[i][b, :]).sum() / label_lengths[i]
        else:
            # pool = multiprocessing.Pool(processes=self.poolcount)
            # pool.close()
            
            samples = self.sample_from_curves(preds)
            #samples_np = to_numpy(samples, astype="float64")
            #print("pre: ", len(targs), targs[0].shape)
            #targs_np = np.array([x.tolist() for x in targs])
            #samples_np = np.ascontiguousarray(samples_np[:, :, :])
            #print(targs_np)
            #print("Targs np shape: ", targs_np.shape)
            #targs_np = np.ascontiguousarray(targs_np[:, :, :])
            for i in range(len(preds)): # loop through BATCH
                
                #a, b = dtw2d.dtw(samples_np[i], targs_np[i, :, :2])
                a, b = self.dtw((samples[i], targs[i][:, :2]))
                loss += abs(samples[i][a, :] - targs[i][b, :2]).sum() / label_lengths[i]
                #a,b = self.dtw((preds[i], targs[i]))
                #loss += abs(preds[i, a, :] - targs[i][b, :]).sum() / label_lengths[i]
        print("loss: ", loss)
        return loss

    @staticmethod
    def sample_from_curves(preds):
        bezier_coefs = torch.t(torch.Tensor([[(1 - t)**2, 2*t*(1 - t), t**2] for t in np.linspace(0.0, 1.0, 5)])).unsqueeze(0)
        #bezier_coefs = torch.t(torch.Tensor([[(1 - t)**3, 3*t*(1 - t)**2, 3*t*(1 - t)**2, t**3] for t in np.linspace(0.0, 1.0, 12)]))
        xs = torch.matmul(preds[:, :, :3], bezier_coefs)
        ys = torch.matmul(preds[:, :, 3:], bezier_coefs)
        zipped = torch.stack((xs, ys), 3)
        samples = zipped.view(zipped.shape[0], zipped.shape[1] * zipped.shape[2], zipped.shape[3])
        return samples

    @staticmethod
    def dtw(input):
        """
        Args:
            input (tuple): targ, pred, label_length

        Returns:

        """
        pred, targ = input
        pred, targ = to_numpy(pred, astype="float64"), to_numpy(targ, astype="float64")
        x1 = np.ascontiguousarray(pred[:, :])  # time step, batch, (x,y)
        x2 = np.ascontiguousarray(targ[:, :])
        dist, cost, a, b = dtw.dtw2d(x1, x2)

        # Cost is weighted by how many GT stroke points, i.e. how long it is
        return a,b

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
    vocab_size = 4
    batch = 3
    time = 16
    y = torch.rand(batch, 1, 60, 60)
    targs = torch.rand(batch, time, vocab_size)
    cnn = CNN(nc=1)
    rnn = BidirectionalRNN(nIn=1024, nHidden=128, nOut=vocab_size, dropout=.5, num_layers=2, rnn_constructor=nn.LSTM)
    cnn_output = cnn(y)
    rnn_output = rnn(cnn_output).permute(1, 0, 2)
    print(rnn_output.shape) # BATCH, TIME, VOCAB
    loss = StrokeLoss(loss_type="")
    loss = loss.main_loss(rnn_output, targs)
    print(loss)



