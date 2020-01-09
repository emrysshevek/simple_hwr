import torch
import robust_loss_pytorch
import numpy as np
import torch.nn as nn
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

BCELoss = torch.nn.BCELoss()

def to_value(loss_tensor):
    return torch.sum(loss_tensor.cpu(), 0, keepdim=False).item()

def tensor_sum(tensor):
    return torch.sum(tensor.cpu(), 0, keepdim=False).item()

class StrokeLoss:
    def __init__(self, loss_names="l1", parallel=False, vocab_size=4, loss_stats=None, counter=None):
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
        self.counter = Counter() if counter is None else counter
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

        self.coefs = coefs
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
        cum_loss = 0

        ## Loop through loss functions
        for i, loss_fn in enumerate(self.loss_fns):
            loss_tensor = loss_fn(preds, targs, label_lengths) * self.coefs[i] ## TEMPORARY WHILE BALANCING LOSS FUNCTIONS! NORMALLY DON'T MULTIPLY BY COEF UNTIL AFTER LOSSES SAVED (SO THAT THEY ARE ALL ON THE SAME SCALE)
            loss = to_value(loss_tensor)
            cum_loss += loss # adjusted to be training-instance based later; don't bother to do point-based right now
            losses[i] = loss_tensor

            # Update loss stat
            self.stats[self.loss_names[i] + suffix].accumulate(loss)

        if suffix == "_train":
            self.counter.update(training_pred_count=total_points)
        elif suffix == "_test":
            self.counter.update(test_pred_count=total_points)

        combined_loss = torch.sum(losses) / batch_size # only for the actual gradient loss so that the loss doesn't change with bigger batch sizes;, not the reported one since it will be divided by instances later
        return combined_loss, cum_loss

    def dtw(self, preds, targs, label_lengths, **kwargs):
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
        return loss #, to_value(loss)

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

    def barron_loss(self, preds, targs, label_lengths, **kwargs):
        # BATCH, TIME, VOCAB
        vocab_size = preds.shape[-1]
        _preds = preds.reshape(-1, vocab_size)
        _targs = targs.reshape(-1, vocab_size)
        loss = torch.sum(self.barron_loss_fn((_preds - _targs)))
        return loss #, to_value(loss)

    @staticmethod
    def variable_l1(preds, targs, label_lengths, **kwargs):
        """ Resmaple the targets to match whatever was predicted

        Args:
            preds:
            targs:
            label_lengths:

        Returns:

        """
        targs, label_lengths = StrokeLoss.resample_gt(preds, targs)
        loss = StrokeLoss.l1(preds, targs, label_lengths) # already takes average loss
        return loss #, to_value(loss)

    @staticmethod
    def l1(preds, targs, label_lengths, **kwargs):
        loss = 0
        for i, pred in enumerate(preds):
          loss += torch.sum(abs(pred[:, :2]-targs[i][:, :2]))
        return loss #, to_value(loss)

    @staticmethod
    def ssl(preds, targs, label_lengths):
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

        for i in range(len(preds)):
            targ_start_strokes = targs[i][torch.nonzero(targs[i][:, 2]).squeeze(1), :2]
            targ_end_strokes = (targ_start_strokes-1)[1:] # get corresponding end strokes - this excludes the final stroke point!!
            k = KDTree(preds[i][:, :2].data)

            # Start indices
            start_indices = k.query(targ_start_strokes)[1]
            pred_gt_fitted = torch.zeros(preds[i].shape[0])
            pred_gt_fitted[start_indices] = 1

            # End indices
            end_indices = k.query(targ_end_strokes)[1]
            # pred_end_fitted = torch.zeros(preds[i].shape[0])
            # pred_end_fitted[end_indices] = 1

            # Do L1 distance loss for start strokes and nearest stroke point
            # loss_tensor += abs(preds[i][start_indices, :2] - targ_start_strokes).sum()
            # loss_tensor += abs(preds[i][end_indices, :2] - targ_end_strokes).sum()

            # Do SOStr classification loss
            # print("preds", pred_gt_fitted)
            loss_tensor += BCELoss(preds[i][:, 2], pred_gt_fitted)

            # print(targ_start_strokes, start_indices)
            # input()
            # print(pred_gt_fitted)
            # input()
            # print(targs[i][:,2])
            # print(loss_tensor)
            # input()

            # Do EOSeq prediction - not totally fair, again, we should evaluate it based on the nearest point to the last prediction
            loss_tensor += BCELoss(preds[i][:, 3], targs[i][:, 3])

            # # Do L1 distance loss
            # loss += abs(preds[i][start_indices, :2] - targ_start_strokes).sum() / len(start_indices)
            # loss += abs(preds[i][end_indices, :2] - targ_end_strokes).sum() / len(end_indices)
            # loss += 0.1 * abs(pred_gt_fitted - targs[i][:, 2]).sum()
            # loss += 0.1 * abs(pred_end_fitted - targs[i][:, 3]).sum()
        loss = to_value(loss_tensor)
        #print("Time to compute ssl: ", time.time() - start_time)
        return loss_tensor #, loss

    def juncture_is_correct(gts, preds):
    # Algorithm: make segments along the function, then O(n^2) check
    # the segments against each other for intersection.  If an intersection is
    # found, move along the path until finding a postprocessed prediction point,
    # then ensure that the postprocessed prediction points are in the same order
    # in time as the ground truth points.
        def postprocess(preds, kd, gts):
            _, closest = kd.query(preds)
            return [gts[i] for i in closest], set(closest), closest

        def seek(i, move, corrected, which):
            while i not in corrected:
                i = move(i)
            first_pred_match = next((j for j in range(len(which)) if which[j] == i), None)
            return first_pred_match

        def line_line(p1, q1, p2, q2):
            line1 = (p1, q1)
            line2 = (p2, q2)
            xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
            ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

            def det(a, b):
                return a[0] * b[1] - a[1] * b[0]

            div = det(xdiff, ydiff)
            if div == 0:
               raise Exception('lines do not intersect')

            d = (det(*line1), det(*line2))
            x = det(d, xdiff) / div
            y = det(d, ydiff) / div
            return x, y

        def intersect(p1, q1, p2, q2):
            def on_segment(p, q, r):
                return q[0] < max(p[0], r[0]) and q[1] > min(p[0], r[0]) and \
                       q[1] < max(p[1], r[1]) and q[1] > min(p[1], r[1])

            def orientation(p, q, r):
                v = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
                return 0 if v == 0 else (1 if v > 0 else 2)
            o1 = orientation(p1, q1, p2)
            o2 = orientation(p1, q1, q2)
            o3 = orientation(p2, q2, p1)
            o4 = orientation(p2, q2, q1)

            if o1 != o2 and o3 != o4:
                return line_line(p1, q1, p2, q2)

            if o1 == 0 and on_segment(p1, p2, q1):
                return line_line(p1, q1, p2, q2)
            if o2 == 0 and on_segment(p1, q2, q1):
                return line_line(p1, q1, p2, q2)
            if o3 == 0 and on_segment(p2, p1, q2):
                return line_line(p1, q1, p2, q2)
            if o4 == 0 and on_segment(p2, q1, q2):
                return line_line(p1, q1, p2, q2)

        kd = KDTree(gts)
        gt_seg = [(gts[i - 1], gts[i]) for i in range(1, len(gts))]
        post, corrected, which_corrected = postprocess(preds, kd, gts)
        for i in range(len(gt_seg) - 1):
            for j in range(i + 2, len(gt_seg)):
                correct = False
                gt_int = intersect(gt_seg[i][0], gt_seg[i][1], gt_seg[j][0], gt_seg[j][1])
                if gt_int is not None:
                    # move forward and back on the line until get corrected
                    a = seek(i - 1, lambda x: x - 1, corrected, which_corrected)
                    b = seek(i, lambda x: x + 1, corrected, which_corrected[a + 1:])
                    if b is not None:
                        c = seek(j - 1, lambda x: x - 1, corrected, which_corrected[a + b + 1:])
                        if c is not None:
                            d = seek(j, lambda x: x + 1, corrected, which_corrected[a + b + c + 1:])
                            if d is not None:
                                correct = True
                    if not correct:
                        return False
        return True
  

    def calculate_nn_distance(self, item, preds):
        """ Can this be done differentiably?

        Args:
            item:
            preds:

        Returns:

        """
        return 0
        # calculate NN distance
        # n_pts = 0
        # cum_dist = 0
        # gt = item["gt_list"]
        # batch_size = len(gt)
        # for i in range(batch_size):
        #     # TODO binarize line images and do dist based on that
        #     kd = KDTree(preds[i][:, :2].data)
        #     cum_dist = sum(kd.query(gt[i][:, :2])[0])
        #     n_pts += gt[i].shape[0]
        #
        # return (cum_dist / n_pts) * batch_size # THIS WILL BE DIVIDED BY THE NUMBER OF INSTANCES LATER
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

