from collections import OrderedDict

from torch import nn
from torch.autograd import Variable
#from torchvision.models import resnet

# Use ResNet?
# Increase LSTM dropout

from utils.hwr_utils import *
from models.CRCR import CRCR
from models.deprecated_crnn import *


MAX_LENGTH = 60


class Encoder(nn.Module):

    def __init__(self, output_dim, dropout=0.5):
        super(Encoder, self).__init__()
        self.cnn = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(6, 4), stride=(4, 2))),
            ('relu1', nn.LeakyReLU()),
            ('conv2', nn.Conv2d(in_channels=8, out_channels=32, kernel_size=(6, 4), stride=(1, 1), padding=(1, 1))),
            ('relu2', nn.LeakyReLU()),
            ('pool1', nn.MaxPool2d(kernel_size=(4, 2), stride=(4, 2))),
            ('conv3', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1))),
            ('relu3', nn.LeakyReLU()),
            ('pool2', nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)))
        ]))
        self.rnn = nn.LSTM(input_size=64, hidden_size=128, num_layers=3, dropout=dropout, bidirectional=True, batch_first=True)
        self.conv4 = nn.Conv2d(in_channels=1, out_channels=output_dim, kernel_size=(256, 1), stride=(1, 1))

    def forward(self, x, online=None, classifier_output=None):
        """

        :param x: [batch_size, n_channels, height, width]
        :param online:
        :param classifier_output:
        :return x: [seq_len, batch_size, vocab_dim]
        :return rnn_input: [batch_size, seq_len, vocab_dim]
        """

        rnn_input = self.cnn(x).squeeze().permute(0, 2, 1)
        # print(rnn_input.shape)
        x, _ = self.rnn(rnn_input)
        x = x.permute(0, 2, 1).unsqueeze(dim=1)
        # print(x.shape)
        x = self.conv4(x).squeeze().permute(2, 0, 1)
        # print(x.shape)
        return x, rnn_input


class basic_CRNN(nn.Module):
    """ CRNN with writer classifier
    """
    def __init__(self, cnnOutSize, nc, alphabet_size, rnn_hidden_dim, rnn_layers=2, leakyRelu=False, recognizer_dropout=.5, rnn_input_dimension=1024, rnn_constructor=nn.LSTM, cnn_type="default"):
        super().__init__()
        self.softmax = nn.LogSoftmax()
        self.dropout = recognizer_dropout
        if cnn_type in ["default", "intermediates"] or "resnet" in cnn_type:
            self.cnn = CNN(cnnOutSize, nc, leakyRelu=leakyRelu, type=cnn_type)
        elif cnn_type=="crcr":
            self.cnn = CRCR(cnnOutSize, nc, leakyRelu=leakyRelu, type=cnn_type)
        else:
            raise Exception("Invalid CNN specified")
        self.rnn = BidirectionalRNN(rnn_input_dimension, rnn_hidden_dim, alphabet_size, dropout=recognizer_dropout, num_layers=rnn_layers, rnn_constructor=rnn_constructor)

    def my_eval(self):
        self.rnn.rnn.dropout = 0

    def my_train(self):
        self.rnn.rnn.dropout = self.dropout

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.my_eval()

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True
        self.my_train()

    def forward(self, input, online=None, classifier_output=None):
        """

        Args:
            input:
            online:
            classifier_output:

        Returns:
            tuple: normal prediction, refined prediction, normal CNN encoding, nudged CNN encoding

        """
        conv = self.cnn(input)
        rnn_input = conv # [width/time, batch, feature_maps]

        if online is not None:
            rnn_input = torch.cat([rnn_input, online.expand(conv.shape[0], -1, -1)], dim=2)
        recognizer_output = self.rnn(rnn_input)
        return recognizer_output, rnn_input


class LabelSmoothing(torch.nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = torch.nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

