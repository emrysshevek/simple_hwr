import torch
from torch import nn
from hwr_utils import *
import os
from torch.autograd import Variable
from basic import *


class CNN(nn.Module):
    def __init__(self, cnnOutSize=1024, nc=3, leakyRelu=False, type="default"):
        """ Height must be set to be consistent; width is variable, longer images are fed into BLSTM in longer sequences

        The CNN learns some kind of sequential ordering because the maps are fed into the LSTM sequentially.

        Args:
            cnnOutSize: DOES NOT DO ANYTHING! Determined by architecture
            nc:
            leakyRelu:
        """
        super(CNN, self).__init__()
        self.cnnOutSize = cnnOutSize
        #self.average_pool = nn.AdaptiveAvgPool2d((512,2))
        self.pool = nn.MaxPool2d(3, (4, 1), padding=1)
        self.intermediate_pass = 13 if type == "intermediates" else None

        print("Intermediate pass {}".format(self.intermediate_pass))

        if type in ["default", "intermediates"]:
            self.cnn = self.default_CNN(nc=nc, leakyRelu=leakyRelu)
        elif "resnet" in type:
            from models import resnet
            if type=="resnet":
                #self.cnn = torchvision.models.resnet101(pretrained=False)
                self.cnn = resnet.resnet18(pretrained=False, channels=nc)
            elif type=="resnet34":
                self.cnn = resnet.resnet34(pretrained=False, channels=nc)
            elif type=="resnet101":
                self.cnn = resnet.resnet101(pretrained=False, channels=nc)


    def default_CNN(self, nc=3, leakyRelu=False):

        ks = [3, 3, 3, 3, 3, 3, 2] # kernel size 3x3
        ps = [1, 1, 1, 1, 1, 1, 0] # padding
        ss = [1, 1, 1, 1, 1, 1, 1] # stride
        nm = [64, 128, 256, 256, 512, 512, 512] # number of channels/maps

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(in_channels=nIn, out_channels=nOut, kernel_size=ks[i], stride=ss[i], padding=ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))
            #cnn.add_module(f"printAfter{i}", PrintLayer(name=f"printAfter{i}"))

        # input: 16, 1, 60, 256; batch, channels, height, width
        convRelu(0) # 16, 64, 60, 1802
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 16, 64, 30, 901
        convRelu(1) # 16, 128, 30, 901
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 16, 128, 15, 450
        convRelu(2, True) # 16, 256, 15, 450
        convRelu(3) # 16, 256, 15, 450
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 16, 256, 7, 451 # kernel_size, stride, padding
        convRelu(4, True) # 16, 512, 7, 451
        convRelu(5) # 16, 512, 7, 451
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 16, 512, 3, 452
        convRelu(6, True)  # 16, 512, 2, 451
        return cnn

    def post_process(self, conv):
        b, c, h, w = conv.size() # something like 16, 512, 2, 406
        #print(conv.size())
        conv = conv.view(b, -1, w)  # batch, Height * Channels, Width

        # Width effectively becomes the "time" seq2seq variable
        output = conv.permute(2, 0, 1)  # [w, b, c], first time: [404, 8, 1024] ; second time: 213, 8, 1024
        return output

    def intermediate_process(self, final, intermediate):
        new = self.post_process(self.pool(intermediate))
        final = self.post_process(final)
        return torch.cat([final, new], dim=2)

    def forward(self, input):
        # INPUT: BATCH, CHANNELS (1 or 3), Height, Width
        if self.intermediate_pass is None:
            x = self.post_process(self.cnn(input))
            #assert self.cnnOutSize == x.shape[1] * x.shape[2]
            return x
        else:
            conv = self.cnn[0:self.intermediate_pass](input)
            conv2 = self.cnn[self.intermediate_pass:](conv)
            final = self.intermediate_process(conv2, conv)
            return final

class CRNN(nn.Module):
    """ Original CRNN

    Modified to add some parameters to put it on even ground with the writer-classifier
    """
    def __init__(self, cnnOutSize, nc, alphabet_size, nh, n_rnn=2, leakyRelu=False, recognizer_dropout=.5, rnn_constructor=nn.LSTM):
        super(CRNN, self).__init__()

        self.cnn = CNN(cnnOutSize, nc, leakyRelu=leakyRelu)
        self.rnn = BidirectionalRNN(cnnOutSize, nh, alphabet_size, dropout=recognizer_dropout, rnn_constructor=rnn_constructor)
        self.softmax = nn.LogSoftmax()

    def forward(self, input):
        conv = self.cnn(input)
        output = self.rnn(conv)
        return output,