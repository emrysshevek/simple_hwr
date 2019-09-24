from torch import nn
from torch.nn import functional as F
from utils.utils import *
from utils.character_set import PAD_IDX
import os
from torch.autograd import Variable

from models.attention import Attention
from models.decoder import Decoder
from models.language_model import DeepFusion
from models.seq2seq import Seq2Seq

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# Use ResNet?
# Increase LSTM dropout


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LENGTH=60


class MLP(nn.Module):
    def __init__(self, input_size, classifier_output_dimension, hidden_layers, dropout=.5, embedding_idx=None):
        """

        Args:
            input_size (int): Dimension of input
            classifier_output_dimension (int): Dimension of output layer
            hidden_layers (list): A list of hidden layer dimensions
            dropout (float): 0 means no dropout
            embedding_idx: The hidden layer index of the embedding layer starting with 0
                e.g. input-> 16 nodes -> 8 nodes [embedding] -> 16 nodes would be 1
        """

        super(MLP, self).__init__()
        classifier = nn.Sequential()

        def addLayer(i, input, output, use_nonlinearity=True):
            classifier.add_module('drop{}'.format(i), nn.Dropout(dropout))
            classifier.add_module('fc{}'.format(i), nn.Linear(input, output))
            if use_nonlinearity:
                classifier.add_module('relu{}'.format(i), nn.ReLU(True))

        next_in = input_size
        for i, h in enumerate(hidden_layers):
            addLayer(i, next_in, h)
            next_in = h

        # Last Layer - don't use nonlinearity
        addLayer(len(hidden_layers), next_in, classifier_output_dimension, use_nonlinearity=False)
        self.classifier = classifier

        if embedding_idx is None:
            self.embedding_idx = len(classifier) # if no embedding specified, assume embedding=output
        else:
            self.embedding_idx = (embedding_idx + 1) * 3 # +1 for zero index, 3 items per layer

    def forward(self, input, layer="output"):
        input = input.view(input.shape[0], -1)

        if layer == "output":
            output = self.classifier(input) # batch size, everything else
            return output
        elif layer == "output+embedding":
            embedding = self.classifier[0:self.embedding_idx](input)
            output = self.classifier[self.embedding_idx:](embedding)  # batch size, everything else
            return output, embedding
        elif layer == "embedding":
            embedding = self.classifier[0:self.embedding_idx](input)
            return embedding


class BidirectionalRNN(nn.Module):

    def __init__(self, nIn, nHidden, nOut, dropout=.5, num_layers=2, rnn_constructor=nn.LSTM):
        super(BidirectionalRNN, self).__init__()
        print(f"Creating {rnn_constructor.__name__}: in:{nIn} hidden:{nHidden} dropout:{dropout} layers:{num_layers} out:{nOut}")
        self.rnn = rnn_constructor(nIn, nHidden, bidirectional=True, dropout=dropout, num_layers=num_layers)
        self.embedding = nn.Linear(nHidden * 2, nOut) # add dropout?

    def forward(self, input):
        # input [time size, batch size, output dimension], e.g. 404, 8, 1024
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CNN(nn.Module):
    def __init__(self, cnnOutSize=1024, nc=3, leakyRelu=False):
        """ Height must be set to be consistent; width is variable, longer images are fed into BLSTM in longer sequences

        The CNN learns some kind of sequential ordering because the maps are fed into the LSTM sequentially.

        Args:
            cnnOutSize: DOES NOT DO ANYTHING! Determined by architecture
            nc:
            leakyRelu:
        """

        super(CNN, self).__init__()

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

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16 = channels, height, width
        self.cnn = cnn

    def post_process(self, conv):
        b, c, h, w = conv.size()
        conv = conv.view(b, -1, w)  # batch, Height * Channels, Width

        # Width effectively becomes the "time" seq2seq variable
        output = conv.permute(2, 0, 1)  # [w, b, c], first time: [404, 8, 1024] ; second time: 213, 8, 1024
        return output

    def forward(self, input, intermediate_level=None):
        # INPUT: BATCH, CHANNELS (1 or 3), Height, Width
        if intermediate_level is None:
            return self.post_process(self.cnn(input))
        else:
            conv = self.cnn[0:intermediate_level](input)
            conv2 = self.cnn[intermediate_level:](conv)
            return self.post_process(conv2), self.post_process(conv)


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


class CRNN2(nn.Module):
    """ CRNN with writer classifier
        nh: LSTM dimension
    """

    def __init__(self, rnn_input_dim, nc, alphabet_size, nh, number_of_writers=512, writer_rnn_output_size=128, leakyRelu=False,
                 embedding_size=64, writer_dropout=.5, writer_rnn_dimension=128, mlp_layers=(64, None, 128), recognizer_dropout=.5,
                 detach_embedding=True, online_augmentation=False, use_writer_classifier=True, rnn_constructor=nn.LSTM):
        super(CRNN2, self).__init__()
        self.cnn = CNN(cnnOutSize=1024, nc=nc, leakyRelu=leakyRelu)
        self.softmax = nn.LogSoftmax()
        self.use_writer_classifier = use_writer_classifier

        self.rnn = BidirectionalRNN(rnn_input_dim, nh, alphabet_size, dropout=recognizer_dropout, rnn_constructor=rnn_constructor)

        if self.use_writer_classifier:
            self.writer_classifier = BidirectionalRNN(rnn_input_dim, writer_rnn_dimension, writer_rnn_output_size, dropout=writer_dropout)
            self.detach_embedding=detach_embedding

            ## Create a MLP on the end to create an embedding
            if "embedding" in mlp_layers:
                embedding_idx = mlp_layers.index("embedding")
                if embedding_idx != get_last_index(mlp_layers, "embedding"):
                    warnings.warn("Multiple dimensions in MLP specified as 'embedding'")
                mlp_layers = [m if m != "embedding" else embedding_size for m in mlp_layers] # replace None with embedding size
            else:
                embedding_idx = None

            self.mlp = MLP(writer_rnn_output_size, number_of_writers, mlp_layers, dropout=writer_dropout, embedding_idx=embedding_idx) # dropout = 0 means no dropout

    def forward(self, input, online=None, classifier_output=None):
        conv = self.cnn(input)

        # Vanilla classifier
        rnn_input = conv

        # concatenate online flag as needed
        if online is not None:
            rnn_input = torch.cat([rnn_input, online.expand(conv.shape[0], -1, -1)], dim=2)

        # concatenate hwr with classifier
        if self.use_writer_classifier:
            classifier_output1 = torch.mean(self.writer_classifier(rnn_input),0,keepdim=False) # RNN dimensional vector
            classifier_output, embedding = self.mlp(classifier_output1, layer="output+embedding") # i.e. 671 dimensional vector

            # Attach style/classifier embedding
            if self.detach_embedding:
                rnn_input = torch.cat([rnn_input, embedding.expand(conv.shape[0], -1, -1).detach()], dim=2) # detach embedding
            else:
                rnn_input = torch.cat([rnn_input, embedding.expand(conv.shape[0], -1, -1)], dim=2)  # keep embedding attached

        # rnn features
        recognizer_output = self.rnn(rnn_input)

        return recognizer_output, classifier_output


class CRNN_2Stage(nn.Module):
    """ CRNN with writer classifier
        nh: LSTM dimension
        nc: number of channels

    """
    def __init__(self, rnn_input_dim, nc, alphabet_size, rnn_hidden_dim, n_rnn=2, leakyRelu=False, recognizer_dropout=.5, online_augmentation=False,
                 first_rnn_out_dim=128, rnn_constructor=nn.LSTM):
        super(CRNN_2Stage, self).__init__()
        self.softmax = nn.LogSoftmax()
        self.cnn = CNN(1024, nc, leakyRelu=leakyRelu)
        self.first_rnn  = BidirectionalRNN(rnn_input_dim, rnn_hidden_dim, first_rnn_out_dim, dropout=recognizer_dropout, rnn_constructor=rnn_constructor)
        self.second_rnn = BidirectionalRNN(rnn_input_dim + first_rnn_out_dim, rnn_hidden_dim, alphabet_size, dropout=recognizer_dropout, rnn_constructor=rnn_constructor)

    def forward(self, input, online=None, classifier_output=None):
        conv = self.cnn(input)
        rnn_input = conv # [width/time, batch, feature_maps]

        if online is not None:
            rnn_input = torch.cat([rnn_input, online.expand(conv.shape[0], -1, -1)], dim=2)

        # First Stage
        first_stage_output = self.first_rnn(rnn_input)

        # Second stage
        cnn_rnn_concat = torch.cat([rnn_input, first_stage_output], dim=2)
        recognizer_output = self.second_rnn(cnn_rnn_concat)

        #print(first_stage_output.shape)
        #print(conv.shape)
        #print(cnn_rnn_concat.shape)

        return recognizer_output, rnn_input


class CRNN_UNet(nn.Module):
    """ CRNN with writer classifier
    """

    def __init__(self, cnnOutSize, nc, alphabet_size, rnn_hidden_dim, rnn_layers=2, leakyRelu=False,
                 recognizer_dropout=.5, online_augmentation=False, rnn_constructor=nn.LSTM):
        super(basic_CRNN, self).__init__()
        self.softmax = nn.LogSoftmax()
        self.dropout = recognizer_dropout
        rnn_expansion_dimension = 1 if online_augmentation else 0
        rnn_in_dim = cnnOutSize + rnn_expansion_dimension
        self.cnn = CNN(cnnOutSize, nc, leakyRelu=leakyRelu)
        self.rnn = BidirectionalRNN(rnn_in_dim, rnn_hidden_dim, alphabet_size, dropout=recognizer_dropout,
                                    num_layers=rnn_layers, rnn_constructor=rnn_constructor)

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
        rnn_input = conv  # [width/time, batch, feature_maps]

        if online is not None:
            rnn_input = torch.cat([rnn_input, online.expand(conv.shape[0], -1, -1)], dim=2)
        recognizer_output = self.rnn(rnn_input)
        return recognizer_output, rnn_input


class basic_CRNN(nn.Module):
    """ CRNN with writer classifier
    """
    def __init__(self, cnnOutSize, nc, alphabet_size, rnn_hidden_dim, rnn_layers=2, leakyRelu=False, recognizer_dropout=.5, online_augmentation=False, rnn_constructor=nn.LSTM):
        super(basic_CRNN, self).__init__()
        self.softmax = nn.LogSoftmax()
        self.dropout = recognizer_dropout
        rnn_expansion_dimension = 1 if online_augmentation else 0
        rnn_in_dim = cnnOutSize + rnn_expansion_dimension
        self.cnn = CNN(cnnOutSize, nc, leakyRelu=leakyRelu)
        self.rnn = BidirectionalRNN(rnn_in_dim, rnn_hidden_dim, alphabet_size, dropout=recognizer_dropout, num_layers=rnn_layers, rnn_constructor=rnn_constructor)

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


class Nudger(nn.Module):

    def __init__(self, rnn_input_dim, nc, rnn_hidden_dim, rnn_layers=2, rnn_dropout=.5, leakyRelu=False):
        """
        Args:
            final_rnn (nn.Module): The RNN that this output is added to
            rnn_input_dim: Dimension of RNN input - this should already include e.g. augmentation dimension flag
            nc: number of channels in image
            alphabet_size: Number of letters in output alphabet
            rnn_hidden_dim: Dimension of context/state vectors
            rnn_layers: Number of layers in RNN
            leakyRelu:
        """

        super(Nudger, self).__init__()
        self.nudger_rnn = BidirectionalRNN(rnn_input_dim, rnn_hidden_dim, rnn_input_dim, dropout=rnn_dropout, num_layers=rnn_layers, rnn_constructor=rnn_constructor)

    def forward(self, feature_maps, recognizer_rnn, classifier_output=None):
        """

        Args:
            feature_maps: The output of the CNN plus additional flags
            recognizer_rnn: The nn.Module that classifies text
            classifier_output:
        Returns:

        """
        # Nudger
        nudger_output = self.nudger_rnn(feature_maps)

        # Second stage
        nudged_cnn_encoding = feature_maps + nudger_output
        recognizer_output_refined = recognizer_rnn(nudged_cnn_encoding)
        return recognizer_output_refined, nudged_cnn_encoding


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

