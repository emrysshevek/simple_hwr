import torch
from torch import nn

# Use ResNet?
# Shuffle data-loaders?



class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True, dropout=0.5, num_layers=2)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output

class CNN(nn.Module):
    def __init__(self, cnnOutSize, nc, leakyRelu=False):
        """ Height must be set to be consistent; width is variable, longer images are fed into BLSTM in longer sequences
        Args:
            cnnOutSize:
            nc:
            leakyRelu:
        """

        super(CNN, self).__init__()

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
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
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn

    def forward(self, input):
        # INPUT: BATCH, CHANNELS (3), Height, Width
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        conv = conv.view(b, -1, w) # batch, Height * Channels, Width

        # Width effectively becomes the "time" seq2seq variable
        output = conv.permute(2, 0, 1)  # [w, b, c], first time: [404, 8, 1024] ; second time: 213, 8, 1024

        return output

class CRNN(nn.Module):
    """ Original CRNN
    """
    def __init__(self, cnnOutSize, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()

        self.cnn = CNN(cnnOutSize, nc, leakyRelu=leakyRelu)
        self.rnn = BidirectionalLSTM(cnnOutSize, nh, nclass)
        self.softmax = nn.LogSoftmax()

    def forward(self, input):
        conv = self.cnn(input)
        # rnn features
        output = self.rnn(conv)
        return output

class CRNN2(nn.Module):
    """ CRNN with writer classifier
        nh: LSTM dimension
    """
    def __init__(self, cnnOutSize, nc, alphabet_size, nh, n_rnn=2, class_size=512, leakyRelu=False, embedding_input_size=64, dropout=.5, writer_rnn_output_size=128):
        super(CRNN2, self).__init__()
        self.cnn = CNN(cnnOutSize, nc, leakyRelu=leakyRelu)
        self.rnn = BidirectionalLSTM(cnnOutSize+embedding_input_size, nh, alphabet_size)
        self.writer_classifier = BidirectionalLSTM(cnnOutSize, 128, class_size)
        self.softmax = nn.LogSoftmax()

        ## Create a MLP on the end to create an embedding
        self.embedding_input_size = embedding_input_size
        self.mlp = MLP(class_size, class_size, [64,embedding_input_size,128], dropout=dropout) # dropout = 0 means no dropout
        #rnn_input = torch.cat([conv, online.expand(conv.shape[0], -1, -1)], dim=2)

    def forward(self, input):
        conv = self.cnn(input)

        # hwr classifier
        classifier_output1 = torch.mean(self.writer_classifier(conv),0,keepdim=False) # 671 dimensional vector
        classifier_output, embedding = self.mlp(classifier_output1, layer="output+embedding")

        # get embedding
        rnn_input = torch.cat([conv, embedding.expand(conv.shape[0], -1, -1)], dim=2) # duplicate embedding across

        # rnn features
        recognizer_output = self.rnn(rnn_input)

        ## Append 16-bit embedding

        return recognizer_output, classifier_output

def create_CRNN(config):
    crnn = CRNN(config['cnn_out_size'], config['num_of_channels'], config['alphabet_size'], 512)
    return crnn

def create_CRNNClassifier(config):
    crnn = CRNN2(config['cnn_out_size'], config['num_of_channels'], config['alphabet_size'], nh=512, 
                 class_size=config["num_of_writers"], embedding_input_size=config["embedding_size"], dropout=config["dropout"], writer_rnn_output_size=config["writer_rnn_output_size"])
    return crnn

class MLP(nn.Module):
    def __init__(self, input_size, classifier_output_dimension, hidden_layers, dropout=.9):

        super(MLP, self).__init__()
        classifier = nn.Sequential()

        def addLayer(i, input, output, dropout=False, use_nonlinearity=True):
            classifier.add_module('drop{}'.format(i), nn.Dropout(dropout))
            classifier.add_module('fc{}'.format(i), nn.Linear(input, output))
            if use_nonlinearity:
                classifier.add_module('relu{}'.format(i), nn.ReLU(True))

        next_in = input_size
        for i, h in enumerate(hidden_layers):
            addLayer(i, next_in, h)
            next_in = h
        addLayer(i+1, next_in, classifier_output_dimension, use_nonlinearity=False)
        self.classifier = classifier

    def forward(self, input, layer="output"):
        input = input.view(input.shape[0], -1)

        if layer == "output":
            output = self.classifier(input) # batch size, everything else
            return output
        elif layer == "output+embedding":
            embedding = self.classifier[0:6](input)
            output = self.classifier[6:](embedding)  # batch size, everything else
            return output, embedding
        elif layer == "embedding":
            embedding = self.classifier[0:6](input)
            return embedding
