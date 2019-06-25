import warnings
import torch
from torch import nn
from utils import *
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# Use ResNet?
# Increase LSTM dropout

import transformer.Constants as Constants
#from dataset import TranslationDataset, paired_collate_fn
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MAX_LENGTH=60
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=.5):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=True)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
	
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, bidirectional=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class BidirectionalLSTM2(nn.Module):

    def __init__(self, nIn, nHidden, nOut, dropout=.5):
        super(BidirectionalLSTM, self).__init__()

        self.encoder = EncoderRNN(nIn, nHidden, dropout=dropout, num_layers=2)
        self.encoder_hidden = self.encoder.initHidden()
        
        self.decoder = AttnDecoderRNN(nHidden, nOut, dropout=dropout, max_length=55)

    def forward(self, _in):
        embedding, hidden = self.encoder(_in, self.encoder_hidden)
        out, *_ = self.decoder(embedding, hidden)

        #T, b, h = out.size()
        #t_rec = out.view(T * b, h)

        #output = self.embedding(t_rec)  # [T * b, nOut]
        #output = output.view(T, b, -1) # Time, batch size, alphabet size

        return out

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut, dropout=.5):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True, dropout=dropout, num_layers=2)
        self.embedding = nn.Linear(nHidden * 2, nOut) # add dropout?

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

    def forward(self, input):
        # INPUT: BATCH, CHANNELS (1 or 3), Height, Width
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        conv = conv.view(b, -1, w) # batch, Height * Channels, Width

        # Width effectively becomes the "time" seq2seq variable
        output = conv.permute(2, 0, 1)  # [w, b, c], first time: [404, 8, 1024] ; second time: 213, 8, 1024
        return output

class CRNN(nn.Module):
    """ Original CRNN

    Modified to add some parameters to put it on even ground with the writer-classifier
    """
    def __init__(self, cnnOutSize, nc, alphabet_size, nh, n_rnn=2, leakyRelu=False, recognizer_dropout=.5):
        super(CRNN, self).__init__()

        self.cnn = CNN(cnnOutSize, nc, leakyRelu=leakyRelu)
        self.rnn = BidirectionalLSTM(cnnOutSize, nh, alphabet_size, dropout=recognizer_dropout)
        self.softmax = nn.LogSoftmax()

    def forward(self, input):
        conv = self.cnn(input)
        output = self.rnn(conv)
        return output,

class CRNN2(nn.Module):
    """ CRNN with writer classifier
        nh: LSTM dimension
    """
    def __init__(self, cnnOutSize, nc, alphabet_size, nh, n_rnn=2, number_of_writers=512, writer_rnn_output_size=128, leakyRelu=False,
                 embedding_size=64, writer_dropout=.5, writer_rnn_dimension=128, mlp_layers=(64, None, 128), recognizer_dropout=.5, detach_embedding=True):
        super(CRNN2, self).__init__()
        self.cnn = CNN(cnnOutSize, nc, leakyRelu=leakyRelu)
        self.rnn = BidirectionalLSTM(cnnOutSize + embedding_size, nh, alphabet_size, dropout=recognizer_dropout)
        self.writer_classifier = BidirectionalLSTM(cnnOutSize, writer_rnn_dimension, writer_rnn_output_size, dropout=writer_dropout)
        self.softmax = nn.LogSoftmax()
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
        #rnn_input = torch.cat([conv, online.expand(conv.shape[0], -1, -1)], dim=2)

    def forward(self, input):
        conv = self.cnn(input)

        # hwr classifier
        classifier_output1 = torch.mean(self.writer_classifier(conv),0,keepdim=False) # RNN dimensional vector
        classifier_output, embedding = self.mlp(classifier_output1, layer="output+embedding") # i.e. 671 dimensional vector

        # get embedding
        if self.detach_embedding:
            rnn_input = torch.cat([conv, embedding.expand(conv.shape[0], -1, -1).detach()], dim=2) # detach embedding
        else:
            rnn_input = torch.cat([conv, embedding.expand(conv.shape[0], -1, -1)], dim=2)  # detach embedding
        # rnn features
        recognizer_output = self.rnn(rnn_input)

        ## Append 16-bit embedding

        return recognizer_output, classifier_output

def create_CRNN(config):
    # For apples-to-apples comparison, CNN outsize is OUT_SIZE + EMBEDDING_SIZE
    crnn = CRNN(config['cnn_out_size'], config['num_of_channels'], config['alphabet_size'], config["rnn_dimension"], recognizer_dropout=config["recognizer_dropout"])

    return crnn

def create_CRNNClassifier(config):
    crnn = CRNN2(config['cnn_out_size'], config['num_of_channels'], config['alphabet_size'], nh=config["rnn_dimension"],
                 number_of_writers=config["num_of_writers"], writer_rnn_output_size=config['writer_rnn_output_size'], embedding_size=config["embedding_size"],
                 writer_dropout=config["writer_dropout"], recognizer_dropout=config["recognizer_dropout"], writer_rnn_dimension=config["writer_rnn_dimension"],
                 mlp_layers=config["mlp_layers"], detach_embedding=config["detach_embedding"])
    return crnn


    transformer = Transformer(
        opt.src_vocab_size,
        opt.tgt_vocab_size,
        opt.max_token_seq_len,
        tgt_emb_prj_weight_sharing=opt.proj_share_weight,
        emb_src_tgt_weight_sharing=opt.embs_share_weight,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout).to(device)













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
