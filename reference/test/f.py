class CRNN_2Stage(nn.Module):
    """ CRNN with writer classifier
        nh: LSTM dimension
    """
    def __init__(self, cnnOutSize, nc, alphabet_size, rnn_hidden_dim, n_rnn=2, leakyRelu=False, recognizer_dropout=.5, online_augmentation=False,
                 first_rnn_out_dim=128, second_rnn_out_dim=512):
        super(CRNN_2Stage, self).__init__()
        self.softmax = nn.LogSoftmax()
        rnn_expansion_dimension = 1 if online_augmentation else 0

        self.cnn = CNN(cnnOutSize, nc, leakyRelu=leakyRelu)
        self.first_rnn  = BidirectionalLSTM(cnnOutSize + rnn_expansion_dimension, rnn_hidden_dim, first_rnn_out_dim, dropout=0)
        self.second_rnn = BidirectionalLSTM(cnnOutSize + rnn_expansion_dimension + first_rnn_out_dim, rnn_hidden_dim, alphabet_size, dropout=recognizer_dropout)

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

        return recognizer_output,
