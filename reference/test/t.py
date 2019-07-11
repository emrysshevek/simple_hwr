class CRNN2(nn.Module):
    """ CRNN with writer classifier
        nh: LSTM dimension
    """

    def __init__(self, cnnOutSize, nc, alphabet_size, nh, number_of_writers=512, writer_rnn_output_size=128, leakyRelu=False,
                 embedding_size=64, writer_dropout=.5, writer_rnn_dimension=128, mlp_layers=(64, None, 128), recognizer_dropout=.5,
                 detach_embedding=True, online_augmentation=False, use_writer_classifier=True):
        super(CRNN2, self).__init__()
        self.cnn = CNN(cnnOutSize, nc, leakyRelu=leakyRelu)
        self.softmax = nn.LogSoftmax()
        self.use_writer_classifier = use_writer_classifier

        rnn_expansion_dimension = embedding_size + 1 if online_augmentation else embedding_size
        self.rnn = BidirectionalLSTM(cnnOutSize + rnn_expansion_dimension, nh, alphabet_size, dropout=recognizer_dropout)

        if self.use_writer_classifier:
            self.writer_classifier = BidirectionalLSTM(cnnOutSize, writer_rnn_dimension, writer_rnn_output_size, dropout=writer_dropout)
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

        # hwr with classifier
        if self.use_writer_classifier:
            classifier_output1 = torch.mean(self.writer_classifier(conv),0,keepdim=False) # RNN dimensional vector
            classifier_output, embedding = self.mlp(classifier_output1, layer="output+embedding") # i.e. 671 dimensional vector

            # Attach style/classifier embedding
            if self.detach_embedding:
                rnn_input = torch.cat([conv, embedding.expand(conv.shape[0], -1, -1).detach()], dim=2) # detach embedding
            else:
                rnn_input = torch.cat([conv, embedding.expand(conv.shape[0], -1, -1)], dim=2)  # detach embedding
        # else, vanilla classifier
        else:
            rnn_input = conv

        # concatenate online flag as needed
        if online is not None:
            rnn_input = torch.cat([rnn_input, online.expand(conv.shape[0], -1, -1)], dim=2)

        # rnn features
        recognizer_output = self.rnn(rnn_input)

        return recognizer_output, classifier_output
