import torch
from torch import nn

from models.crnn import basic_CRNN, LabelSmoothing
from models.deprecated_crnn import CRNN, CRNN_with_writer_classifier, CRNN_2Stage, Nudger
from models.attention import Attention
from models.decoder import Decoder
from models.language_model import DeepFusion
from models.seq2seq import Seq2Seq


def check_inputs(config):
    if not config["style_encoder"] or config["style_encoder"] in ["2StageNudger", "2Stage"]:
        use_writer_classifier = False
        config["embedding_size"] = 0
        config["num_of_writers"] = 0
        config['writer_rnn_output_size'] = 0
        config["embedding_size"] = 0
        config["writer_dropout"] = 0
        config["mlp_layers"] = []

    # Setup RNN input dimension
    config["rnn_input_dimension"] = config["cnn_out_size"] + config["embedding_size"]
    if config["online_augmentation"]:
        config["rnn_input_dimension"] += 1

    if config["rnn_type"].lower() == "gru":
        config["rnn_constructor"] = nn.GRU
    elif config["rnn_type"].lower() == "lstm" or True:
        config["rnn_constructor"] = nn.LSTM
    return config


def create_CRNN(config):
    check_inputs(config)
    # For apples-to-apples comparison, CNN outsize is OUT_SIZE + EMBEDDING_SIZE
    crnn = basic_CRNN(cnnOutSize=config['cnn_out_size'], nc=config['num_of_channels'],
                      alphabet_size=config['alphabet_size'], rnn_hidden_dim=config["rnn_dimension"],
                      recognizer_dropout=config["recognizer_dropout"], rnn_layers=config["rnn_layers"],
                      rnn_constructor=config["rnn_constructor"])
    return crnn


def create_seq2seq_recognizer(config):
    check_inputs(config)

    encoder = basic_CRNN(cnnOutSize=config['cnn_out_size'], nc=config['num_of_channels'],
                         alphabet_size=config['alphabet_size'], rnn_hidden_dim=config["alphabet_size"],
                         recognizer_dropout=config["recognizer_dropout"], rnn_layers=config["rnn_layers"],
                         rnn_constructor=config["rnn_constructor"])

    if config['cnn_load_path']:
        pretrained_state_dict = {name: val for name, val in torch.load(config['cnn_load_path']).items() if
                                 'cnn' in name}
        encoder.load_state_dict(pretrained_state_dict, strict=False)

    attention = Attention(embed_dim=config['alphabet_size'])

    decoder = Decoder(vocab_size=config['alphabet_size'], embed_dim=config['alphabet_size'],
                      context_dim=config['alphabet_size'], n_layers=5, hidden_dim=config['alphabet_size'],
                      char_freq=None)

    lm = DeepFusion(char_freq=None, vocab_size=config['alphabet_size'], n_layers=5)

    seq2seq = Seq2Seq(encoder, attention, decoder, lm, output_max_len=config['max_seq_len'],
                      vocab_size=config['alphabet_size'], sos_token=config['sos_idx'])

    if config['freeze_cnn']:
        for param in seq2seq.encoder.cnn.parameters():
            param.requires_grad = False

    return seq2seq


def create_CRNNClassifier(config, use_writer_classifier=True):
    # Don't use writer classifier
    check_inputs(config)
    crnn = CRNN_with_writer_classifier(rnn_input_dim=config["rnn_input_dimension"], nc=config['num_of_channels'],
                 alphabet_size=config['alphabet_size'], nh=config["rnn_dimension"],
                 number_of_writers=config["num_of_writers"], writer_rnn_output_size=config['writer_rnn_output_size'],
                 embedding_size=config["embedding_size"],
                 writer_dropout=config["writer_dropout"], recognizer_dropout=config["recognizer_dropout"],
                 writer_rnn_dimension=config["writer_rnn_dimension"],
                 mlp_layers=config["mlp_layers"], detach_embedding=config["detach_embedding"],
                 online_augmentation=config["online_augmentation"], use_writer_classifier=use_writer_classifier,
                 rnn_constructor=config["rnn_constructor"])
    return crnn


def create_2Stage(config):
    check_inputs(config)
    crnn = CRNN_2Stage(rnn_input_dim=config["rnn_input_dimension"], nc=config['num_of_channels'],
                       alphabet_size=config['alphabet_size'], rnn_hidden_dim=config["rnn_dimension"],
                       n_rnn=2, leakyRelu=False, recognizer_dropout=config["recognizer_dropout"],
                       online_augmentation=config["online_augmentation"], first_rnn_out_dim=128,
                       rnn_constructor=config["rnn_constructor"])
    return crnn


def create_Nudger(config):
    check_inputs(config)
    crnn = Nudger(rnn_input_dim=config["rnn_input_dimension"], nc=config['num_of_channels'],
                  rnn_hidden_dim=config["rnn_dimension"],
                  rnn_layers=config["nudger_rnn_layers"], leakyRelu=False, rnn_dropout=config["recognizer_dropout"],
                  rnn_constructor=config["rnn_constructor"])
    return crnn
