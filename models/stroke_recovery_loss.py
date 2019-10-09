import torch
#import robust_loss_pytorch

def l1_loss(preds, targs, label_lengths=16):
    """ Preds: [x], [y], [start stroke], [end stroke], [end of sequence]

    Args:
        preds: Will be in the form [picture_width, batch, 5
        targs:

    Returns:

    # Adapatively invert stroke targs if first instance is on the wrong end?? sounds sloooow

    """
    #print(preds.shape, targs.shape)
    #targs = targs.reshape(-1,16,5) # batch, width, 5
    return abs(preds-targs).sum()


def l1_loss(preds, targs, label_lengths=16):
    """ Preds: [x], [y], [start stroke], [end stroke], [end of sequence]

    Args:
        preds: Will be in the form [picture_width, batch, 5
        targs:

    Returns:

    # Adapatively invert stroke targs if first instance is on the wrong end?? sounds sloooow

    """
    # robust_loss_pytorch.adaptive.AdaptiveLossFunction(
    #     num_dims=1, float_dtype=np.float32, device='cpu')

    #print(preds.shape, targs.shape)
    #targs = targs.reshape(-1,16,5) # batch, width, 5
    return abs(preds-targs).sum()

if __name__ == "__main__":
    from models.basic import CNN, BidirectionalRNN
    from torch import nn
    batch = 3
    y = torch.rand(batch, 1, 60, 60)
    targs = torch.rand(batch, 16, 5)
    cnn = CNN(nc=1)
    rnn = BidirectionalRNN(nIn=1024, nHidden=128, nOut=5, dropout=.5, num_layers=2, rnn_constructor=nn.LSTM)
    cnn_output = cnn(y)
    rnn_output = rnn(cnn_output).permute(1, 0, 2)
    print(rnn_output.shape)
    loss = l1_loss(rnn_output, targs)
    print(loss)
