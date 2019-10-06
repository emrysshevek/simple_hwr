import torch
import torch.nn as nn

'''
An alternative implementation for PyTorch with auto-infering the x-y dimensions.
'''
class AddCoords(nn.Module):

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConv(nn.Module):
    def __init__(self, in_channels, out_channels, with_r=False, verbose=False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        self.verbose = verbose
        in_size = in_channels+2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

    def forward(self, x):
        ret = self.addcoords(x)
        ## See the coordConvs:
        if self.verbose:
            print(ret[0,-1])
            print(ret[0,-2])
        ret = self.conv(ret)
        return ret

def test_cnn():
    import torch
    from models.basic import BidirectionalRNN, CNN, CCNN
    import torch.nn as nn

    cnn = CCNN(nc=1, conv_op=CoordConv, verbose=False)
    # cnn = CCNN(nc=1, conv_op=nn.Conv2d)
    batch = 7
    y = torch.rand(batch, 1, 60, 1024)
    a = cnn(y)
    globals().update(locals())


if __name__=="__main__":
    test_cnn()