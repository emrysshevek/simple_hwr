import torch
import torch.nn as nn

'''
An alternative implementation for PyTorch with auto-infering the x-y dimensions.
'''

class AddCoords(nn.Module):

    def __init__(self, with_r=False, zero_center=True, rectangle_x=False, both_x=False):
        """ Include a rectangle and non-rectangle x

        Args:
            with_r:
            zero_center:
            rectangle_x:
            both_x:
        """
        super().__init__()
        self.with_r = with_r
        self.rectangle_x = rectangle_x
        self.zero_center = zero_center
        self.both = both_x

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        # Rescale from 0 to 1
        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        # Rescale from -1 to 1
        if self.zero_center:
            yy_channel = yy_channel * 2 - 1
            xx_channel = xx_channel * 2 - 1

        if self.both:
            xx_rec_channel = xx_channel * x_dim / y_dim

            xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
            yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
            xx_rec_channel = xx_rec_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

            ret = torch.cat([
                input_tensor,
                xx_channel.type_as(input_tensor),
                xx_rec_channel.type_as(input_tensor),
                yy_channel.type_as(input_tensor)], dim=1)
        else:
            if self.rectangle_x:
                xx_channel *= x_dim / y_dim

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
    def __init__(self, in_channels, out_channels, with_r=False, verbose=False, rectangle_x=False, both_x=False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r, rectangle_x=rectangle_x, both_x=both_x)
        self.verbose = verbose
        in_size = in_channels+2
        if with_r or both_x:
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
    from models.basic import BidirectionalRNN, CNN
    import torch.nn as nn

    cnn = CCNN(nc=1, conv_op=CoordConv, verbose=False)
    # cnn = CCNN(nc=1, conv_op=nn.Conv2d)
    batch = 7
    y = torch.rand(batch, 1, 60, 1024)
    a = cnn(y)
    globals().update(locals())


if __name__=="__main__":
    test_cnn()