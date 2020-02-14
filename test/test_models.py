from models import start_points
from torch import nn
import torch
import numpy as np


def test_start_point_attn():
    spa = start_points.StartPointAttnModel()
    shape = (2,1,60,100)
    input_item = np.random.random(shape)
    print(input_item)
    spa(input_item)


if __name__=='__main__':
    test_start_point_attn()