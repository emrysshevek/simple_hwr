import numpy as np
import os
import torch

# Input
def load_Coordinates(xml_path):
    Data_Label = []
    DATA_Label_K = []
    main_path = [f for f in np.sort(os.listdir(Data_Train_Path)) if f.endswith('online')]
    for s in main_path:
        sub = Data_Train_Path + s
        sub_path = [f for f in np.sort(os.listdir(sub)) if f.endswith('.txt')]
        for t in sub_path:
            txt_path = sub + '/' + t
            file_B = open(txt_path, 'r')
            data = file_B.read()
            x = data.split('\n')
            online_data = np.zeros([50, 2], np.int32)
            online_data_K = np.zeros([51, 2], np.int32)

            for y in range(50):
                z = x[y].split(" ")
                online_data[y] = [int(z[0]), int(z[1])]
                online_data_K[y + 1] = [int(z[0]), int(z[1])]
            online_data_K = online_data_K[:50, :]
            # print(online_data)
            file_B.close()
            Data_Label.append(online_data)
            DATA_Label_K.append(online_data_K)
    return np.array(Data_Label), np.array(DATA_Label_K)

def l1_loss(preds, targs):
    loss = torch.sum(torch.abs(preds-targs))
    return loss

def test_cnn():
    import torch
    from models.basic import BidirectionalRNN, CNN
    import torch.nn as nn

    cnn = CNN(nc=1)
    pool = nn.MaxPool2d(3, (4, 1), padding=1)
    batch = 7
    y = torch.rand(batch, 1, 60, 1024)
    a, b = cnn(y, intermediate_level=13)
    new = cnn.post_process(pool(b))

    final = torch.cat([a, new], dim=2)
    print(a.size())
    print(final.size())

    for x in range(1000,1100):
        y = torch.rand(2, 1, 60, x)
        a,b = cnn(y, intermediate_level=13)

        print(a.size(), b.size())
        new = cnn.post_process(pool(b)).size()
        print(new)
        assert new == a.size()

def test_loss():
    import torch
    batch = 3
    y = torch.rand(batch, 1, 60, 1024)
    x = y
    z = l1_loss(y,x)
    print(z)

if __name__=="__main__":
    test_loss()