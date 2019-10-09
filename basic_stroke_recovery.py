from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from models.basic import CNN, BidirectionalRNN
from torch import nn
from models.stroke_recovery_loss import l1_loss
import torch
from models.CoordConv import CoordConv
from crnn import TrainerStrokeRecovery
from hwr_utils.hw_dataset import StrokeRecoveryDataset
from hwr_utils.stroke_recovery import *
from hwr_utils import utils
from torch.optim import lr_scheduler

# pip install git+https://github.com/jonbarron/robust_loss_pytorch

## TO DO:
## Read proposals
## Second committee member

# Loss functions:
    # Predicted points - nearest neighbor to rendered drawing
    # Use Barron loss
    # Calculate angle / trajectory to next point
    # Allow reverse strokes?
    # Small penalty for matching neighboring point
        # i.e. a smoothed loss, most weight is on current point matching
        # OR take the smallest possible loss of 3 most relevant points

# Attention
# Variable width -- add more instances etc.
# Add arrows to graphs!!

## Dataset:
    # Fix rendering issue
    # Keep time component constant - have possibility of "multiple blanks" etc.
    # Make dataset bigger

# Make a link from script directory to result directory
# LARGE without blur

class StrokeRecoveryModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = CNN(nc=1, first_conv_op=CoordConv)
        self.rnn = BidirectionalRNN(nIn=1024, nHidden=128, nOut=5, dropout=.5, num_layers=2, rnn_constructor=nn.LSTM)
        self.sigmoid =torch.nn.Sigmoid().to(device)

    def forward(self, input):
        cnn_output = self.cnn(input)
        rnn_output = self.rnn(cnn_output) # width, batch, alphabet
        rnn_output[:,:,2:] = self.sigmoid(rnn_output[:,:,2:])



        return rnn_output


def run_epoch(dataloader, report_freq=10):
    loss_list = []

    # for i in range(0, 16):
    #     line_imgs = torch.rand(batch, 1, 60, 60)
    #     targs = torch.rand(batch, 16, 5)
    for i, item in enumerate(dataloader):
        targs = item["gt"]
        line_imgs = item["line_imgs"].to(device)
        loss, *_ = trainer.train(line_imgs, targs)
        loss_list += [loss]
        if i % report_freq == 0 and i > 0:
            print(i, np.mean(loss_list[-report_freq:])/batch_size)
    return np.mean(loss_list)/batch_size

def test(dataloader):
    loss_list = []
    for i, item in enumerate(dataloader):
        #print(item)
        targs = item["gt"]
        line_imgs = item["line_imgs"].to(device)
        loss, preds = trainer.test(line_imgs, targs)
        preds_to_graph = preds.permute([0,2,1])
        loss_list += [loss]
    graph(preds_to_graph, item)
    return np.mean(loss_list)/batch_size

def graph(preds, batch):
    _epoch = str(epoch)
    (output / _epoch).mkdir(parents=True, exist_ok=True)
    for i, el in enumerate(batch["paths"]):
        img_path = el
        pred = utils.to_numpy(preds[i])
        pred[2:,:] = np.round(pred[2:,:])
        gts = utils.to_numpy(batch["gt"][i]).transpose()

        render_points_on_image(gts=pred, img_path=img_path, save_path=output / _epoch / f"temp{i}.png")
        render_points_on_image(gts=gts, img_path=img_path, save_path=output / _epoch / f"temp{i}_gt.png")
        if i > 8:
            break

torch.cuda.empty_cache()
device=torch.device("cuda")
#device=torch.device("cpu")

output = utils.increment_path(name="Run", base_path=Path("./results/stroke_recovery"))

output.mkdir(parents=True, exist_ok=True)
model = StrokeRecoveryModel().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=.95)
trainer = TrainerStrokeRecovery(model, optimizer, config=None, loss_criterion=l1_loss)
batch_size=32


train_dataset=StrokeRecoveryDataset(["online_coordinate_data/3_stroke_16/train_online_coords.json"],
                        img_height = 60,
                        num_of_channels = 1
                        )

train_dataloader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=6,
                              collate_fn=train_dataset.collate,
                              pin_memory=False)

test_dataset=StrokeRecoveryDataset(["online_coordinate_data/3_stroke_16/test_online_coords.json"],
                        img_height = 60,
                        num_of_channels = 1
                        )

test_dataloader = DataLoader(test_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=3,
                              collate_fn=train_dataset.collate,
                              pin_memory=False)

for i in range(0,400):
    epoch = i
    loss = run_epoch(train_dataloader)
    print(f"Epoch: {i}, Training Loss: {loss}")
    test_loss = test(test_dataloader)
    print(f"Epoch: {i}, Test Loss: {test_loss}")


