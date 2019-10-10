from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from models.basic import CNN, BidirectionalRNN
from torch import nn
from models.stroke_recovery_loss import StrokeLoss
import torch
from models.CoordConv import CoordConv
from crnn import TrainerStrokeRecovery
from hwr_utils.hw_dataset import StrokeRecoveryDataset
from hwr_utils.stroke_recovery import *
from hwr_utils import utils
from torch.optim import lr_scheduler
from robust_loss_pytorch import AdaptiveLossFunction


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

## Augmentation:
    ## Warp
    ## Translate

## Increase the number of instances
## Find the Indian datasets
## Calculate a direction/angle
## Make variable width
## BEZIER CURVES
## Normalize time WRT ink distance

## New Data
# Devnagari
# http://mile.ee.iisc.ernet.in/hpl/DevnagariCharacters/Online/hpl-devnagari-iso-char-online-train-1.0.rar
# http://mile.ee.iisc.ernet.in/hpl/DevnagariCharacters/Online/hpl-devnagari-iso-char-online-test-1.0.rar
# http://mile.ee.iisc.ernet.in/hpl/DevnagariCharacters/Online/hpl-devnagari-iso-char-online-1.0.rar
#
# Telugu
# http://lipitk.sourceforge.net/datasets/teluguchardata.htm
# http://mile.ee.iisc.ernet.in/hpl/TelugCharacters/Online/hpl-telugu-iso-char-test-online-1.0.rar
# http://mile.ee.iisc.ernet.in/hpl/TelugCharacters/Online/hpl-telugu-iso-char-train-online-1.0.rar
# http://mile.ee.iisc.ernet.in/hpl/TelugCharacters/Online/hpl-telugu-iso-char-online-1.0.rar
#
# Tamil
# http://lipitk.sourceforge.net/datasets/tamilchardata.htm
#
# http://mile.ee.iisc.ernet.in/hpl/TamilCharacters/Online/hpl-tamil-iso-char-online-1.0.tar.gz
# http://mile.ee.iisc.ernet.in/hpl/TamilCharacters/Online/hpl-tamil-iso-char-train-online-1.0.tar.gz
# http://mile.ee.iisc.ernet.in/hpl/TamilCharacters/Online/hpl-tamil-iso-char-test-online-1.0.tar.gz
#
# http://mile.ee.iisc.ernet.in/hpl/TamilCharacters/Offline/hpl-tamil-iso-char-offline-1.0.tar.gz
# http://mile.ee.iisc.ernet.in/hpl/TamilCharacters/Offline/hpl-tamil-iso-char-train-offline-1.0.tar.gz
# http://mile.ee.iisc.ernet.in/hpl/TamilCharacters/Offline/hpl-tamil-iso-char-test-offline-1.0.tar.gz


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


def run_epoch(dataloader, report_freq=500):
    loss_list = []

    # for i in range(0, 16):
    #     line_imgs = torch.rand(batch, 1, 60, 60)
    #     targs = torch.rand(batch, 16, 5)
    instances = 0
    for i, item in enumerate(dataloader):
        targs = item["gt"] #.to(device)
        line_imgs = item["line_imgs"].to(device)
        current_batch_size = line_imgs.shape[0]
        instances += current_batch_size
        loss, preds, *_ = trainer.train(line_imgs, targs)
        loss_list += [loss]
        if i % report_freq == 0 and i > 0:
            print(i, np.mean(loss_list[-report_freq:])/batch_size)
    preds_to_graph = preds.permute([0, 2, 1])
    graph(preds_to_graph, item, _type="train")
    return np.mean(loss_list)/batch_size

def test(dataloader):
    loss_list = []
    for i, item in enumerate(dataloader):
        #print(item)
        targs = item["gt"]
        line_imgs = item["line_imgs"].to(device)
        loss, preds = trainer.test(line_imgs, targs)
        loss_list += [loss]
    preds_to_graph = preds.permute([0, 2, 1])
    graph(preds_to_graph, item)

    return np.mean(loss_list)/batch_size

def graph(preds, batch, _type="test"):
    _epoch = str(epoch)
    (output / _epoch / _type).mkdir(parents=True, exist_ok=True)
    for i, el in enumerate(batch["paths"]):
        img_path = el
        pred = utils.to_numpy(preds[i])
        pred[2:,:] = np.round(pred[2:,:])
        gts = utils.to_numpy(batch["gt"][i]).transpose()

        render_points_on_image(gts=pred, img_path=img_path, save_path=output / _epoch / _type / f"temp{i}.png")
        render_points_on_image(gts=gts, img_path=img_path, save_path=output / _epoch /  _type / f"temp{i}_gt.png")
        if i > 8:
            break

torch.cuda.empty_cache()
device=torch.device("cuda")
#device=torch.device("cpu")

output = utils.increment_path(name="Run", base_path=Path("./results/stroke_recovery"))

output.mkdir(parents=True, exist_ok=True)
model = StrokeRecoveryModel().to(device)


#loss_fnc = StrokeLoss(loss_type="robust").main_loss
loss_fnc = StrokeLoss(loss_type="None").main_loss

optimizer = torch.optim.Adam(model.parameters(), lr=.0005)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=.95)
trainer = TrainerStrokeRecovery(model, optimizer, config=None, loss_criterion=loss_fnc)
batch_size=32

folder = Path("online_coordinate_data/3_stroke_16_v2")
test_size = 2000
train_size = None
train_dataset=StrokeRecoveryDataset([folder / "train_online_coords.json"],
                        img_height = 60,
                        num_of_channels = 1,
                        max_images_to_load = train_size
                        )

train_dataloader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=6,
                              collate_fn=train_dataset.collate,
                              pin_memory=False)

test_dataset=StrokeRecoveryDataset([folder / "test_online_coords.json"],
                        img_height = 60,
                        num_of_channels = 1.,
                        max_images_to_load = test_size
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


