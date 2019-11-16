from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from models.basic import CNN, BidirectionalRNN
from torch import nn
from models.stroke_recovery_loss import StrokeLoss
import torch
from models.CoordConv import CoordConv
from crnn import TrainerStrokeRecovery
from hwr_utils.stroke_dataset import StrokeRecoveryDataset
from hwr_utils.stroke_recovery import *
from hwr_utils import utils
from torch.optim import lr_scheduler
from timeit import default_timer as timer

torch.cuda.empty_cache()

## Variations:
# Relative position
# CoordConv - 0 center, X-as-rectanlge
# L1 loss, DTW
# Dataset size

### PREDS -- truncate right away if that's what we're doing!


class StrokeRecoveryModel(nn.Module):
    def __init__(self, vocab_size=5, device="cuda"):
        super().__init__()

        self.cnn = CNN(nc=1, first_conv_op=CoordConv, cnn_type="default64")
        self.rnn = BidirectionalRNN(nIn=1024, nHidden=128, nOut=vocab_size, dropout=.5, num_layers=2, rnn_constructor=nn.LSTM)
        self.sigmoid =torch.nn.Sigmoid().to(device)

    def forward(self, input):
        cnn_output = self.cnn(input)
        rnn_output = self.rnn(cnn_output) # width, batch, alphabet
        rnn_output[:,:,2:] = self.sigmoid(rnn_output[:,:,2:]) # force SOS (start of stroke) and EOS (end of stroke) to be probabilistic
        return rnn_output

def run_epoch(dataloader, report_freq=500):
    loss_list = []

    # for i in range(0, 16):
    #     line_imgs = torch.rand(batch, 1, 60, 60)
    #     targs = torch.rand(batch, 16, 5)
    instances = 0
    start_time = timer()
    print(epoch)
    loss_fn = loss_obj.dtw if epoch > 5 else loss_obj.variable_l1
    loss_fn = loss_obj.variable_l1
    for i, item in enumerate(dataloader):
        line_imgs = item["line_imgs"].to(device)
        current_batch_size = line_imgs.shape[0]
        instances += current_batch_size

        # Feed the right targets
        # if loss_fn==loss_obj.dtw:
        #     loss, preds, *_ = trainer.train(loss_fn, line_imgs, item["gt_list"])
        # elif loss_fn==loss_obj.variable_l1:
        #     loss, preds, *_ = trainer.train(loss_fn, line_imgs, item)
        # else:
        #     raise Exception("Unknown loss")
        loss, preds, *_ = trainer.train(loss_fn, item)

        loss_list += [loss]
        if i % report_freq == 0 and i > 0:
            print(i, np.mean(loss_list[-report_freq:])/batch_size)

    end_time = timer()
    print("Epoch duration:", end_time-start_time)

    #preds_to_graph = preds.permute([0, 2, 1])
    preds_to_graph = [p.permute([1, 0]) for p in preds]
    graph(item, preds=preds_to_graph, _type="train", x_relative_positions=x_relative_positions)
    return np.mean(loss_list)/batch_size

def test(dataloader):
    loss_fn = loss_obj.variable_l1
    loss_list = []
    for i, item in enumerate(dataloader):
        #print(item)
        targs = item["gt_list"]
        line_imgs = item["line_imgs"].to(device)

        # if loss_fn==loss_obj.dtw:
        #     loss, preds, *_ = trainer.train(loss_fn, line_imgs, item["gt_list"])
        # elif loss_fn==loss_obj.variable_l1:
        #     loss, preds, *_ = trainer.train(loss_fn, line_imgs, item)
        # else:
        #     raise Exception("Unknown loss")
        loss, preds, *_ = trainer.train(loss_fn, item)

        loss_list += [loss]
    preds_to_graph = [p.permute([1, 0]) for p in preds]
    graph(item, preds=preds_to_graph, _type="test", x_relative_positions=x_relative_positions)
    return np.mean(loss_list)/batch_size

def graph(batch, preds=None,_type="test", save_folder=None, x_relative_positions=False):
    if save_folder is None:
        _epoch = str(epoch)
        save_folder = (output / _epoch / _type)
    else:
        save_folder = Path(save_folder)

    save_folder.mkdir(parents=True, exist_ok=True)

    def subgraph(coords, img_path, name, is_gt=True):

        if not is_gt:
            if coords is None:
                return
            coords = utils.to_numpy(coords[i])
            coords[2:, :] = np.round(coords[2:, :]) # VOCAB SIZE, LENGTH
            suffix=""
        else:
            suffix="_gt"
            coords = utils.to_numpy(coords).transpose() # LENGTH, VOCAB => VOCAB SIZE, LENGTH

        ## Undo relative positions for X for graphing
        if x_relative_positions:
            coords[0] = relativefy(coords[0], reverse=True)

        render_points_on_image(gts=coords, img_path=img_path, save_path=save_folder / f"temp{i}_{name}{suffix}.png")

    # Loop through each item in batch
    for i, el in enumerate(batch["paths"]):
        img_path = el
        name=Path(batch["paths"][i]).stem
        subgraph(batch["gt_list"][i], img_path, name, is_gt=True)
        subgraph(preds, img_path, name, is_gt=False)
        if i > 8:
            break

def main():
    global epoch, device, trainer, batch_size, output, loss_obj, x_relative_positions
    torch.cuda.empty_cache()
    device=torch.device("cuda")
    #device=torch.device("cpu")

    output = utils.increment_path(name="Run", base_path=Path("./results/stroke_recovery"))

    output.mkdir(parents=True, exist_ok=True)
    loss_obj = StrokeLoss()

    folder = Path("online_coordinate_data/3_stroke_32_v2")
    folder = Path("online_coordinate_data/3_stroke_vSmall")
    #folder = Path("online_coordinate_data/3_stroke_vFull")
    folder = Path("online_coordinate_data/8_stroke_vFull")
    #folder = Path("online_coordinate_data/8_stroke_vSmall_16")

    test_size = 2000
    train_size = None
    batch_size=32
    x_relative_positions=False
    vocab_size = 4

    model = StrokeRecoveryModel(vocab_size=vocab_size, device=device).to(device)
    cnn = model.cnn # if set to a cnn object, then it will resize the GTs to be the same size as the CNN output
    print("Current dataset: ", folder)
    train_dataset=StrokeRecoveryDataset([folder / "train_online_coords.json"],
                            img_height = 60,
                            num_of_channels = 1,
                            max_images_to_load = train_size,
                            x_relative_positions=x_relative_positions,
                            cnn=cnn
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
                            max_images_to_load = test_size,
                            x_relative_positions=x_relative_positions,
                            cnn=cnn
                            )

    test_dataloader = DataLoader(test_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=3,
                                  collate_fn=train_dataset.collate,
                                  pin_memory=False)

    # example = next(iter(test_dataloader)) # BATCH, WIDTH, VOCAB
    # vocab_size = example["gt"].shape[-1]
    optimizer = torch.optim.Adam(model.parameters(), lr=.0005 * batch_size/32)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=.95)
    trainer = TrainerStrokeRecovery(model, optimizer, config=None, loss_criterion=loss_obj)
    globals().update(locals())
    for i in range(0,200):
        epoch = i+1
        loss = run_epoch(train_dataloader)
        print(f"Epoch: {epoch}, Training Loss: {loss}")
        test_loss = test(test_dataloader)
        print(f"Epoch: {epoch}, Test Loss: {test_loss}")


    ## Bezier curve
    # Have network predict whether it has reached the end of a stroke or not
    # If it has not reached the end of a stroke, the starting point = previous end point

if __name__=="__main__":
    main()