from hwr_utils import visualize
from torch.utils.data import DataLoader
from models.basic import CNN, BidirectionalRNN
from torch import nn
from stroke_recovery_loss import StrokeLoss
from models.CoordConv import CoordConv
from trainers import TrainerStrokeRecovery
from hwr_utils.stroke_dataset import StrokeRecoveryDataset
from hwr_utils.stroke_recovery import *
from hwr_utils import utils
from torch.optim import lr_scheduler
from timeit import default_timer as timer
import argparse
from hwr_utils.hwr_logger import logger
import losses

## Change CWD to the folder containing this script
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

## Variations:
# Relative position
# CoordConv - 0 center, X-as-rectanlge
# L1 loss, DTW
# Dataset size

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="./configs/stroke_config/baseline.yaml", help='Path to the config file.')
    #parser.add_argument('--name', type=str, default="", help='Optional - special name for this run')
    opts = parser.parse_args()
    return opts

class StrokeRecoveryModel(nn.Module):
    def __init__(self, vocab_size=5, device="cuda", cnn_type="default64", first_conv_op=CoordConv, first_conv_opts=None, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        if first_conv_op:
            first_conv_op = CoordConv
        self.cnn = CNN(nc=1, first_conv_op=first_conv_op, cnn_type=cnn_type, first_conv_opts=first_conv_opts)
        self.rnn = BidirectionalRNN(nIn=1024, nHidden=128, nOut=vocab_size, dropout=.5, num_layers=2, rnn_constructor=nn.LSTM)

    def forward(self, input):
        if self.training:
            return self._forward(input)
        else:
            with torch.no_grad():
                return self._forward(input)

    def _forward(self, input):
        cnn_output = self.cnn(input)
        rnn_output = self.rnn(cnn_output) # width, batch, alphabet
        # sigmoids are done in the loss
        return rnn_output

def run_epoch(dataloader, report_freq=500):
    loss_list = []

    # for i in range(0, 16):
    #     line_imgs = torch.rand(batch, 1, 60, 60)
    #     targs = torch.rand(batch, 16, 5)
    instances = 0
    start_time = timer()
    logger.info(("Epoch: ", epoch))

    for i, item in enumerate(dataloader):
        current_batch_size = item["line_imgs"].shape[0]
        instances += current_batch_size
        loss, preds, *_ = trainer.train(item, train=True)

        loss_list += [loss]
        config.stats["Actual_Loss_Function_train"].accumulate(loss)

        if config.counter.updates % report_freq == 0 and i > 0:
            utils.reset_all_stats(config, keyword="_train")
            logger.info(("update: ", config.counter.updates, "combined loss: ", config.stats["Actual_Loss_Function_train"].get_last()))

    end_time = timer()
    logger.info(("Epoch duration:", end_time-start_time))

    #preds_to_graph = preds.permute([0, 2, 1])
    preds_to_graph = [p.permute([1, 0]) for p in preds]
    save_folder = graph(item, config=config, preds=preds_to_graph, _type="train", x_relative_positions=config.x_relative_positions, epoch=epoch)
    utils.write_out(save_folder, "example_data", f"{str(item['gt_list'][0])}\nPREDS\n{str(preds_to_graph[0])}")

    config.scheduler.step()
    return np.sum(loss_list) / config.n_train_instances

def test(dataloader):
    for i, item in enumerate(dataloader):
        loss, preds, *_ = trainer.test(item)
        config.stats["Actual_Loss_Function_test"].accumulate(loss)
    preds_to_graph = [p.permute([1, 0]) for p in preds]
    save_folder = graph(item, config=config, preds=preds_to_graph, _type="test", x_relative_positions=config.x_relative_positions, epoch=epoch)
    utils.reset_all_stats(config, keyword="_test")

    return config.stats["Actual_Loss_Function_test"].get_last()

def graph(batch, config=None, preds=None, _type="test", save_folder=None, x_relative_positions=False, epoch="current"):
    if save_folder is None:
        _epoch = str(epoch)
        save_folder = (config.image_dir / _epoch / _type)
    else:
        save_folder = Path(save_folder)

    save_folder.mkdir(parents=True, exist_ok=True)

    def subgraph(coords, img_path, name, is_gt=True):
        if not is_gt:
            if coords is None:
                return
            coords = utils.to_numpy(coords[i])
            #print("before round", coords[2])
            coords[2:, :] = np.round(coords[2:, :]) # VOCAB SIZE, LENGTH
            #print("after round", coords[2])

            suffix=""

        else:
            suffix="_gt"
            coords = utils.to_numpy(coords).transpose() # LENGTH, VOCAB => VOCAB SIZE, LENGTH

        ## Undo relative positions for X for graphing
        ## In normal mode, the cumulative sum has already been taken
        # if config.pred_relativefy:
        if "stroke_number" in config.gt_format:
            idx = config.gt_format.index("stroke_number")
            coords[idx] = relativefy_numpy(coords[idx], reverse=False)

        render_points_on_image(gts=coords, img_path=img_path, save_path=save_folder / f"temp{i}_{name}{suffix}.png")

    # Loop through each item in batch
    for i, el in enumerate(batch["paths"]):
        img_path = el
        name=Path(batch["paths"][i]).stem
        if _type != "eval":
            subgraph(batch["gt_list"][i], img_path, name, is_gt=True)
        subgraph(preds, img_path, name, is_gt=False)
        if i > 8:
            break
    return save_folder

def main(config_path):
    global epoch, device, trainer, batch_size, output, loss_obj, config, LOGGER
    torch.cuda.empty_cache()
    utils.kill_gpu_hogs()
    os.chdir(ROOT_DIR)

    config = utils.load_config(config_path, hwr=False)
    test_size = config.test_size
    train_size = config.train_size
    batch_size = config.batch_size
    vocab_size = config.vocab_size

    config.device = "cuda" if torch.cuda.is_available() and config.gpu_if_available else "cpu"
    device = config.device

    #output = utils.increment_path(name="Run", base_path=Path("./results/stroke_recovery"))
    output = Path(config.results_dir)
    output.mkdir(parents=True, exist_ok=True)
    # folder = Path("online_coordinate_data/3_stroke_32_v2")
    # folder = Path("online_coordinate_data/3_stroke_vSmall")
    # folder = Path("online_coordinate_data/3_stroke_vFull")
    # folder = Path("online_coordinate_data/8_stroke_vFull")
    # folder = Path("online_coordinate_data/8_stroke_vSmall_16")
    folder = Path(config.dataset_folder)

    model = StrokeRecoveryModel(vocab_size=vocab_size,
                                device=device,
                                cnn_type=config.cnn_type,
                                first_conv_op=config.coordconv,
                                first_conv_opts=config.coordconv_opts).to(device)
    cnn = model.cnn # if set to a cnn object, then it will resize the GTs to be the same size as the CNN output
    logger.info(("Current dataset: ", folder))

    ## LOAD DATASET
    train_dataset=StrokeRecoveryDataset([folder / "train_online_coords.json"],
                            img_height = 60,
                            num_of_channels = 1,
                            root=config.data_root,
                            max_images_to_load = train_size,
                            gt_format=config.gt_format,
                            cnn=cnn
                            )

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=6,
                                  collate_fn=train_dataset.collate,
                                  pin_memory=False)

    config.n_train_instances = len(train_dataloader.dataset)

    test_dataset=StrokeRecoveryDataset([folder / "test_online_coords.json"],
                            img_height = 60,
                            num_of_channels = 1.,
                            root=config.data_root,
                            max_images_to_load=test_size,
                            gt_format=config.gt_format,
                            cnn=cnn
                            )

    test_dataloader = DataLoader(test_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=3,
                                  collate_fn=train_dataset.collate,
                                  pin_memory=False)

    n_test_points = 0
    for i in test_dataloader:
        n_test_points += sum(i["label_lengths"])
    config.n_test_instances = len(test_dataloader.dataset)
    config.n_test_points = int(n_test_points)

    # example = next(iter(test_dataloader)) # BATCH, WIDTH, VOCAB
    # vocab_size = example["gt"].shape[-1]

    ## Stats
    # Generic L1 loss
    config.L1 = losses.L1(loss_indices=slice(0, 2))

    if config.use_visdom:
        utils.start_visdom(port=config.visdom_port)
        visualize.initialize_visdom(config["full_specs"], config)
    utils.stat_prep_strokes(config)

    # Create loss object
    config.loss_obj = StrokeLoss(loss_stats=config.stats, counter=config.counter)
    config.loss_obj.build_losses(config.loss_fns)

    optimizer = torch.optim.Adam(model.parameters(), lr=.0005 * batch_size/32)
    config.scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=.95)
    trainer = TrainerStrokeRecovery(model, optimizer, config=config, loss_criterion=config.loss_obj)

    config.optimizer=optimizer
    config.trainer=trainer
    config.model = model

    for i in range(0,200):
        epoch = i+1
        #config.counter.epochs = epoch
        config.counter.update(epochs=1)
        loss = run_epoch(train_dataloader, report_freq=config.update_freq)
        logger.info(f"Epoch: {epoch}, Training Loss: {loss}")
        test_loss = test(test_dataloader)
        logger.info(f"Epoch: {epoch}, Test Loss: {test_loss}")
        if config.first_loss_epochs and epoch == config.first_loss_epochs:
            config.loss_obj.build_losses(config.loss_fns2)

        if epoch % config.save_freq == 0: # how often to save
            utils.save_model_stroke(config, bsf=False)

    ## Bezier curve
    # Have network predict whether it has reached the end of a stroke or not
    # If it has not reached the end of a stroke, the starting point = previous end point

if __name__=="__main__":
    opts = parse_args()
    main(config_path=opts.config)
    
    # TO DO:
        # logging
        # Get running on super computer - copy the data!