from hwr_utils import visualize
from torch.utils.data import DataLoader
from models.basic import CNN, BidirectionalRNN
from torch import nn
from loss_module.stroke_recovery_loss import StrokeLoss
from models.CoordConv import CoordConv
from trainers import TrainerStrokeRecovery
from hwr_utils.stroke_dataset import BasicDataset
from hwr_utils.stroke_recovery import *
from hwr_utils import utils
from torch.optim import lr_scheduler
from train_stroke_recovery import StrokeRecoveryModel, parse_args, graph
from hwr_utils.hwr_logger import logger
from pathlib import Path

import os
from subprocess import Popen


utils.kill_gpu_hogs()
pid = os.getpid()
command=f"pgrep -fl python | awk '!/{pid}/{{print $1}}' | xargs kill"
result = Popen(command, shell=True)

#@debugger
def main(config_path):
    global epoch, device, trainer, batch_size, output, loss_obj, x_relative_positions, config, LOGGER
    torch.cuda.empty_cache()
    config = "/media/data/GitHub/simple_hwr/~RESULTS/20191213_155358-baseline-GOOD_long/TEST.yaml"

    # Make these the same as whereever the file is being loaded from; make the log_dir and results dir be a subset
    # main_model_path, log_dir, full_specs, results_dir, load_path


    config = utils.load_config(config_path, hwr=False)
    batch_size = config.batch_size
    x_relative_positions = config.x_relative_positions

    if x_relative_positions == "both":
        raise Exception("Not implemented")
    vocab_size = config.vocab_size

    device=torch.device("cuda")
    #device=torch.device("cpu")

    #output = utils.increment_path(name="Run", base_path=Path("./results/stroke_recovery"))
    output = Path(config.results_dir)
    output.mkdir(parents=True, exist_ok=True)
    folder = Path(config.dataset_folder)

    # OVERLOAD
    folder = Path("/media/data/GitHub/simple_hwr/data/prepare_IAM_Lines/lines/")
    model = StrokeRecoveryModel(vocab_size=vocab_size, device=device, cnn_type=config.cnn_type, first_conv_op=config.coordconv, first_conv_opts=config.coordconv_opts).to(device)

    ## Loader
    logger.info(("Current dataset: ", folder))
    # Dataset - just expecting a folder
    eval_dataset=BasicDataset(root=folder, cnn=model.cnn)

    eval_loader=DataLoader(eval_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=6,
                                  collate_fn=eval_dataset.collate, # this should be set to collate_stroke_eval
                                  pin_memory=False)
    config.n_train_instances = None
    config.n_test_instances = len(eval_loader.dataset)
    config.n_test_points = None

    ## Stats
    if config.use_visdom:
        visualize.initialize_visdom(config["full_specs"], config)
    utils.stat_prep_strokes(config)

    # Create loss object
    config.loss_obj = StrokeLoss(loss_names=config.loss_fns, loss_stats=config.stats, counter=config.counter)
    optimizer = torch.optim.Adam(model.parameters(), lr=.0005 * batch_size/32)
    config.scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=.95)
    trainer = TrainerStrokeRecovery(model, optimizer, config=config, loss_criterion=config.loss_obj)

    config.model = model
    config.load_path = "/media/SuperComputerGroups/fslg_hwr/taylor_simple_hwr/results/stroke_config/GOOD/baseline_model.pt"
    config.load_path = "/media/data/GitHub/simple_hwr/~RESULTS/20191213_155358-baseline-GOOD_long"

    ## LOAD THE WEIGHTS
    utils.load_model(config) # should be load_model_strokes??????
    model = model.to(device)

    eval_only(eval_loader, model)

def eval_only(dataloader, model):
    for i, item in enumerate(dataloader):
        preds = TrainerStrokeRecovery.eval(item["line_imgs"], model,
                                           label_lengths=item["label_lengths"],
                                           relative_indices=config.pred_relativefy)

        preds_to_graph = [p.permute([1, 0]) for p in preds]
        graph(item, preds=preds_to_graph, _type="eval", x_relative_positions=x_relative_positions, epoch="current", config=config)


if __name__=="__main__":
    opts = parse_args()
    main(config_path=opts.config)


class StrokeRecoveryModel(nn.Module):
    def __init__(self, vocab_size=5, device="cuda", first_conv_op=CoordConv, first_conv_opts=None):
        super().__init__()
        if first_conv_op:
            first_conv_op = CoordConv
        self.cnn = CNN(nc=1, first_conv_op=first_conv_op, cnn_type="default64", first_conv_opts=first_conv_opts)
        self.attn = nn.MultiheadAttention(embed_dim=1024, num_heads=4)
        # self.rnn = nn.LSTM(2*1024, hidden_size=1024, num_layers=2)
        self.rnn = BidirectionalRNN(1024, 128, vocab_size)
        self.linear_out = nn.Linear(1024, vocab_size)
        self.sigmoid = torch.nn.Sigmoid().to(device)

    def get_cnn(self):
        return self.cnn

    def forward(self, input):
        cnn_output = self.cnn(input)
        attn_output, attn_weights = self.attn(cnn_output, cnn_output, cnn_output)
        # rnn_output = self.rnn(torch.cat([cnn_output, attn_output], dim=-1))
        rnn_output = self.rnn(attn_output)
        rnn_output[:,:,2:] = self.sigmoid(rnn_output[:,:,2:]) # force SOS (start of stroke) and EOS (end of stroke) to be probabilistic
        return rnn_output