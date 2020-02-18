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
from models.stroke_model import StrokeRecoveryModel
from train_stroke_recovery import parse_args, graph
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
    config =     "/media/SuperComputerGroups/fslg_hwr/taylor_simple_hwr/RESULTS/ver1/RESUME.yaml"

    # Make these the same as whereever the file is being loaded from; make the log_dir and results dir be a subset
    # main_model_path, log_dir, full_specs, results_dir, load_path


    config = utils.load_config(config_path, hwr=False)

    # Free GPU memory if necessary
    if config.device == "cuda":
        utils.kill_gpu_hogs()

    batch_size = config.batch_size

    vocab_size = config.vocab_size

    device=torch.device(config.device)
    #device=torch.device("cpu")

    #output = utils.increment_path(name="Run", base_path=Path("./results/stroke_recovery"))
    output = Path(config.results_dir)
    output.mkdir(parents=True, exist_ok=True)
    folder = Path(config.dataset_folder)

    # OVERLOAD
    folder = Path("/media/data/GitHub/simple_hwr/data/prepare_IAM_Lines/lines/")
    #folder = Path(r"fish:////taylor@localhost:2222/media/data/GitHub/simple_hwr/data/prepare_IAM_Lines/")
    folder = Path("/media/data/GitHub/simple_hwr/data/prepare_IAM_Lines/words")
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
    config.load_path = "/media/SuperComputerGroups/fslg_hwr/taylor_simple_hwr/RESULTS/ver1/20200215_014143-normal/normal_model.pt"
    config.load_path = "/media/SuperComputerGroups/fslg_hwr/taylor_simple_hwr/RESULTS/ver2/20200217_033031-normal2/normal2_model.pt"

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
        graph(item, preds=preds_to_graph, _type="eval", epoch="current", config=config)

        if config.counter.updates % config.update_freq == 0 and i > 0:
            utils.reset_all_stats(config, keyword="_train")
            logger.info(("update: ", config.counter.updates, "combined loss: ", config.stats["Actual_Loss_Function_train"].get_last()))


if __name__=="__main__":
    opts = parse_args()
    main(config_path=opts.config)
