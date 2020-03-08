from models.basic import CNN, BidirectionalRNN
from torch import nn
from models.CoordConv import CoordConv
from hwr_utils import visualize
from torch.utils.data import DataLoader
from loss_module.stroke_recovery_loss import StrokeLoss
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

pid = os.getpid()

#@debugger
def main(config_path):
    global epoch, device, trainer, batch_size, output, loss_obj, x_relative_positions, config, LOGGER
    torch.cuda.empty_cache()

    PROJ_ROOT = "/media/data/GitHub/simple_hwr/"
    config_path = "/media/data/GitHub/simple_hwr/~RESULTS/20191213_155358-baseline-GOOD_long/TEST.yaml"
    config_path = "/media/SuperComputerGroups/fslg_hwr/taylor_simple_hwr/RESULTS/ver1/RESUME.yaml"
    config_path = PROJ_ROOT + "RESULTS/OFFLINE_PREDS/good/normal_preload.yaml"

    load_path_override= "/media/SuperComputerGroups/fslg_hwr/taylor_simple_hwr/results/stroke_config/GOOD/baseline_model.pt"
    load_path_override = "/media/data/GitHub/simple_hwr/~RESULTS/20191213_155358-baseline-GOOD_long"
    load_path_override= "/media/SuperComputerGroups/fslg_hwr/taylor_simple_hwr/RESULTS/ver1/20200215_014143-normal/normal_model.pt"
    load_path_override = "/media/SuperComputerGroups/fslg_hwr/taylor_simple_hwr/RESULTS/ver2/20200217_033031-normal2/normal2_model.pt"
    load_path_override = PROJ_ROOT + "RESULTS/pretrained/new_best/good.pt"

    load_path_override = PROJ_ROOT + "RESULTS/OFFLINE_PREDS/all_data.npy"
    _load_path_override = Path(load_path_override)

    OUTPUT = PROJ_ROOT / Path("RESULTS/OFFLINE_PREDS/") / _load_path_override.stem
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # Make these the same as whereever the file is being loaded from; make the log_dir and results dir be a subset
    # main_model_path, log_dir, full_specs, results_dir, load_path


    config = utils.load_config(config_path, hwr=False, results_dir_override=OUTPUT.as_posix())

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
    folder = PROJ_ROOT / Path("data/prepare_IAM_Lines/lines/")
    gt_path = PROJ_ROOT / Path("data/prepare_IAM_Lines/gts/lines/txt")
    #folder = Path(r"fish:////taylor@localhost:2222/media/data/GitHub/simple_hwr/data/prepare_IAM_Lines/")
    #folder = Path("/media/data/GitHub/simple_hwr/data/prepare_IAM_Lines/words")
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
    config.load_path = load_path_override if ("load_path_override" in locals()) else config.load_path

    config.sigmoid_indices = TrainerStrokeRecovery.get_indices(config.pred_opts, "sigmoid")

    # Load the GTs
    GT_DATA = load_all_gts(gt_path)
    print("Number of images: {}".format(len(eval_loader.dataset)))
    print("Number of GTs: {}".format(len(GT_DATA)))

    ## LOAD THE WEIGHTS
    utils.load_model_strokes(config) # should be load_model_strokes??????
    model = model.to(device)
    model.eval()
    eval_only(eval_loader, model)
    globals().update(locals())

def post_process(pred,gt):
    # DON'T POST PROCESS FOR NOW
    #return move_bad_points(reference=gt, moving_component=pred, reference_is_image=True)
    return pred

def eval_only(dataloader, model):
    final_out = []
    for i, item in enumerate(dataloader):
        preds = TrainerStrokeRecovery.eval(item["line_imgs"], model,
                                           label_lengths=item["label_lengths"],
                                           relative_indices=config.pred_relativefy,
                                           sigmoid_activations=config.sigmoid_indices)

        # Pred comes out of eval WIDTH x VOCAB
        preds_to_graph = [post_process(p, item["line_imgs"][i]).permute([1, 0]) for i,p in enumerate(preds)]

        # Get GTs, save to file
        if i<10:
            # Save a sample
            save_folder = graph(item, preds=preds_to_graph, _type="eval", epoch="current", config=config)
            output_path = (save_folder / "data")
            output_path.mkdir(exist_ok=True, parents=True)

        names = [Path(p).stem.lower() for p in item["paths"]]
        output = []
        for ii, name in enumerate(names):
            if name in GT_DATA:
                output.append({"stroke": preds[ii].detach().numpy(),
                               "text":GT_DATA[name],
                               "id": name
                               })
            else:
                print(f"{name} not found")
        utils.pickle_it(output, output_path / f"{i}.pickle")
        np.save(output_path / f"{i}.npy", output)
        final_out += output
    utils.pickle_it(final_out, output_path / f"all_data.pickle")
    np.save(output_path / f"all_data.npy", final_out)

def load_all_gts(gt_path):
    global GT_DATA
    from hwr_utils.hw_dataset import HwDataset
    data = HwDataset.load_data(data_paths=gt_path.glob("*.json"))
    #{'gt': 'He rose from his breakfast-nook bench', 'image_path': 'prepare_IAM_Lines/lines/m01/m01-049/m01-049-00.png',
    GT_DATA = {}
    for i in data:
        key = Path(i["image_path"]).stem.lower()
        assert not key in GT_DATA
        GT_DATA[key] = i["gt"]
    return GT_DATA

if __name__=="__main__":
    opts = parse_args()
    main(config_path=opts.config)
    # gt_path = Path("./data/prepare_IAM_Lines/gts/lines/txt")
    # load_all_gts(gt_path)
