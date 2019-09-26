from __future__ import print_function
from builtins import range
import faulthandler
from utils import character_set
from data import hw_dataset
from data.hw_dataset import HwDataset
from models import crnn
from models import model_builders as builder
from models import trainers
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import tensor
from torch import nn

### TO DO:
# Add ONLINE flag to regular CRNN
# Download updated JSONs/Processing


# EMAIL SUPERCOMPUTER?
# "right" way to make an embedding
# CycleGAN - threshold
# Deepwriting - clean up generated images?
# Dropout schedule

from torch.nn import CrossEntropyLoss
import traceback

import visualize
# matplotlib.use('TkAgg')
from utils.utils import *
from torch.optim import lr_scheduler

## Notes on usage
# conda activate hw2
# python -m visdom.server -p 8080


faulthandler.enable()


def test(model, dataloader, idx_to_char, device, configig, with_analysis=False):
    sum_loss = 0.0
    steps = 0.0
    model.eval()
    for i, x in enumerate(dataloader):
        line_imgs = x['line_imgs'].to(device)
        gts = x['gt']  # actual string ground truth
        online = x['online'].view(1, -1, 1).to(device) if config["online_augmentation"] else None
        loss, err, pred_strs = config["trainer"].test(line_imgs, online, gts)

        # Only do one test
        if config["TESTING"]:
            break

        if i == len(dataloader) - 1:
            LOGGER.info(f'\tPrediction:   [{pred_strs[0]}]')
            LOGGER.info(f'\tGround Truth: [{gts[0]}]')

    accumulate_stats(config)
    test_cer = config["stats"][config["designated_test_cer"]].y[-1]  # most recent test CER

    LOGGER.debug(config["stats"])
    return test_cer


def plot_images(line_imgs, name, text_str):
    # Save images
    f, axarr = plt.subplots(len(line_imgs), 1)
    if len(line_imgs) > 1:
        for j, img in enumerate(line_imgs):
            axarr[j].set_xlabel(f"{text_str[j]}")
            axarr[j].imshow(img.squeeze().detach().cpu().numpy(), cmap='gray')
    else:
        axarr.imshow(line_imgs.squeeze().detach().cpu().numpy(), cmap='gray')

    # plt.show()
    path = os.path.join(config["image_dir"], '{}.png'.format(name))
    plt.savefig(path, dpi=300)
    plt.close('all')


def improver(model, dataloader, ctc_criterion, optimizer, dtype, config):
    model.train()  # make sure gradients are tracked
    lr = .01
    model.my_eval()  # set dropout to 0

    for i, x in enumerate(dataloader):
        LOGGER.debug("Improving Iteration: {}".format(i))
        line_imgs = tensor(x['line_imgs'].type(dtype)).clone().detach().requires_grad_(True)
        params = [torch.nn.Parameter(line_imgs)]
        config["trainer"].optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)

        labels = tensor(x['labels'], requires_grad=False)  # numeric indices version of ground truth
        label_lengths = tensor(x['label_lengths'], requires_grad=False)
        gt = x['gt']  # actual string ground truth

        # Add online/offline binary flag
        online = tensor(x['online'].type(dtype), requires_grad=False).view(1, -1, 1) if config[
            "online_augmentation"] else None

        loss, initial_err, first_pred_str = config["trainer"].train(params[0], online, labels, label_lengths, gt,
                                                                    step=config["global_step"])
        # Nudge it X times
        for ii in range(50):
            loss, final_err, final_pred_str = config["trainer"].train(params[0], online, labels, label_lengths, gt,
                                                                      step=config["global_step"])
            # print(torch.abs(x['line_imgs']-params[0]).sum())
            accumulate_stats(config)
            training_cer = config["stats"][config["designated_training_cer"]].y[-1]  # most recent training CER
            LOGGER.info(f"{training_cer}")

        plot_images(params[0], i, gt)
        plot_images(x['line_imgs'], f"{i}_original", first_pred_str)

    return training_cer


def run_epoch(model, dataloader, ctc_criterion, optimizer, dtype, config):
    model.train()
    config["stats"]["epochs"] += [config["current_epoch"]]
    plot_freq = config["plot_freq"]

    for i, x in enumerate(dataloader):
        LOGGER.debug(f"Training Iteration: {i}")
        line_imgs = Variable(x['line_imgs'].type(dtype), requires_grad=False)
        labels = Variable(x['labels'], requires_grad=False)  # numeric indices version of ground truth
        label_lengths = Variable(x['label_lengths'], requires_grad=False)
        gts = x['gt']  # actual string ground truth
        config["global_step"] += 1
        config["global_instances_counter"] += line_imgs.shape[0]
        config["stats"]["instances"] += [config["global_instances_counter"]]

        # Add online/offline binary flag
        online = Variable(x['online'].type(dtype), requires_grad=False).view(1, -1, 1) if config[
            "online_augmentation"] else None

        loss, err, pred_strs = config["trainer"].train(line_imgs, online, labels, label_lengths, gts, step=config["global_step"])

        # Update visdom every 50 instances
        if config["global_step"] % plot_freq == 0 and config["global_step"] > 0:
            config["stats"]["updates"] += [config["global_step"]]
            config["stats"]["epoch_decimal"] += [
                config["current_epoch"] + i * config["batch_size"] * 1.0 / config['n_train_instances']]
            LOGGER.info(f"updates: {config['global_step']}")
            accumulate_stats(config)
            visualize.plot_all(config)

            LOGGER.info(f'\tPrediction:   [{pred_strs[0]}]')
            LOGGER.info(f'\tGround Truth: [{gts[0]}]')

        if config["TESTING"]:
            config["stats"]["updates"] += [config["global_step"]]
            config["stats"]["epoch_decimal"] += [
                config["current_epoch"] + i * config["batch_size"] / config['n_train_instances']]
            LOGGER.info(f"updates: {config['global_step']}")
            accumulate_stats(config)
            break

    training_cer = config["stats"][config["designated_training_cer"]].y[-1]  # most recent training CER
    LOGGER.debug(config["stats"])
    return training_cer


def make_dataloaders(config, device="cpu"):
    if config['seq2seq']:
        collate_fct = lambda x: hw_dataset.seq2seq_collate(x, config['sos_idx'], config['eos_idx'], config['pad_idx'], config['max_seq_len'], device=device)
    else:
        collate_fct = lambda x: hw_dataset.collate(x, device=device)

    if config['SMALL_TRAINING']:
        shuffle = False
        train_size = config['SMALL_TRAINING'] * config['batch_size']
        test_size = config['batch_size']
    else:
        shuffle = True
        train_size = -1
        test_size = -1

    train_dataset = HwDataset(config["training_jsons"], config["char_to_idx"], img_height=config["input_height"],
                              num_of_channels=config["num_of_channels"], root=config["training_root"],
                              warp=config["training_warp"], writer_id_paths=config["writer_id_pickles"], size=train_size)

    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=shuffle,
                                  num_workers=0, collate_fn=collate_fct, pin_memory=device == "cpu", drop_last=True)

    test_dataset = HwDataset(config["testing_jsons"], config["char_to_idx"], img_height=config["input_height"],
                             num_of_channels=config["num_of_channels"], root=config["testing_root"],
                             warp=config["testing_warp"], size=test_size)

    test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=shuffle,
                                 num_workers=0, collate_fn=collate_fct, drop_last=True)

    return train_dataloader, test_dataloader, train_dataset, test_dataset


def load_data(config):
    # Load characters and prep datasets
    config["char_to_idx"], config["idx_to_char"], config["char_freq"], config['sos_idx'], config['eos_idx'], \
        config['pad_idx'] = character_set.make_char_set(config['training_jsons'], root=config["training_root"])

    train_dataloader, test_dataloader, train_dataset, test_dataset = make_dataloaders(config=config)

    config['alphabet_size'] = len(config["idx_to_char"])   # alphabet size to be recognized
    config['num_of_writers'] = train_dataset.classes_count + 1

    config['n_train_instances'] = len(train_dataloader.dataset)
    log_print("Number of training instances:", config['n_train_instances'])
    log_print("Number of test instances:", len(test_dataloader.dataset), '\n')
    return train_dataloader, test_dataloader, train_dataset, test_dataset


def check_gpu(config):
    # GPU stuff
    use_gpu = torch.cuda.is_available() and not config["TESTING"] and config['use_gpu']
    device = torch.device("cuda" if use_gpu else "cpu")
    dtype = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
    if use_gpu:
        log_print("Using GPU")
    elif not torch.cuda.is_available():
        log_print("No GPU found")
    elif config["TESTING"]:
        log_print("Testing, not using GPU")
    return device, dtype


def main():
    global config, LOGGER
    opts = parse_args()
    config = load_config(opts.config)
    LOGGER = config["logger"]
    config["global_step"] = 0
    config["global_instances_counter"] = 0

    device, dtype = check_gpu(config)

    # Use small batch size when using CPU/testing
    if config["TESTING"]:
        config["batch_size"] = 1

    # Prep data loaders
    train_dataloader, test_dataloader, train_dataset, test_dataset = load_data(config)
    config['char_freq'] = torch.tensor(config['char_freq']).to(device).to(torch.float32)

    # Prep optimizer
    if not config['seq2seq']:
        ctc = torch.nn.CTCLoss()
        log_softmax = torch.nn.LogSoftmax(dim=2).to(device)
        criterion = lambda x, y, z, t: ctc(log_softmax(x), y, z, t)
    elif config['label_smoothing']:
        criterion = crnn.LabelSmoothing(len(config['idx_to_char']), config['pad_idx'], config['smoothing'])
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=config['pad_idx'])

    # Decoder
    config["calc_cer_training"] = calculate_cer
    use_beam = config["decoder_type"] == "beam"
    config["decoder"] = Decoder(idx_to_char=config["idx_to_char"], beam=use_beam)
    # print(config['rnn_constructor'])
    # Create classifier
    if config["style_encoder"] == "basic_encoder":
        hw = builder.create_CRNNClassifier(config)
    elif config["style_encoder"] == "fake_encoder":
        hw = builder.create_CRNNClassifier(config)
    elif config["style_encoder"] == "2Stage":
        hw = builder.create_2Stage(config)
        config["embedding_size"] = 0
    elif config["style_encoder"] == "2StageNudger":
        hw = builder.create_CRNN(config)
        config["nudger"] = builder.create_Nudger(config).to(device)
        config["embedding_size"] = 0
        config["nudger_optimizer"] = torch.optim.Adam(config["nudger"].parameters(), lr=config['learning_rate'])
    elif config['seq2seq']:
        hw = builder.create_seq2seq_recognizer(config)
        # pretrained_state_dict = torch.load(config['cnn_load_path'])
        # hw.encoder = pretrained_state_dict['model']
    else:  # basic HWR
        config["embedding_size"] = 0
        hw = builder.create_CRNN(config)

    LOGGER.info(hw)

    hw.to(device)

    # Setup defaults
    config["starting_epoch"] = 1
    config["model"] = hw
    config['lowest_loss'] = float('inf')
    config["train_cer"] = []
    config["test_cer"] = []

    # Launch visdom
    if config["use_visdom"]:
        visualize.initialize_visdom(config["full_specs"], config)

    # Stat prep - must be after visdom
    stat_prep(config)

    # Create optimizer
    optimizer = torch.optim.Adam(hw.parameters(), lr=config['learning_rate'])
    config["optimizer"] = optimizer

    scheduler = lr_scheduler.StepLR(optimizer, step_size=config["scheduler_step"], gamma=config["scheduler_gamma"])
    config["scheduler"] = scheduler

    ## LOAD FROM OLD MODEL
    if config["load_path"]:
        load_model(config)
        hw = config["model"].to(device)
        # DOES NOT LOAD OPTIMIZER, SCHEDULER, ETC?

    # Create trainer
    if config["style_encoder"] == "2StageNudger":
        train_baseline = False if config["load_path"] else True
        config["trainer"] = trainers.TrainerNudger(hw, config["nudger_optimizer"], config, criterion,
                                               train_baseline=train_baseline)
    elif config['seq2seq']:
        config['trainer'] = trainers.TrainerSeq2Seq(hw, optimizer, config, criterion)
    else:
        config["trainer"] = trainers.TrainerBaseline(hw, optimizer, config, criterion)

    # Alternative Models
    if config["style_encoder"] == "basic_encoder":
        config["secondary_criterion"] = CrossEntropyLoss()
    else:  # config["style_encoder"] = False
        config["secondary_criterion"] = None

    for epoch in range(config["starting_epoch"], config["starting_epoch"] + config["epochs_to_run"] + 1):
        LOGGER.info("Epoch: {}".format(epoch))
        config["current_epoch"] = epoch

        # Only test
        if config["improve_image"]:
            training_cer = improver(hw, test_dataloader, criterion, optimizer, dtype, config)
            break

        elif not config["test_only"]:
            training_cer = run_epoch(hw, train_dataloader, criterion, optimizer, dtype, config)

            scheduler.step()

            LOGGER.info("Training CER: {}".format(training_cer))
            config["train_cer"].append(training_cer)

        # CER plot
        test_cer = test(hw, test_dataloader, config["idx_to_char"], device, config)
        LOGGER.info("Test CER: {}".format(test_cer))
        config["test_cer"].append(test_cer)

        if config["use_visdom"]:
            config["visdom_manager"].update_plot("Test Error Rate", [epoch], test_cer)

        if config["test_only"]:
            break

        if not config["results_dir"] is None:

            # Save BSF
            if config['lowest_loss'] > test_cer:
                config['lowest_loss'] = test_cer
                log_print("Saving Best")
                save_model(config, bsf=True)

            elif epoch % config["save_freq"] == 0:
                log_print("Saving most recent model")
                save_model(config, bsf=False)

            plt_loss(config)

    return config['lowest_loss']


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
        traceback.print_exc()
    finally:
        torch.cuda.empty_cache()

# https://github.com/theevann/visdom-save/blob/master/vis.py
