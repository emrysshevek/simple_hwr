
import socket
import argparse
import matplotlib.pyplot as plt
import torch
import re
import shutil
import time
import pickle
import yaml
import json
import os
import datetime
from Bio import pairwise2
import numpy as np
import warnings

def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError as te:
        return False

def unpickle_it(path):
    with open(path, 'rb') as f:
        dict = pickle.load(f)  # , encoding = 'latin-1'
    return dict

def pickle_it(obj, path):
    with open(path, 'wb') as f:
        dict = pickle.dump(obj, f)  # , encoding = 'latin-1'

def print_tensor(tensor):
    print(tensor, tensor.shape)

def read_config(config):
    print(config)
    if config[-5:].lower() == ".json":
        with open(config) as f:
            return json.load(f)
    elif config[-5:].lower() == ".yaml":
        with open(config, 'r') as stream:
            return fix_scientific_notation(yaml.load(stream))
    else:
        raise "Unknown Filetype {}".format(config)

def setup_logging(folder, log_std_out=False):
    global LOGGER
    ## Set up logging
    import logging

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logger = logging.getLogger(__name__)

    today = datetime.datetime.now()
    log_path = "{}/{}.log".format(folder, today.strftime("%m-%d-%Y"))
    if folder is None:
        log_path = None
    logging.basicConfig(filename=log_path,
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)
    if not log_std_out:
        logger.addHandler(logging.StreamHandler())
        logger.setLevel("INFO")

    LOGGER = logger
    return logger

def log_print(*args, print_statements=True):
    if print_statements:
        print(" ".join([str(a) for a in args]))
    else:
        LOGGER.info(" ".join([str(a) for a in args]))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="./configs/taylor.yaml", help='Path to the config file.')
    #parser.add_argument('--name', type=str, default="", help='Optional - special name for this run')

    opts = parser.parse_args()
    return opts
    # if "name" not in config.keys():
    #     config["name"] = opts.name
    # elif not config["name"] and opts.name:
    #     config["name"] = opts.name


def load_config(config_path):
    par, chld = os.path.split(config_path)

    # Correct config paths
    if par=="":
        config_path = os.path.join("./configs", config_path)
    if config_path[-5:].lower() != ".yaml":
        config_path = config_path + ".yaml"
    config = read_config(config_path)

    # Main output folder
    output_root = os.path.join(config["output_folder"], config["experiment"])

    hyper_parameter_str='{}_lr_{}_bs_{}_warp_{}_arch_{}'.format(
         config["name"],
         config["learning_rate"],
         config["batch_size"],
         config["training_warp"],
         config["style_encoder"]
     )

    train_suffix = '{}-{}'.format(    
        time.strftime("%Y%m%d_%H%M%S"),
        hyper_parameter_str)

    if config["TESTING"]:
        train_suffix = "TEST_"+train_suffix

    config["full_specs"] = train_suffix

    # Directory overrides
    if 'results_dir' not in config.keys():
        config['results_dir']=os.path.join(output_root, train_suffix)
    if 'output_predictions' not in config.keys():
        config['output_predictions']=False
    if "log_dir" not in config.keys():
        config["log_dir"]=os.path.join(output_root, train_suffix)

    # Create paths
    for path in (output_root, config["results_dir"], config["log_dir"]):
        print(path)
        if path is not None and len(path) > 0 and not os.path.exists(path):
            os.makedirs(path)

    # Copy config to output folder
    #parent, child = os.path.split(config)
    shutil.copy(config_path, config['results_dir'])

    logger = setup_logging(folder=config["log_dir"])

    if config["debug"]:
        logger.setLevel("DEBUG")

    log_print("Using config file", config_path)
    log_print(json.dumps(config, indent=2))

    config["logger"] = logger


    # Set defaults if unspecified
    if "load_path" not in config.keys():
        config["load_path"] = False
    if "training_suffle" not in config.keys():
        config['training_suffle'] = False
    if "testing_suffle" not in config.keys():
        config['testing_suffle'] = False
    if config["style_encoder"] == "fake_encoder":
        config["detach_embedding"] = True
    else:
        config["detach_embedding"] = False
    if "scheduler_step" not in config.keys() or "scheduler_gamma" not in config.keys():
        config["scheduler_step"] = 1
        config["scheduler_gamma"] = 1
    if "test_only" not in config.keys():
        config["test_only"] = False

    config = computer_defaults(config)
    return config

def computer_defaults(config):
    if socket.gethostname() == "Galois":
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    else:
        config["use_visdom"]=False
    return config

def get_computer():
    return socket.gethostname()

def wait_for_gpu():
    if get_computer() != "Galois":
        return

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    ## Wait until GPU is available -- only on Galois
    import GPUtil
    GPUtil.showUtilization()
    GPUs = GPUtil.getGPUs()
    utilization = GPUs[0].load * 100  # memoryUtil
    memory_utilization = GPUs[0].memoryUtil * 100  #
    print(utilization)
    if memory_utilization > 40:
        warnings.warn("Memory utilization is high; close other GPU processes")

    while utilization > 45:
        print("Waiting 30 minutes for GPU...")
        time.sleep(1800)
        utilization = GPUtil.getGPUs()[0].load * 100  # memoryUtil
    torch.cuda.empty_cache()
    return

def is_iterable(object, string_is_iterable=True):
    """Returns whether object is an iterable. Strings are considered iterables by default.

    Args:
        object (?): An object of unknown type
        string_is_iterable (bool): True (default) means strings will be treated as iterables
    Returns:
        bool: Whether object is an iterable

    """

    if not string_is_iterable and type(object) == type(""):
        return False
    try:
        iter(object)
    except TypeError as te:
        return False
    return True


def fix_scientific_notation(config):
    exp = re.compile("[0-9.]+e[-0-9]+")
    for key,item in config.items():
        #print(item, is_iterable(item, string_is_iterable=False))
        if type(item) is str and exp.match(item):
            config[key] = float(item)
    return config

def get_last_index(my_list, value):
    return len(my_list) - 1 - my_list[::-1].index(value)


def write_out(folder, fname, text):
    with open(os.path.join(folder, fname), "a") as f:
        f.writelines(text+"\n\n")

class CharAcc:
    def __init__(self, char_to_idx):
        self.correct = np.zeros(len(char_to_idx.keys()))
        self.actual_counts = np.zeros(len(char_to_idx.keys()))
        self.false_positive = self.correct.copy() # thought letter was found, was not
        self.false_negative = self.correct.copy() # missed true classification

        char_to_idx = char_to_idx.copy()
        if min(char_to_idx.values())==1:
            for key, val in char_to_idx.items():
                char_to_idx[key] = val-1

        self.char_to_idx = char_to_idx
        self.char_to_idx["|"] = self.char_to_idx["-"]

    def char_accuracy(self, pred, gt):
        pred = pred.replace("-", "|")
        gt = gt.replace("-", "|")
        pred_algn, gt_algn, *_ = pairwise2.align.globalxx(pred, gt)[0]

        for i, c in enumerate(pred_algn):
            guess_char=pred_algn[i]
            true_char=gt_algn[i]
            self.actual_counts[self.char_to_idx[true_char]] += 1
            if guess_char == true_char:
                self.correct[self.char_to_idx[c]] += 1
            elif true_char == "-": # model inserted a letter incorrectly
                self.false_positive[self.char_to_idx[guess_char]] += 1
            elif guess_char == "-": # model missed letter
                self.false_negative[self.char_to_idx[true_char]] += 1
            elif guess_char != true_char: # model missed one letter, and incorrectly posited another
                self.false_positive[self.char_to_idx[guess_char]] += 1
                self.false_negative[self.char_to_idx[true_char]] += 1

def load_model(config):
    # User can specify folder or .pt file; other files are assumed to be in the same folder
    if os.path.isfile(config["load_path"]):
        old_state = torch.load(config["load_path"])
        path, child = os.path.split(config["load_path"])
    else:
        old_state = torch.load(os.path.join(config["load_path"], "baseline_model.pt"))
        path = config["load_path"]

    if "model" in old_state.keys():
        config["model"].load_state_dict(old_state["model"])
        config["optimizer"].load_state_dict(old_state["optimizer"])
        config["global_counter"] = old_state["global_step"]
        config["starting_epoch"] = old_state["epoch"]
    else:
        config["model"].load_state_dict(old_state)

    # Launch visdom
    config["visdom_manager"].load_log(os.path.join(path, "visdom.json"))

    # Load Loss History
    with open(os.path.join(path, "losses.json"), 'r') as fh:
        losses = json.load(fh)
    config["train_losses"] = losses["train"]
    config["test_losses"] = losses["test"]
    if config["train_losses"]:
        config["lowest_loss"] = min(config["train_losses"])

def mkdir(path):
    if path is not None and len(path) > 0 and not os.path.exists(path):
        os.makedirs(path)

def save_model(config, bsf=False):
    # Save the best model
    if bsf:
        path = os.path.join(config["results_dir"], "BSF")
        mkdir(path)
    else:
        path = config["results_dir"]

    log_print("Saving Best")

    state_dict = {
        'epoch': config["current_epoch"] + 1,
        'model': config["model"].state_dict(),
        'optimizer': config["optimizer"].state_dict(),
        'global_step': config["global_counter"]
    }

    torch.save(state_dict, os.path.join(path, "{}_model.pt".format(config['name'])))

    # Save losses/CER
    results = {'train': config["train_losses"], 'test': config["test_losses"]}
    with open(os.path.join(path, "losses.json"), 'w') as fh:
        json.dump(results, fh, indent=4)

    # Save visdom
    config["visdom_manager"].save_env(file_path=os.path.join(path, "visdom.json"))


def plt_loss(config):
    ## Plot with matplotlib
    try:
        x_axis = [(i + 1) * config["n_train_instances"] for i in range(len(config["train_losses"]))]
        plt.figure()
        plt.plot(x_axis, config["train_losses"], label='train')
        plt.plot(x_axis, config["test_losses"], label='test')
        plt.legend()
        plt.ylim(top=.2)
        plt.ylabel("CER")
        plt.xlabel("Number of Instances")
        plt.title("CER Loss")
        plt.savefig(os.path.join(config["results_dir"], config['name'] + ".png"))
        plt.close()
    except Exception as e:
        log_print("Problem graphing: {}".format(e))

if __name__=="__main__":
    from visualize import Plot
    viz = Plot()
    viz.viz.close()
    viz.load_all_env("./results")

