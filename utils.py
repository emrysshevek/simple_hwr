import pathlib
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
import string_utils
import error_rates
import glob
from pathlib import Path

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

def find_config(config_name):
    # Correct config paths
    if os.path.isfile(config_name):
        return config_name

    found_paths = []
    for d,s,fs in os.walk("./configs"):
        for f in fs:
            if config_name == f:
                found_paths.append(os.path.join(d, f))

    # Error handling
    if len(found_paths) == 1:
        return found_paths[0]
    elif len(found_paths) > 1:
        raise Exception("Multiple {} config were found: {}".format(config_name, "\n".join(found_paths)))
    elif len(found_paths) < 1:
        raise Exception("{} config not found".format(config_name))

def load_config(config_path):

    par, chld = os.path.split(config_path)

    if chld[-5:].lower() != ".yaml":
        chld = chld + ".yaml"

    if not os.path.isfile(config_path):
        config_path = find_config(chld)

    config = read_config(config_path)
    config["name"] = Path(config_path).stem
    defaults = {"load_path":False,
                "training_suffle": False,
                "testing_suffle": False,
                "test_only": False,
                "TESTING": False,
                "plot_freq": 50,
                "SMALL_TRAINING": False,
                "rnn_layers": 2,
                "nudger_rnn_layers": 2,
                "nudger_rnn_dimension": 512,
                "improve_image": False,
                "decoder_type" : "naive",
                "rnn_type": "lstm",
                "cnn": "default",
                "online_flag": True
                }

    for k in defaults.keys():
        if k not in config.keys():
            config[k] = defaults[k]

    if not config["TESTING"]:
        wait_for_gpu()

    if config["style_encoder"] == "fake_encoder":
        config["detach_embedding"] = True
    else:
        config["detach_embedding"] = False

    if "scheduler_step" not in config.keys() or "scheduler_gamma" not in config.keys():
        config["scheduler_step"] = 1
        config["scheduler_gamma"] = 1

    if config["SMALL_TRAINING"]:
        config["plot_freq"] = 1


    # Removing online jsons if not using online
    for data_path in config["training_jsons"]:
        if "online" in data_path and not config["online_augmentation"]:
            config["training_jsons"].remove(data_path)
            config["online_flag"] = False # turn off flag if no online data provided

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

    if config["TESTING"] or config["SMALL_TRAINING"]:
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
        if path is not None and len(path) > 0 and not os.path.exists(path):
            os.makedirs(path)

    # Make a link to most recent run
    try:
        link = "./RECENT.lnk"
        old_link = "./RECENT.lnk2"
        if os.path.exists(old_link):
            os.remove(old_link)
        if os.path.exists(link):
            os.rename(link, old_link)
        symlink(config['results_dir'], link)
    except Exception as e:
        print("Problem with RECENT link stuff: {}".format(e))


    # Save images
    if config["improve_image"]:
        config["save_improved_images"] = True
        config["image_dir"] = os.path.join(config["results_dir"], "images")
        mkdir(config["image_dir"])


    # Copy config to output folder
    #parent, child = os.path.split(config)
    shutil.copy(config_path, config['results_dir'])

    logger = setup_logging(folder=config["log_dir"])

    if config["debug"]:
        logger.setLevel("DEBUG")

    log_print("Using config file", config_path)
    log_print(json.dumps(config, indent=2))

    config["logger"] = logger

    config["stats"] = {}
    config = computer_defaults(config)
    return config

def computer_defaults(config):
    if socket.gethostname() == "Galois":
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    else:
        config["use_visdom"]=False
    return config

def symlink(target, link_location):
    while True:
        try:
            os.symlink(target, link_location)
            break
        except FileExistsError:
            os.remove(link_location)

def get_computer():
    return socket.gethostname()

def choose_optimal_gpu(priority="memory"):
    import GPUtil
    print(GPUtil.getGPUs())
    if priority == "memory":
        best_gpu = [(x.memoryFree, -x.load, i) for i,x in enumerate(GPUtil.getGPUs())]
    else: # utilization
        best_gpu = [(-x.load, x.memoryFree, i) for i, x in enumerate(GPUtil.getGPUs())]

    best_gpu.sort(reverse=True)
    best_gpu = best_gpu[0][2]
    os.environ['CUDA_VISIBLE_DEVICES'] = str(best_gpu)
    return best_gpu

def wait_for_gpu():
    if get_computer() != "Galois":
        return

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
    if memory_utilization > 40:
        pass
        # alias gpu_reset="kill -9 $(nvidia-smi | sed -n 's/|\s*[0-9]*\s*\([0-9]*\)\s*.*/\1/p' | sort | uniq | sed$

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
    if config["use_visdom"]:
        try:
            config["visdom_manager"].load_log(os.path.join(path, "visdom.json"))
        except:
            warnings.warn("Unable to load from visdom.json; does the file exist?")
            ## RECREAT VISDOM FROM FILE IF VISDOM IS NOT FOUND


    # Load Loss History
    stat_path = os.path.join(path, "all_stats.json")
    loss_path = os.path.join(path, "losses.json")

    with open(loss_path, 'r') as fh:
        losses = json.load(fh)

    try:
        config["train_cer"] = losses["train_cer"]
        config["test_cer"] = losses["test_cer"]
    except:
        warnings.warn("Could not load from losses.json")
        config["train_cer"]=[]
        config["test_cer"] = []

    if config["train_cer"]:
        config["lowest_loss"] = min(config["train_cer"])

    # Load stats
    try:
        with open(stat_path, 'r') as fh:
            stats = json.load(fh)

        for name, stat in config["stats"].items():
            if isinstance(stat, Stat):
                config["stats"][name].y = stats[name]["y"]
            else:
                for i in stats[name]: # so we don't mess up the reference etc.
                    config["stats"][name].append(i)

    except:
        warnings.warn("Could not load from all_stats.json")


def mkdir(path):
    if isinstance(path, str):
        if path is not None and len(path) > 0 and not os.path.exists(path):
            try:
                os.makedirs(path)
            except Exception as e:
                print(e)
    elif isinstance(path, Path):
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(e)
    else:
        warnings.warn("Unknown path type, cannot create folder")


def save_model(config, bsf=False):
    # Save the best model
    if bsf:
        path = os.path.join(config["results_dir"], "BSF")
        mkdir(path)
    else:
        path = config["results_dir"]

    state_dict = {
        'epoch': config["current_epoch"] + 1,
        'model': config["model"].state_dict(),
        'optimizer': config["optimizer"].state_dict(),
        'global_step': config["global_step"]
    }

    torch.save(state_dict, os.path.join(path, "{}_model.pt".format(config['name'])))

    if "nudger" in config.keys():
        state_dict["model"] = config["nudger"].state_dict()
        torch.save(state_dict, os.path.join(path, "{}_nudger_model.pt".format(config['name'])))

    # Save all stats
    results = config["stats"]
    with open(os.path.join(path, "all_stats.json"), 'w') as fh:
        json.dump(results, fh, cls=EnhancedJSONEncoder, indent=4)

    # Save CER
    #results = {"training":config["stats"][config["designated_training_cer"]], "test":config["stats"][config["designated_test_cer"]]}
    results = {"train_cer":config["train_cer"], "test_cer":config["test_cer"]}
    with open(os.path.join(path, "losses.json"), 'w') as fh:
        json.dump(results, fh, cls=EnhancedJSONEncoder, indent=4)

    # Save visdom
    if config["use_visdom"]:
        try:
            path = os.path.join(path, "visdom.json")
            config["visdom_manager"].save_env(file_path=path)
        except:
            warnings.warn(f"Unable to save visdom to {path}; is it started?")
            config["use_visdom"] = False

    # Copy BSF stuff to main directory
    if bsf:
        for filename in glob.glob(path + r"/*"):
            shutil.copy(filename, config["results_dir"])

def plt_loss(config):
    ## Plot with matplotlib
    try:
        x_axis = [(i + 1) * config["n_train_instances"] for i in range(len(config["train_cer"]))]
        plt.figure()
        plt.plot(x_axis, config["train_cer"], label='train')
        plt.plot(x_axis, config["test_cer"], label='test')
        plt.legend()
        plt.ylim(top=.2)
        plt.ylabel("CER")
        plt.xlabel("Number of Instances")
        plt.title("CER Loss")
        plt.savefig(os.path.join(config["results_dir"], config['name'] + ".png"))
        plt.close()
    except Exception as e:
        log_print("Problem graphing: {}".format(e))


class Decoder:
    def __init__(self, idx_to_char, beam=False):
        self.decode_training = self.decode_batch_naive
        self.decode_test = self.decode_batch_naive
        self.idx_to_char = idx_to_char

        if beam:
            from ctcdecode import CTCBeamDecoder
            print("Using beam")
            self.beam_decoder = CTCBeamDecoder(labels=idx_to_char.values(), blank_id=0, beam_width=30, num_processes=3, log_probs_input=True)
            self.decode_test = self.decode_batch_beam

    def decode_batch_naive(self, out):
        out = out.data.cpu().numpy()
        for j in range(out.shape[0]):
            logits = out[j, ...]
            pred, raw_pred = string_utils.naive_decode(logits)
            yield string_utils.label2str(pred, self.idx_to_char, False)

    def decode_batch_beam(self, out):
        pred, scores, timesteps, out_seq_len = self.beam_decoder.decode(out)
        pred = pred.data.int().numpy()
        output_lengths = out_seq_len.data.data.numpy()
        rank = 0 # get top ranked prediction
        # Loop through batches
        for batch in range(pred.shape[0]):
            line_length = output_lengths[batch][rank]
            line = pred[batch][rank][:line_length]
            string = u""
            for char in line:
                string += self.idx_to_char[char]
            yield string

def calculate_cer(pred_strs, gt):
    sum_loss = 0
    steps = 0
    for j, pred_str in enumerate(pred_strs):
        gt_str = gt[j]
        cer = error_rates.cer(gt_str, pred_str)
        sum_loss += cer
        steps += 1
    return sum_loss, steps


def accumulate_stats(config, freq=None):
    for title, stat in config["stats"].items():
        if isinstance(stat, Stat) and stat.accumlator_active and stat.accumulator_freq == freq:
            stat.reset_accumlator()
            config["logger"].debug(stat.name, stat.y[-1])

class Stat:
    def __init__(self, y, x, x_title="", y_title="", name="", plot=True, ymax=None, accumulator_freq=None):
        """

        Args:
            y (list): iterable (e.g. list) for storing y-axis values of statistic
            x (list): iterable (e.g. list) for storing x-axis values of statistic (e.g. epochs)
            x_title:
            y_title:
            name (str):
            plot (str):
            ymax (float):
            accumulator_freq: when should the variable be accumulated (e.g. each epoch, every "step", every X steps, etc.


        """
        super(Stat, self).__init__()
        self.y = y
        self.x = x
        self.current_weight = 0
        self.current_sum = 0
        self.accumlator_active = False
        self.updated_since_plot = False
        self.accumulator_freq = None # epoch or instances; when should this statistic accumulate?

        # Plot details
        self.x_title = x_title
        self.y_title = y_title
        self.ymax = ymax
        self.name = name
        self.plot = plot
        self.plot_update_length = 1 # add last X items from y-list to plot

    def yappend(self, new_item):
        self.y.append(new_item)
        if not self.updated_since_plot:
            self.updated_since_plot = True

    def default(self, o):
        return o.__dict__

    def accumulate(self, sum, weight):
        self.current_sum += sum
        self.current_weight += weight

        if not self.accumlator_active:
            self.accumlator_active = True

    def reset_accumlator(self):
        if self.accumlator_active:
            # print(self.current_weight)
            # print(self.current_sum)
            self.y += [self.current_sum / self.current_weight]
            self.current_weight = 0
            self.current_sum = 0
            self.accumlator_active = False
            self.updated_since_plot = True

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return str(self.__dict__)


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        try:
            return super().default(o)
        except:
            return o.__dict__

def stat_prep(config):
    """ Prep to track statistics/losses, setup plots etc.

    Returns:

    """
    config["stats"]["epochs"] = []
    config["stats"]["epoch_decimal"] = []
    config["stats"]["instances"] = []
    config["stats"]["updates"] = []

    # Prep storage
    config_stats = []
    config_stats.append(Stat(y=[], x=config["stats"]["updates"], x_title="Updates", y_title="Loss", name="HWR Training Loss"))
    config_stats.append(Stat(y=[], x=config["stats"]["epoch_decimal"], x_title="Epochs", y_title="CER", name="Training Error Rate"))
    config_stats.append(Stat(y=[], x=config["stats"]["epochs"], x_title="Epochs", y_title="CER", name="Test Error Rate", ymax=.2))
    config["designated_training_cer"] = "Training Error Rate"
    config["designated_test_cer"] = "Test Error Rate"

    if config["style_encoder"] in ["basic_encoder", "fake_encoder"]:
        config_stats.append(Stat(y=[], x=config["stats"]["updates"], x_title="Updates", y_title="Loss", name="Writer Style Loss"))

    if config["style_encoder"] in ["2StageNudger"]:
        config_stats.append(Stat(y=[], x=config["stats"]["updates"], x_title="Updates", y_title="Loss",name="Nudged Training Loss"))
        config_stats.append(Stat(y=[], x=config["stats"]["epoch_decimal"], x_title="Epochs", y_title="CER", name="Nudged Training Error Rate"))
        config_stats.append(Stat(y=[], x=config["stats"]["epochs"], x_title="Epochs", y_title="CER", name="Nudged Test Error Rate", ymax=.2))
        config["designated_training_cer"] = "Nudged Training Error Rate"
        config["designated_test_cer"] = "Nudged Test Error Rate"

        # Register plots, save in stats dictionary
    for stat in config_stats:
        if config["use_visdom"]:
            config["visdom_manager"].register_plot(stat.name, stat.x_title, stat.y_title, ymax=stat.ymax)
        config["stats"][stat.name] = stat


if __name__=="__main__":
    from visualize import Plot
    viz = Plot()
    viz.viz.close()
    viz.load_all_env("./results")

