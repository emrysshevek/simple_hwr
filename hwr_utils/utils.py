import numbers
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
from Bio import pairwise2
import numpy as np
import warnings
import glob
from pathlib import Path
from easydict import EasyDict as edict
from hwr_utils import visualize, string_utils, error_rates
from hwr_utils.stat import Stat

def to_numpy(tensor):
    if isinstance(tensor,torch.FloatTensor) or isinstance(tensor,torch.cuda.FloatTensor):
        return tensor.detach().cpu().numpy()
    else:
        return tensor

def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError as te:
        return False


def increment_path(name="", base_path="./logs", make_directory=True, ignore="partial"):
    """

    Args:
        name: Base name of directory to be preceded by numeral (e.g. 2MyLog)
        base_path: Log directory
        make_directory: Make the directory
        ignore: Don't include files matching this pattern when finding max number
    Returns:

    """

    # Check for existence
    Path(base_path).mkdir(parents=True, exist_ok=True)
    n, npath = get_max_file(base_path, ignore=ignore)

    # Create
    logdir = Path(os.path.join(base_path, "{:02d}_{}".format(n + 1, name)))
    if make_directory:
        Path(logdir).mkdir(parents=True, exist_ok=True)
    return logdir

def get_max_file(path, ignore=None):
    """ Gets the file with the highest (first) number in the string, ignoring the "ignore" string
    Args:
        path (str): Folder to search
    Returns:

    """
    if ignore:
        filtered = [p for p in os.listdir(path) if not re.search(ignore, p)]
    else:
        filtered = os.listdir(path)
    numbers = [(int(re.search("^[0-9]+", p)[0]), p) for p in filtered if re.search("^[0-9]+", p)]
    n, npath = max(numbers) if numbers else (0, "")
    # print("Last File Version: {}".format(npath))
    return n, os.path.join(path, npath)


def unpickle_it(path):
    with open(path, 'rb') as f:
        dict = pickle.load(f)  # , encoding = 'latin-1'
    return dict

def pickle_it(obj, path):
    with open(path, 'wb') as f:
        dict = pickle.dump(obj, f)  # , encoding = 'latin-1'

def print_tensor(tensor):
    log_print(tensor, tensor.shape)

def read_config(config):
    config = Path(config)
    log_print(config)
    if config.suffix.lower() == ".json":
        return json.load(config.open(mode="r"))
    elif config.suffix.lower() == ".yaml":
        return fix_scientific_notation(yaml.load(config.open(mode="r")))
    else:
        raise "Unknown Filetype {}".format(config)

def setup_logging(folder=None, log_std_out=True, level="INFO"):
    import logging, datetime, sys
    global LOGGER
    # Set up logging

    format = '%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s'
    format = {"fmt":'%(asctime)s %(levelname)s %(message)s', "datefmt":"%H:%M:%S"}

    # Override Pycharm logging - otherwise, logging file may not be created
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logger = logging.getLogger(__name__)

    today = datetime.datetime.now()
    log_path = "{}/{}.log".format(folder, today.strftime("%m-%d-%Y"))
    if folder is None:
        log_path = None
    logging.basicConfig(filename=log_path,
                            filemode='a',
                            format=format["fmt"],
                            datefmt=format["datefmt"],
                            level=level)

    # Send log messages to standard out
    if log_std_out:
        formatter = logging.Formatter(**format)
        std_out = logging.StreamHandler(sys.stdout)
        std_out.setLevel(level)
        std_out.setFormatter(formatter)
        logger.addHandler(std_out)

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

def find_config(config_name, config_root="./configs"):
    # Correct config paths
    if os.path.isfile(config_name):
        return config_name

    found_paths = []
    for d,s,fs in os.walk(config_root):
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

def incrementer(root, base):
    new_folder = Path(root / base)
    increment = 0
    increment_string = ""

    while new_folder.exists():
        increment += 1
        increment_string = f"{increment:02d}" if increment > 0 else ""
        new_folder = Path(root / (base + increment_string))

    new_folder.mkdir(parents=True, exist_ok=True)
    return new_folder

def load_config(config_path):
    config_path = Path(config_path)
    project_path = Path(os.path.realpath(__file__)).parent.parent.absolute()
    config_root = project_path / "configs"

    # Path was specified, but not found
    # if config_root not in config_path.absolute().parents:
    #     raise Exception("Could not find config!")

    # Try adding a suffix
    if config_path.suffix != ".yaml":
        config_path = config_path.with_suffix(".yaml")

    # Go search for it
    if not config_path.exists():
        config_path = find_config(config_path.name, config_root)

    config = edict(read_config(config_path))
    config["name"] = Path(config_path).stem  ## OVERRIDE NAME WITH THE NAME OF THE YAML FILE
    config["project_path"] = project_path
    defaults = {"load_path":False,
                "training_shuffle": False,
                "testing_shuffle": False,
                "test_only": False,
                "TESTING": False,
                "GPU": True,
                "SKIP_TESTING": False,
                "OVERFIT": False,
                "TEST_FREQ": 1,
                "SMALL_TRAINING": False,
                "images_to_load": None,
                "plot_freq": 50,
                "rnn_layers": 2,
                "nudger_rnn_layers": 2,
                "nudger_rnn_dimension": 512,
                "improve_image": False,
                "decoder_type" : "naive",
                "rnn_type": "lstm",
                "cnn": "default",
                "online_flag": True,
                "save_count": 0,
                "training_blur": False,
                "training_blur_level": 1.5,
                "training_random_distortions": False,
                "training_distortion_sigma": 6.0,
                "testing_blur": False,
                "testing_blur_level": 1.5,
                "testing_random_distortions": False,
                "testing_distortion_sigma": 6.0,
                "occlusion_size": None,
                "occlusion_freq": None,
                "logging": "info",
                "n_warp_iterations": 11,
                "testing_occlude": False,
                "testing_warp": False,
                "optimizer_type": "adam",
                "occlusion_level": .4,
                "exclude_offline": False,
                "validation_jsons": [],
                "elastic_transform": False,
                "coord_conv": False
                }

    for k in defaults.keys():
        if k not in config.keys():
            config[k] = defaults[k]

    # Main output folder
    if config["load_path"]:
        _output = incrementer(Path(config_root), "new_experiment") # if it has a load path, create a new experiment in that same folder!
        experiment = _output.stem
        output_root = _output.as_posix()
    else:
        try:
            experiment = Path(config_path).absolute().relative_to(Path(config_root).absolute()).parent
            if str(experiment) == ".": # if experiment is in root directory, use the experiment specified in the yaml
                experiment = config["experiment"]
            output_root = os.path.join(config["output_folder"], experiment)

        except:
            log_print(f"Failed to find relative path of config file {config_root} {config_path}")

    # Use config folder to determine output folder
    config["experiment"] = str(experiment)
    log_print(f"Experiment: {experiment}, Results Directory: {output_root}")

    # hyper_parameter_str='{}_lr_{}_bs_{}_warp_{}_arch_{}'.format(
    #      config["name"],
    #      config["learning_rate"],
    #      config["batch_size"],
    #      config["training_warp"],
    #      config["style_encoder"]
    #  )

    hyper_parameter_str='{}'.format(
         config["name"],
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
    if "image_dir" not in config.keys():
        config["image_dir"] = os.path.join(config['results_dir'], "imgs")

    config["image_test_dir"] = os.path.join(config["image_dir"], "test")
    config["image_train_dir"] = os.path.join(config["image_dir"], "train")

    # Create paths
    for path in (output_root, config["results_dir"], config["log_dir"], config["image_dir"], config["image_train_dir"], config["image_test_dir"]):
        if path is not None and len(path) > 0 and not os.path.exists(path):
            Path(path).mkdir(parents=True, exist_ok=True)

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
        log_print("Problem with RECENT link stuff: {}".format(e))

    # Copy config to output folder
    #parent, child = os.path.split(config)
    try:
        shutil.copy(config_path, config['results_dir'])
    except Exception as e:
        log_print(f"Could not copy config file: {e}")

    config = make_config_consistent(config)

    logger = setup_logging(folder=config["log_dir"], level=config["logging"].upper())
    log_print(f"Effective logging level: {logger.getEffectiveLevel()}")
    log_print("Using config file", config_path)
    #log_print(json.dumps(config, indent=2))

    config["logger"] = logger

    config["stats"] = {}
    config = computer_defaults(config)

    #make_lower(config)
    return config

def make_config_consistent(config):
    if config["SMALL_TRAINING"] or config["TESTING"]:
        config["images_to_load"] = config["batch_size"]

    config["occlusion"] = config["occlusion_size"] and config["occlusion_freq"]
    if not config["testing_occlude"] and not config["testing_warp"]:
        config["n_warp_iterations"] = 0
    elif (config["testing_occlude"] or config["testing_warp"]) and config["n_warp_iterations"] == 0:
        config["n_warp_iterations"] = 7
        log_print("n_warp_iterations set to 0, changing to 11")

    if config["exclude_offline"]:
        training_data = "prepare_IAM_Lines/gts/lines/txt/training.json"
        if training_data in config["training_jsons"]:
            config["training_jsons"].remove(training_data)
            if not config.training_jsons:
                raise Exception("No training data -- check exclude_offline flag")

    # Training data should be a list
    print(type(config.training_jsons))

    if isinstance(config.training_jsons, str):
        config.training_jsons = [config.training_jsons]
    if isinstance(config.testing_jsons, str):
        config.testing_jsons = [config.testing_jsons]
    if isinstance(config.validation_jsons, str):
        config.validation_jsons = [config.validation_jsons]

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
            log_print(f"Online flag not specified -- removing {data_path}")
            config["training_jsons"].remove(data_path)
            config["online_flag"] = False # turn off flag if no online data provided
        if not config.training_jsons:
            log_print("No training json files -- check online_augmentation flag")
            raise Exception("No training json files -- check online_augmentation flag")

    # Save images
    if config["improve_image"]:
        config["save_improved_images"] = True
        config["image_dir"] = os.path.join(config["results_dir"], "images")
        mkdir(config["image_dir"])

    return config

def make_lower(config):
    exclusions=("experiment","name","output_folder","writer_id_pickles", "training_jsons",
                "testing_jsons", "training_root", "testing_root", "results_dir", "log_dir", "img_dir")
    for key,value in config.items():
        if isinstance(config[key], str) and key not in exclusions and "dir" not in key and "path" not in key:
            config[key]=value.lower()

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
    log_print(GPUtil.getGPUs())
    if priority == "memory":
        best_gpu = [(x.memoryFree, -x.load, i) for i,x in enumerate(GPUtil.getGPUs())]
    else: # utilization
        best_gpu = [(-x.load, x.memoryFree, i) for i, x in enumerate(GPUtil.getGPUs())]

    try:
        best_gpu.sort(reverse=True)
        best_gpu = best_gpu[0][2]
        os.environ['CUDA_VISIBLE_DEVICES'] = str(best_gpu)
        return best_gpu
    except:
        return None

def wait_for_gpu():
    if get_computer() != "Galois":
        return

    ## Wait until GPU is available -- only on Galois
    import GPUtil
    GPUtil.showUtilization()
    GPUs = GPUtil.getGPUs()
    utilization = GPUs[0].load * 100  # memoryUtil
    memory_utilization = GPUs[0].memoryUtil * 100  #
    log_print(utilization)
    if memory_utilization > 40:
        warnings.warn("Memory utilization is high; close other GPU processes")

    while utilization > 45:
        log_print("Waiting 30 minutes for GPU...")
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

def dict_to_list(d):
    idx_to_char = []
    for i in range(0, max(d.keys())+1):
        idx_to_char.append(d[i])
    return idx_to_char

def load_model(config):
    # User can specify folder or .pt file; other files are assumed to be in the same folder
    if os.path.isfile(config["load_path"]):
        old_state = torch.load(config["load_path"])
        path, child = os.path.split(config["load_path"])
    else:
        old_state = torch.load(os.path.join(config["load_path"], "baseline_model.pt"))
        path = config["load_path"]

    # Load the definition of the loaded model if it was saved
    # if "model_definition" in old_state.keys():
    #     config["model"] = old_state['model_definition']

    for key in ["idx_to_char", "char_to_idx"]:
        if key in old_state.keys():
            if key == "idx_to_char":
                old_state[key] = dict_to_list(old_state[key])
            config[key] = old_state[key]

    if "model" in old_state.keys():
        config["model"].load_state_dict(old_state["model"])
        config["optimizer"].load_state_dict(old_state["optimizer"])
        config["global_counter"] = old_state["global_step"]
        config["starting_epoch"] = old_state["epoch"]
        config["current_epoch"] = old_state["epoch"]
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
        config["validation_cer"] = losses["validation_cer"]
    except:
        warnings.warn("Could not load from losses.json")
        config["train_cer"]=[]
        config["test_cer"] = []

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

def save_stats(config, bsf):
    if bsf:
        path = os.path.join(config["results_dir"], "BSF")
        mkdir(path)
    else:
        path = config["results_dir"]

    # Save all stats
    results = config["stats"]
    with open(os.path.join(path, "all_stats.json"), 'w') as fh:
        json.dump(results, fh, cls=EnhancedJSONEncoder, indent=4)

    # Save CER
    config.train_cer = config.stats[config["designated_validation_cer"]].y
    config.validation_cer = config.stats[config["designated_training_cer"]].y
    config.test_cer = config.stats[config["designated_test_cer"]].y

    #results = {"training":config["stats"][config["designated_training_cer"]], "test":config["stats"][config["designated_test_cer"]]}
    results = {"train_cer":config.train_cer, "validation_cer":config.validation_cer, "test_cer":config.test_cer}
    with open(os.path.join(path, "losses.json"), 'w') as fh:
        json.dump(results, fh, cls=EnhancedJSONEncoder, indent=4)


def save_model(config, bsf=False):
    # Can pickle everything in config except items below
    # for x, y in config.items():
    #     print(x)
    #     if x in ("criterion", "visdom_manager", "trainer"):
    #         continue
    #     torch.save(y, "TEST.pt")

    # Save the best model
    if bsf:
        path = os.path.join(config["results_dir"], "BSF")
        mkdir(path)
    else:
        path = config["results_dir"]

    #    'model_definition': config["model"],
    state_dict = {
        'epoch': config["current_epoch"] + 1,
        'model': config["model"].state_dict(),
        'optimizer': config["optimizer"].state_dict(),
        'global_step': config["global_step"],
        "idx_to_char": config["idx_to_char"],
        "char_to_idx": config["char_to_idx"]
    }

    config["main_model_path"] = os.path.join(path, "{}_model.pt".format(config['name']))
    torch.save(state_dict, config["main_model_path"])

    if "nudger" in config.keys():
        state_dict["model"] = config["nudger"].state_dict()
        torch.save(state_dict, os.path.join(path, "{}_nudger_model.pt".format(config['name'])))

    save_stats(config, bsf)

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

    if config["save_count"]==0:
        create_resume_training(config)
    config["save_count"] += 1

def create_resume_training(config):
    #export_config = copy.deepcopy(config)
    export_config = config.copy()
    export_config["load_path"] = config["main_model_path"]

    for key in config.keys():
        item = config[key]
        # Only keep items that are numbers, strings, and lists
        if not isinstance(item, str) \
                and not isinstance(item, numbers.Number) \
                and not isinstance(item, list) \
                and item is not None \
                and not isinstance(item, bool):
            del export_config[key]

    output = Path(config["results_dir"])
    with open(Path(output / 'RESUME.yaml'), 'w') as outfile:
        yaml.dump(export_config, outfile, default_flow_style=False, sort_keys=False)

    with open(Path(output / 'TEST.yaml'), 'w') as outfile:
        export_config["test_only"] = True
        if export_config["training_warp"]:
            export_config["testing_warp"] = True
        if export_config["occlusion_level"]:
            export_config["testing_occlude"] = True
        if (export_config["testing_occlude"] or export_config["testing_warp"]) and not export_config["n_warp_iterations"]:
            export_config["n_warp_iterations"] = 21
        yaml.dump(export_config, outfile, default_flow_style=False, sort_keys=False)

def plt_loss(config):
    ## Plot with matplotlib
    try:
        x_axis = [(i + 1) * config["n_train_instances"] for i in range(len(config["train_cer"]))]
        plt.figure()
        plt.plot(x_axis, config["train_cer"], label='train')
        plt.plot(x_axis, config["validation_cer"], label='validation')
        if config["test_cer"]:
            plt.plot(config["test_epochs"], config["test_cer"], label='validation')
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
            log_print("Using beam")
            self.beam_decoder = CTCBeamDecoder(labels=idx_to_char.values(), blank_id=0, beam_width=30, num_processes=3, log_probs_input=True)
            self.decode_test = self.decode_batch_beam

    def decode_batch_naive(self, out, as_string=True):
        out = out.data.cpu().numpy()
        for j in range(out.shape[0]):
            logits = out[j, ...]
            pred, raw_pred = string_utils.naive_decode(logits)
            if as_string:
                yield string_utils.label2str(pred, self.idx_to_char, False)
            else:
                yield pred

    def decode_batch_beam(self, out, as_string=True):
        pred, scores, timesteps, out_seq_len = self.beam_decoder.decode(out)
        pred = pred.data.int().numpy()
        output_lengths = out_seq_len.data.data.numpy()
        rank = 0 # get top ranked prediction
        # Loop through batches
        for batch in range(pred.shape[0]):
            line_length = output_lengths[batch][rank]
            line = pred[batch][rank][:line_length]
            string = u""
            if as_string:
                for char in line:
                    string += self.idx_to_char[char]
                yield string
            else:
                yield line

def calculate_cer(pred_strs, gt):
    sum_loss = 0
    steps = 0
    for j, pred_str in enumerate(pred_strs):
        gt_str = gt[j]
        cer = error_rates.cer(gt_str, pred_str)
        sum_loss += cer
        steps += 1
    return sum_loss, steps


def accumulate_all_stats(config, keyword="", freq=None):
    """ Only update the stats that have the same frequency as the update call

    Args:
        config:
        keyword:
        freq:

    Returns:

    """

    for title, stat in config["stats"].items():
        if isinstance(stat, Stat) and stat.accumlator_active and stat.accumulator_freq == freq and keyword.lower() in stat.name.lower():
            stat.reset_accumlator(epoch=config.current_epoch, instance=config.global_instances_counter)
            config["logger"].debug(f"{stat.name} {stat.y[-1]}")

    try:
        visualize.plot_all(config)
    except:
        print("Problem graphing")

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
    #config["stats"]["instances"] = []
    config["stats"]["updates"] = []

    # Prep storage
    config_stats = []
    config_stats.append(Stat(y=[], x=config["stats"]["updates"], x_title="Updates", y_title="Loss", name="HWR Training Loss"))
    config_stats.append(Stat(y=[], x=[], x_title="Instances", y_title="CER", name="Training Error Rate", accumulator_freq="instance"))
    config_stats.append(Stat(y=[], x=[], x_title="Instances", y_title="CER", name="Test Error Rate", ymax=.2, accumulator_freq="instance"))
    config_stats.append(Stat(y=[], x=[], x_title="Instances", y_title="CER", name="Validation Error Rate", ymax=.2, accumulator_freq="instance"))
    config["designated_training_cer"] = "Training Error Rate"
    config["designated_test_cer"] = "Test Error Rate"
    config["designated_validation_cer"] = "Validation Error Rate" if config["validation_jsons"] else "Test Error Rate"

    if config["style_encoder"] in ["basic_encoder", "fake_encoder"]:
        config_stats.append(Stat(y=[], x=config["stats"]["updates"], x_title="Updates", y_title="Loss", name="Writer Style Loss"))

    if config["style_encoder"] in ["2StageNudger"]:
        config_stats.append(Stat(y=[], x=config["stats"]["updates"], x_title="Updates", y_title="Loss",name="Nudged Training Loss"))
        config_stats.append(Stat(y=[], x=config["stats"]["epoch_decimal"], x_title="Epochs", y_title="CER", name="Nudged Training Error Rate"))
        config_stats.append(Stat(y=[], x=config["stats"]["epochs"], x_title="Epochs", y_title="CER", name="Nudged Test Error Rate", ymax=.2))
        config_stats.append(Stat(y=[], x=config["stats"]["epochs"], x_title="Epochs", y_title="CER", name="Nudged Validation Error Rate",ymax=.2))
        config["designated_training_cer"] = "Nudged Training Error Rate"
        config["designated_test_cer"] = "Nudged Test Error Rate"
        config["designated_validation_cer"] = "Nudged Validation Error Rate" if config["validation_jsons"] else "Nudged Test Error Rate"

        # Register plots, save in stats dictionary
    for stat in config_stats:
        if config["use_visdom"]:
            config["visdom_manager"].register_plot(stat.name, stat.x_title, stat.y_title, ymax=stat.ymax)
        config["stats"][stat.name] = stat

def plot_tensors(tensor):
    for i in range(0, tensor.shape[0]):
        plot_tensor(tensor[i,0])

def plot_tensor(tensor):
    #print(tensor.shape)
    t = tensor.cpu()
    assert not np.isnan(t).any()
    plt.figure(dpi=400)
    plt.imshow(t, cmap='gray')
    plt.show()


if __name__=="__main__":
    from hwr_utils.visualize import Plot
    viz = Plot()
    viz.viz.close()
    viz.load_all_env("./results")

