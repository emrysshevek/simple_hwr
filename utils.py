import torch
import re
import shutil
import time
import pickle
import yaml
import json
import os
import datetime

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
                            level=logging.DEBUG)
    if not log_std_out:
        logger.addHandler(logging.StreamHandler())
        logger.setLevel("DEBUG")

    LOGGER = logger
    return logger

def log_print(*args, print_statements=True, **kwargs):
    if kwargs.pop('new_start', False):
        setup_logging(kwargs["log_dir"], log_std_out=~print_statements)
    if print_statements:
        print(" ".join([str(a) for a in args]))
    else:
        LOGGER.debug(" ".join([str(a) for a in args]))

def load_config(config_path):
    config = read_config(config_path)

    # Main output folder
    output_root = os.path.join(config["output_folder"], config["name"])

    hyper_parameter_str='lr_{}_bs_{}_warp_{}_arch_{}'.format(
         config["learning_rate"],
         config["batch_size"],
         config["warp"],
         config["style_encoder"]
     )

    train_suffix = '{}-{}'.format(
        hyper_parameter_str,
        time.strftime("%Y%m%d-%H%M%S"))
    config["full_specs"] = train_suffix
    # Default options
    if 'results_dir' not in config.keys():
        config['results_dir']=os.path.join(output_root, train_suffix)
    if "log_dir" not in config.keys():
        config["log_dir"]=os.path.join(output_root, train_suffix)
    if "scheduler_step" not in config.keys() or "scheduler_gamma" not in config.keys():
        config["scheduler_step"] = 1
        config["scheduler_gamma"] = 1
    if config["TESTING"]:
        config['results_dir'] = None
        config["log_dir"] = None
        config["output_folder"] = None
    else:
        # Create paths
        for path in (output_root, config["results_dir"], config["log_dir"]):
            if len(path) > 0 and not os.path.exists(path):
                os.makedirs(path)

        # Copy config to output folder
        #parent, child = os.path.split(config)
        shutil.copy(config_path, config['results_dir'])

    log_print("Using config file", config_path, new_start=True, log_dir=config["log_dir"])
    log_print(json.dumps(config, indent=2))

    # Set defaults if unspecified
    if "training_suffle" not in config.keys():
        config['training_suffle'] = False
    if "testing_suffle" not in config.keys():
        config['testing_suffle'] = False

    return config

def wait_for_gpu():
    ## Wait until GPU is available
    import GPUtil
    GPUtil.showUtilization()
    GPUs = GPUtil.getGPUs()
    utilization = GPUs[0].load * 100  # memoryUtil
    print(utilization)

    while utilization > 45:
        print("Waiting 30 minutes for GPU...")
        time.sleep(1800)
        utilization = GPUs[0].load * 100  # memoryUtil
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

if __name__=="__main__":
    from visualize import Plot
    viz = Plot()
    viz.viz.close()
    viz.load_all_env("./results")