import json
import logging
import os

import yaml
from torch import nn


def read_yaml(path):
    with open(path, encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
        return config


def save_json2file(json_, save_dir, save_filename):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_dir + save_filename, "w") as f:
        json.dump(json_, f)
    return


def save_str2file(str_, save_dir, save_filename):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_dir + save_filename, "w") as f:
        f.write(str_)
    return


def transpose_to_4d_input(x):
    while len(x.shape) < 4:
        x = x.unsqueeze(-1)
    return x.permute(0, 3, 1, 2)


def init_weight_bias(model):
    for module in model.modules():
        if hasattr(module, "weight"):
            if not ("BatchNorm" in module.__class__.__name__):
                nn.init.xavier_uniform_(module.weight, gain=1)
            else:
                nn.init.constant_(module.weight, 1)
        if hasattr(module, "bias"):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


def get_logger(save_result, save_dir, save_file):
    logger = logging.getLogger(__name__)  # Use a specific logger name to avoid conflicts
    logger.setLevel(logging.INFO)

    # Check if handlers already exist to avoid adding them multiple times
    if not logger.hasHandlers():
        formatter = logging.Formatter(fmt="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

        # StreamHandler for console output
        str_handler = logging.StreamHandler()
        str_handler.setFormatter(formatter)
        logger.addHandler(str_handler)

        # If saving logs to file, add a FileHandler
        if save_result:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            file_handler = logging.FileHandler(os.path.join(save_dir, save_file), mode='w')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    # Disable propagation to avoid duplicate logging from root logger
    logger.propagate = False

    return logger

