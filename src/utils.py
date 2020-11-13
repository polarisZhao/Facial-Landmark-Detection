# -*- coding:utf-8 -*-
import os
import logging
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim import optimizer

import yaml


# --------------------------------------------------------
class AverageMeter(object):
    """ Computes and stores the average and current value """
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ------------------------------------------------------------------------
def get_optimizer(cfg, model):
    optimizer = None
    if cfg["TRAIN"]["OPTIMIZER"] == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad,
                                     model.parameters()),
                              lr=cfg["TRAIN"]["LR"],
                              momentum=cfg["TRAIN.MOMENTUM"],
                              weight_decay=cfg["TRAIN"]["WD"],
                              nesterov=cfg["TRAIN"]["NESTEROV"])
    elif cfg["TRAIN"]["OPTIMIZER"] == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                      model.parameters()),
                               lr=cfg["TRAIN"]["LR"])
    elif cfg["TRAIN"]["OPTIMIZER"] == 'rmsprop':
        optimizer = optim.RMSprop(filter(lambda p: p.requires_grad,
                                         model.parameters()),
                                  lr=cfg["TRAIN"]["LR"],
                                  momentum=cfg["TRAIN"]["MOMENTUM"],
                                  weight_decay=cfg["TRAIN"]["WD"],
                                  alpha=cfg["TRAIN"]["RMSPROP_ALPHA"],
                                  centered=cfg["TRAIN"]["RMSPROP_CENTERED"])

    return optimizer


def get_scheduler(cfg, optimizer, last_epoch):
    if isinstance(cfg["TRAIN"]["LR_STEP"], list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, cfg["TRAIN"]["LR_STEP"], cfg["TRAIN"]["LR_FACTOR"],
            last_epoch - 1)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, cfg["TRAIN"]["LR_STEP"], cfg["TRAIN"]["LR_FACTOR"],
            last_epoch - 1)
    return lr_scheduler


# ---------------------------------------------------------------


def save_checkpoint(states, is_best, output_dir, filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))

    if is_best and 'state_dict' in states.keys():
        torch.save(states['state_dict'].module.state_dict(),
                   os.path.join(output_dir, 'model_best.pth'))


def load_checkpoint(model_state_file, model, optimizer):
    checkpoint = torch.load(model_state_file)
    last_epoch = checkpoint['epoch']
    best_nme = checkpoint['best_nme']

    model.load_state_dict(checkpoint['state_dict'].state_dict())
    optimizer.load_state_dict(checkpoint['optimizer'])

    print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

    return checkpoint, last_epoch, best_nme


# -------------------------------------------------
def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg["OUTPUT_DIR"])
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg["DATASET"]["DATASET"]
    model = cfg["MODEL"]["NAME"]
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg["OUTPUT_DIR"]) / cfg["LOG_DIR"] / dataset / model / \
                        (cfg_name + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


# -------------------------------------------------
def configparse(cfg):
    """ parse config """
    config = {}
    if os.path.isfile(cfg):
        f = open(cfg)
        config = yaml.load(f)
    return config
