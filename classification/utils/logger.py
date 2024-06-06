# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import sys
import logging
import functools
from termcolor import colored
import math
import numpy as np

import torch
from tensorboardX import SummaryWriter


class MyAverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self, name=None, max_len=-1):
        self.val_list = []
        self.count = []
        self.avg_name = name
        self.max_len = max_len
        self.val = 0
        self.avg = 0
        self.var = 0

    def update(self, val):
        self.val = val
        self.avg = 0
        self.var = 0
        if not math.isnan(val) and not math.isinf(val):
            self.val_list.append(val)
        # else:
        #     print(f'Nan in {self.avg_name}')
        if self.max_len > 0 and len(self.val_list) > self.max_len:
            self.val_list = self.val_list[-self.max_len:]
        if len(self.val_list) > 0:
            self.avg = np.mean(np.array(self.val_list))
            self.var = np.std(np.array(self.val_list))


@functools.lru_cache()
def create_logger(output_dir, dist_rank=0, name=''):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    # create file handlers
    file_handler = logging.FileHandler(os.path.join(output_dir, f'log_rank{dist_rank}.txt'), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger


class TensorboardLogger(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(logdir=log_dir)
        self.step = 0

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, head='scalar', step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.writer.add_scalar(head + "/" + k, v, self.step if step is None else step)

    def add_image(self, name, image):
        self.writer.add_image(name, image)
        
    def flush(self):
        self.writer.flush()