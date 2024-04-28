import os
import torch
import numpy as np
import random
import os
import yaml
import json

from tools.optimization import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_args(filename, args):
    with open(filename, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    for key, group in data_loaded.items():
        for key, val in group.items():
            setattr(args, key, val)


def write_json(filename, content):
    with open(filename, 'w') as f:
        json.dump(content, f)


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


def get_optimizer(model, config):
    if config.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    return optimizer


def get_scheduler(optimizer, config, num_batches=-1):
    if not hasattr(config, 'scheduler'):
        return None
    if config.scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    elif config.scheduler == 'linear_w_warmup' or config.scheduler == 'cosine_w_warmup':
        assert num_batches != -1
        num_training_steps = num_batches * config.epochs
        num_warmup_steps = int(config.warmup_proportion * num_training_steps)
        if config.scheduler == 'linear_w_warmup':
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)
        if config.scheduler == 'cosine_w_warmup':
            scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)
    return scheduler


def step_scheduler(scheduler, config, bid, num_batches):
    if config.scheduler in ['StepLR']:
        if bid + 1 == num_batches:    # end of the epoch
            scheduler.step()
    elif config.scheduler in ['linear_w_warmup', 'cosine_w_warmup']:
        scheduler.step()

    return scheduler
