"""
解析配置的yaml文件
根据配置的yaml文件在注册表中找到并定义指定的model、dataset、processor、task、lr_scheduler等基础组件
将基础组件插入Trainer，调用trainer.train()进行训练和验证
"""
import os
from pathlib import Path
import warnings
import argparse
from omegaConf import OmegaConf
import random
import numpy as np
import torch
import torch.distributed as dist
from common.dist_utils import(
    init_distributed_mode,
    main_process
)
from common.registry import registry
from common.logger import setup_logger
from tasks import setup_task
from trainer import Trainer

from common.optims import (
    LinearWarmupStepLRScheduler,
    LinearWarmupCosineLRScheduler,
    ConstantLRScheduler,
)

from processors import load_processor
from models import *
from datasets import load_dataset

warnings.filterwarnings('ignore')
def now():
    from datetime import datetime
    return datetime.now().strftime('%Y%m%d%H%M')[:-1]

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

def get_config(args):
    cfg_path = Path(args.cfg_path)
    config = omegaConf.load(cfg_path)
    init_distributed_mode(config.run)
    return config

def get_transforms(config) -> dict:
    dataset_cfg = config.dataset
    transforms = {}
    transforms['train'] = load_processor(**dataset_cfg.train_cfg.transform)
    transforms['val'] = load_processor(**dataset_cfg.val_cfg.transform)
    return transforms

def get_model(config):
    model_cfg = config.model
    return registry.get_model_class(model_cfg.arch).from_config(model_cfg)



def get_datasets(config, transforms) -> dict:
    dataset_cfg = config.dataset
    datasets = {}

    val_cfg = dict(dataset_cfg.pop('val_cfg'))
    train_cfg = dict(dataset_cfg.pop('train_cfg'))
    train_cfg['transform'],val_cfg['transform'] = transforms['train'], transforms['val']
    datasets['train'] = load_dataset(train_cfg.pop('name'), train_cfg)
    datasets['val'] = load_dataset(val_cfg.pop('name'), val_cfg)
    return datasets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg-path', type=str)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)
    config = get_config(args)
    setup_logger()
    transforms = get_transforms(config)
    datasets = get_datasets(config, transforms)
    model = get_model(config)
    task = setup_task(config)
    job_id = now()

    trainer = Trainer(config, model, datasets, task,job_id)
    trainer.train()

if __name == '__main__':
    main()
