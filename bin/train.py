import argparse
import json
import re

import torch
from torch.utils.data import DataLoader

from tool.init_everything import (
    init_model, 
    init_scheduler,
    init_optimizer
    )
from tool.trainer import train_joint, train_single
from tool.dataset import Dataset
from tool.utils import read_symbol_table

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", required=True, type=str, help="train data list")
    parser.add_argument("--dev_data", required=True, type=str, help="dev data list")
    parser.add_argument("--model_dir", required=True, help="save model to model_dir")
    parser.add_argument("--checkpoint", default=None, help="model checkpoint")
    parser.add_argument("--config", required=True, type=str, help="config path")
    parser.add_argument("--format", default="raw", type=str, help="data format")
    parser.add_argument("--symbol_table", type=str, default=None)
    parser.add_argument("--ddp.rank",
                        dest='rank',
                        default=-1,
                        type=int,
                        help='''number of total processes/gpus for
                        distributed training''')
    parser.add_argument('--ddp.dist_backend',
                        dest='dist_backend',
                        default='nccl',
                        choices=['nccl', 'gloo'],
                        help='distributed backend')
    parser.add_argument('--ddp.init_method',
                        dest='init_method',
                        default=None,
                        help='ddp init method')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--cmvn', default=None, help='global cmvn file')           
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    config= json.load(args.config)
    epoch_num = config.pop('epoch_num', 99)
    mission = config['mission']
    joint = len(mission.split('_')) > 1
    if not joint:
        scheduler_config = config.pop('scheduler')
        scheduler = init_scheduler(scheduler_config)
        model_config = config
        model = init_model(model_config)
        
    else:
        mission1, mission2 = mission.split('_')
        if mission2 == "asr":
            assert args.symbol_table
        
        model = init_model(model_config)
        optimizer_config = config.pop('optimizer')
        optimizer = init_optimizer(optimizer_config)
        scheduler_config = config.pop('scheduler')
        scheduler = init_scheduler(scheduler_config)
        n_epoch = config.pop('epoch_num')

        dataset_config = config.pop('dataset_conf', {})
        if args.symbol_table:
            symbol_table = read_symbol_table(args.symbol_table)
            
        train_dataset = Dataset(args.format,
                          args.train_data,
                          symbol_table,
                          dataset_config)
        dev_dataset = Dataset(args.format,
                            args.dev_data,
                            symbol_table,
                            dataset_config)

        train_dataloader = DataLoader(train_dataset,
                                    batch_size=None,
                                    pin_memory=args.pin_memory,
                                    num_workers=args.num_workers,
                                    prefetch_factor=args.prefetch)
        dev_dataloader = DataLoader(dev_dataset,
                                    batch_size=None,
                                    pin_memory=args.pin_memory,
                                    num_workers=args.num_workers,
                                    prefetch_factor=args.prefetch)
        train_joint(model, 
                    optimizer,
                    scheduler,
                    train_dataloader,
                    dev_dataloader,
                    n_epoch)
    return

if __name__ == "__main__":
    main()