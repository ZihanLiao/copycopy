import torch
import argparse
import json
import re
from tool.model_utils import init_model
from tool.train_misson import train_joint, train_single


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", required=True, type=str, help="train data list")
    parser.add_argument("--dev_data", required=True, type=str, help="dev data list")
    parser.add_argument("--model_dir", required=True, help="save model to model_dir")
    parser.add_argument("--checkpoint", default=None, help="model checkpoint")
    parser.add_argument("--config", required=True, type=str, help="config path")
    parser.add_argument("--mission", 
                        required=True, 
                        type=str, 
                        type=["vad", "enh", "enh_asr"]
                        help="ongoing mission")
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
    joint = config.pop('joint')
    if not joint:
        model_config = config
        model_config["mission"] = args.mission
        model = init_model(model_config)

        train_single(model, loss, scheduler, train_dataloader, dev_dataloader)
    else:
        mission1, mission2 = re.split("_", args.mission)
        model_config = config_dict["model"]
        model_config["mission"] = [mission1, mission2]
        model = init_model(model_config)
        train_joint(model, model_config, train_dataloader, dev_dataloader)

    return

if __name__ == "__main__":
    main()