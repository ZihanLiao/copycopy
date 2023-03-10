import argparse
import yaml
import re
import logging
import os

import torch
import torch.distributed as dist
from tensorboardX import SummaryWriter

from torch.utils.data import DataLoader

from tool.init_everything import (
    init_model, 
    init_scheduler,
    init_optimizer
    )
from tool.dataset import Dataset
from tool.utils import read_symbol_table
from tool.executor import Executor
from tool.checkpoint import (load_checkpoint, save_checkpoint,
                                    load_trained_modules)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", required=True, type=str, help="train data list")
    parser.add_argument("--dev_data", required=True, type=str, help="dev data list")
    parser.add_argument("--model_dir", required=True, help="save model to model_dir")
    parser.add_argument("--checkpoint", default=None, help="model checkpoint")
    parser.add_argument("--config", required=True, type=str, help="config path")
    parser.add_argument("--format", default="raw", type=str, help="data format")
    parser.add_argument("--symbol_table", type=str, default=None)
    parser.add_argument('--tensorboard_dir',
                        default='tensorboard',
                        help='tensorboard log dir')
    parser.add_argument("--ddp.rank",
                        dest='rank',
                        default=-1,
                        type=int,
                        help='''number of total processes/gpus for
                        distributed training''')
    parser.add_argument('--ddp.world_size',
                        dest='world_size',
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
    parser.add_argument('--pin_memory',
                        action='store_true',
                        default=False,
                        help='Use pinned memory buffers used for reading')
    parser.add_argument('--cmvn', default=None, help='global cmvn file')           
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG, 
                        format='%(asctime)s %(levelname)s %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    # Set random seed
    torch.manual_seed(777)
    
    distributed = args.world_size > 1
    if distributed:
        logging.info('training on multiple gpus, this gpu {}'.format(args.gpu))
        dist.init_process_group(args.dist_backend,
                                init_method=args.init_method,
                                world_size=args.world_size,
                                rank=args.rank)

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    epoch_num = config.pop('epoch_num', 99)
    mission = config.pop('mission')
    print("current mission is: {}".format(mission))

    symbol_table = None
    if 'asr' in mission:
        assert args.symbol_table
        symbol_table = read_symbol_table(args.symbol_table)

    # Init optimizer
    optimizer_config = config.pop('optimizer')
    optimizer = init_optimizer(optimizer_config)
    # Init scheduler
    scheduler_config = config.pop('scheduler')
    scheduler = init_scheduler(scheduler_config)
        
    # Init model
    model_config = config
    config['output_dim'] = len(symbol_table.items())
    model = init_model(model_config, mission)
    num_params = sum(p.numel() for p in model.parameters())
    print('the number of model params: {:,d}'.format(num_params))

    dataset_config = config.pop('dataset_conf', {})
    
            
    if args.checkpoint is not None:
        infos = load_checkpoint(model, args.checkpoint)
    else:
        infos = {}
    start_epoch = infos.get('epoch', -1) + 1
    cv_loss = infos.get('cv_loss', 0.0)
    step = infos.get('step', -1)

    model_dir = args.model_dir
    
    writer = None
    if args.rank == 0:
        os.makedirs(model_dir, exist_ok=True)
        exp_id = os.path.basename(model_dir)
        writer = SummaryWriter(os.path.join(args.tensorboard_dir, exp_id))

    if distributed:
        assert (torch.cuda.is_available())

        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(
            model, find_unused_parameters=True)
    else:
        use_cuda = args.gpu >= 0 and torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')
        model = model.to(device)

    final_epoch = None
    config['rank'] = args.rank
    config['is_distributed'] = distributed

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
    executor = Executor()
    executor.step = step
    scheduler.set_step(step)

    scaler = None
    if start_epoch == 0 and args.rank == 0:
        save_model_path = os.path.join(model_dir, 'init.pt')
        save_checkpoint(model, save_model_path)

    for epoch in range(start_epoch, epoch_num):
        train_dataset.set_epoch(epoch)
        lr = optimizer.param_groups[0]['lr']
        logging.info('Epoch {} TRAIN info lr {}'.format(epoch, lr))
        executor.train(model, optimizer, scheduler, train_dataloader, device,
                       writer, config, scaler)
        total_loss, num_seen_utts = executor.cv(model, dev_dataloader, device, config)
        cv_loss = total_loss / num_seen_utts
        logging.info('Epoch {} CV info cv_loss {}'.format(epoch, cv_loss))
        if args.rank == 0:
            save_model_path = os.path.join(model_dir, '{}.pt'.format(epoch))
            save_checkpoint(
                model, save_model_path, {
                    'epoch': epoch,
                    'lr': lr,
                    'cv_loss': cv_loss,
                    'step': executor.step
                })
            writer.add_scalar('epoch/cv_loss', cv_loss, epoch)
            writer.add_scalar('epoch/lr', lr, epoch)
        final_epoch = epoch

    if final_epoch is not None and args.rank == 0:
        final_model_path = os.path.join(model_dir, 'final.pt')
        os.remove(final_model_path) if os.path.exists(final_model_path) else None
        os.symlink('{}.pt'.format(final_epoch), final_model_path)
        writer.close()
    return

if __name__ == "__main__":
    main()