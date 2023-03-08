import logging
from contextlib import nullcontext

import torch

def train_single(model,
                optimizer,
                scheduler,
                train_dataloader,
                dev_dataloader,
                **kwargs):
    return

def train_joint(model,
                optimizer,
                scheduler,
                train_dataloader,
                dev_dataloader,
                n_epoch,
                **kwargs):
    n_epoch = kwargs.get('epoch_num', 99)
    device = kwargs.get('device', 'cpu')
    is_distributed = kwargs.get('is_distributed', False)
    rank = kwargs.get('rank', 0)
    accum_grad = kwargs.get('accum_grad', 1)
    logging.info('using accumulate grad, new batch size is {} times'
                     ' larger than before'.format(accum_grad))
    model.train()

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_context = model.join
    else:
        model_context = nullcontext

    for current_epoch in range(n_epoch):
        logging.info("INFO: ############ epoch {} ############".format(current_epoch))
        with model_context():
            for batch_idx, batch in enumerate(train_dataloader):
                (sorted_key, padding_wav, padding_label, padding_wav_mix, wav_lengths, label_lengths) = batch
                loss = model(padding_wav_mix,
                             padding_wav,
                             wav_lengths,
                             padding_label,
                             label_lengths)
    return

