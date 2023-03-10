# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
from typing import Optional
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset

import tool.processor as processor
from tool.utils import read_lists
from tool.lmbd_data import LmdbData

class Processor(IterableDataset):
    def __init__(self, source, f, *args, **kw):
        assert callable(f)
        self.source = source
        self.f = f
        self.args = args
        self.kw = kw

    def set_epoch(self, epoch):
        self.source.set_epoch(epoch)

    def __iter__(self):
        """ Return an iterator over the source dataset processed by the
            given processor.
        """
        assert self.source is not None
        assert callable(self.f)
        return self.f(iter(self.source), *self.args, **self.kw)

    def apply(self, f):
        assert callable(f)
        return Processor(self, f, *self.args, **self.kw)


class DistributedSampler:
    def __init__(self, shuffle=True, partition=True):
        self.epoch = -1
        self.update()
        self.shuffle = shuffle
        self.partition = partition

    def update(self):
        assert dist.is_available()
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.worker_id = 0
            self.num_workers = 1
        else:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers
        return dict(rank=self.rank,
                    world_size=self.world_size,
                    worker_id=self.worker_id,
                    num_workers=self.num_workers)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def sample(self, data):
        """ Sample data according to rank/world_size/num_workers

            Args:
                data(List): input data list

            Returns:
                List: data list after sample
        """
        data = list(range(len(data)))
        # TODO(Binbin Zhang): fix this
        # We can not handle uneven data for CV on DDP, so we don't
        # sample data by rank, that means every GPU gets the same
        # and all the CV data
        if self.partition:
            if self.shuffle:
                random.Random(self.epoch).shuffle(data)
            data = data[self.rank::self.world_size]
        data = data[self.worker_id::self.num_workers]
        return data


class DataList(IterableDataset):
    def __init__(self, lists, shuffle=True, partition=True):
        self.lists = lists
        self.sampler = DistributedSampler(shuffle, partition)

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

    def __iter__(self):
        sampler_info = self.sampler.update()
        indexes = self.sampler.sample(self.lists)
        for index in indexes:
            # yield dict(src=src)
            data = dict(src=self.lists[index])
            data.update(sampler_info)
            yield data


def Dataset(data_type,
            data_list_file,
            symbol_table: Optional[dict],
            conf,
            bpe_model=None,
            non_lang_syms=None,
            partition=True):
    """ Construct dataset from arguments

        We have two shuffle stage in the Dataset. The first is global
        shuffle at shards tar/raw file level. The second is global shuffle
        at training samples level.

        Args:
            data_type(str): raw/shard
            bpe_model(str): model for english bpe part
            partition(bool): whether to do data partition in terms of rank
    """
    assert data_type in ['raw', 'shard']
    lists = read_lists(data_list_file)
    shuffle = conf.get('shuffle', True)
    dataset = DataList(lists, shuffle=shuffle, partition=partition)
    if data_type == 'shard':
        dataset = Processor(dataset, processor.url_opener)
        dataset = Processor(dataset, processor.tar_file_and_group)
    else:
        dataset = Processor(dataset, processor.parse_raw)

    dataset = Processor(dataset, processor.tokenize, symbol_table, bpe_model,
                        non_lang_syms, conf.get('split_with_space', False))
    filter_conf = conf.get('filter_conf', {})
    dataset = Processor(dataset, processor.filter, **filter_conf)

    resample_conf = conf.get('resample_conf', {})
    dataset = Processor(dataset, processor.resample, **resample_conf)

    speed_perturb = conf.get('speed_perturb', False)
    if speed_perturb:
        dataset = Processor(dataset, processor.speed_perturb)
    
    add_noise = conf.get('add_noise', False)
    if add_noise:
        noise_conf = conf.get('noise_conf', {})
        noise_lmdb = noise_conf.get('noise_source', None)
        noise_prob = noise_conf.get('aug_prob', 0)
        if noise_lmdb and noise_prob > 0:
            noise_data = LmdbData(noise_lmdb)
            dataset = Processor(dataset, processor.add_noise, noise_data, noise_prob)

    add_reverb = conf.get('add_reverb', False)
    if add_reverb:
        reverb_conf = conf.get('reverb_conf', {})
        reverb_lmdb = reverb_conf.get('reverb_source', None)
        reverb_prob = reverb_conf.get('aug_prob', 0)
        if reverb_lmdb and reverb_prob > 0:
            reverb_data = LmdbData(reverb_lmdb)
            dataset = Processor(dataset, processor.add_reverb, reverb_data, reverb_prob)
    
    yield_wav = conf.get('yield_wav', False)
    if not yield_wav:
        feats_type = conf.get('feats_type', 'fbank')
        assert feats_type in ['fbank', 'mfcc']
        if feats_type == 'fbank':
            fbank_conf = conf.get('fbank_conf', {})
            dataset = Processor(dataset, processor.compute_fbank, **fbank_conf)
        elif feats_type == 'mfcc':
            mfcc_conf = conf.get('mfcc_conf', {})
            dataset = Processor(dataset, processor.compute_mfcc, **mfcc_conf)

        spec_aug = conf.get('spec_aug', True)
        spec_sub = conf.get('spec_sub', False)
        spec_trim = conf.get('spec_trim', False)
        if spec_aug:
            spec_aug_conf = conf.get('spec_aug_conf', {})
            dataset = Processor(dataset, processor.spec_aug, **spec_aug_conf)
        if spec_sub:
            spec_sub_conf = conf.get('spec_sub_conf', {})
            dataset = Processor(dataset, processor.spec_sub, **spec_sub_conf)
        if spec_trim:
            spec_trim_conf = conf.get('spec_trim_conf', {})
            dataset = Processor(dataset, processor.spec_trim, **spec_trim_conf)
    
    if shuffle:
        shuffle_conf = conf.get('shuffle_conf', {})
        dataset = Processor(dataset, processor.shuffle, **shuffle_conf)

    sort = conf.get('sort', True)
    if sort:
        sort_conf = conf.get('sort_conf', {})
        dataset = Processor(dataset, processor.sort, **sort_conf)

    batch_conf = conf.get('batch_conf', {})
    dataset = Processor(dataset, processor.batch, **batch_conf)
    if not yield_wav:
        dataset = Processor(dataset, processor.padding)
    else:
        dataset = Processor(dataset, processor.padding_raw_wav)
    return dataset


if __name__ == '__main__':
    from tool.utils import read_symbol_table
    import json
    with open("conf/enh_dccrn.json", 'r') as f:
        conf = json.load(f)
    dataset_conf = conf['dataset_conf']
    symbol_table = read_symbol_table("data/local/dict/lang_char.txt")
    dataset = Dataset('raw', 'data/local/train/train.list', symbol_table,
                    dataset_conf)

    from tool.init_everything import (
    init_model, 
    init_scheduler,
    init_optimizer
    )
    fp = open("/Users/marlowe/workspace/myownspeechtoolbox/shell/enh/conf/enh_dccrn.json", 'r')
    config = json.load(fp)
    fp.close()
    model = init_model(config, 'enh_asr')
    for x in dataset:
        (sorted_key, padding_wav, padding_label, padding_wav_mix, wav_lengths, label_lengths) = x
        enh_loss, asr_loss, loss = model(speech_mix=padding_wav_mix,
                    speech_ref=padding_wav,
                    speech_length=wav_lengths,
                    target=padding_label,
                    target_length=label_lengths)
        loss.backward()
        break
