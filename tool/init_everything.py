import os
import yaml

from model.dccrn import DCCRN
from legos.enh.loss import SiSnr
from legos.general.loss import loss_classes
from legos.general.scheduler import scheduler_classes
from legos.general.optimizer import optim_classes
from legos.general.model import EnhAsr
from tool.feature_extractor import feature_classes

# Collaborate with wenet
import wenet.utils.init_model as wenet_init_model
import wenet.transformer.asr_model as WenetTransfromer
# from legos.general.model_pattern import JointModel
WenetASRModel = WenetTransfromer.ASRModel

model_zoo = {
    'dccrn': DCCRN,
    'wenet_conformer':WenetASRModel
}


def init_model(config: dict, mission: str):

    if len(mission.split('_')) == 1:
        model_name = config['frontend']['model']['name']
        model_type = config['frontend']['model']['type']
        hyper_params = config['frontend']['model']['parameter']

        if not config.get('integrate_feat_extractor'):
            feat_config = config['frontend']['feature']
            feat_extractor = init_feature_extractor(feat_config)
            hyper_params['enh_feature_extractor'] = feat_extractor
        model_class = model_zoo.get(model_name, None)
        if not model_class:
            raise TypeError("ERROR: {} is not implemented yet".format(model_name))
        model = model_class(**hyper_params)
        return model
    elif len(mission.split('_')) == 2:
        frontend_type = config['frontend']['type']
        # init frontend feature extractor
        frontend_feat_config = config['frontend']['feature']
        frontend_feat_extractor = init_feature_extractor(frontend_feat_config)

        # init frontend model
        frontend_model_name = config['frontend']['model']['name']
        frontend_hyper_params = config['frontend']['model']['parameter']
        frontend_model_class = model_zoo.get(frontend_model_name, None)
        frontend_model = frontend_model_class(**frontend_hyper_params)
        
        

        # init frontend loss
        frontend_loss_specified = False
        if "loss" in config['frontend']:
            frontend_loss_config = config['frontend']['loss']
            frontend_loss = init_loss(frontend_loss_config)
            frontend_loss_specified = True

        downstream_type = config['downstream']['type']
        # init downstream feature extractor
        downstream_feat_config = config['downstream']['feature']
        downstream_feat_extractor = init_feature_extractor(downstream_feat_config)

        # init downstream model
        use_wenet_downstream = config['downstream'].get('use_wenet_downstream', False)
        if not use_wenet_downstream:
            downstream_model_name = config['downstream']['model']['name']
            downstream_model_params = config['downstream']['model']['parameter']
            downstream_model_class = model_zoo.get(downstream_model_name, None)
            downstream_model = downstream_model_class(**downstream_model_params)
            # init downstream loss
            downstream_loss_specified = False
            if "loss" in config['downstream']:
                downstream_loss_config = config['downstream']['loss']
                downstream_loss = init_loss(downstream_loss_config)
                downstream_loss_specified = True
        else:
            assert 'wenet_downstream_config' in config['downstream']
            assert os.path.exists(config['downstream']['wenet_downstream_config'])
            with open(config['downstream']['wenet_downstream_config'], 'r') as f:
                wenet_downstream_config = yaml.load(f, Loader=yaml.FullLoader)
            downstream_model = wenet_init_model.init_model(wenet_downstream_config)

        if frontend_type == "enh" and downstream_type == "asr":
            model = EnhAsr(
                enh_model = frontend_model,
                asr_model = downstream_model,
                enh_feature_extractor=frontend_feat_extractor,
                asr_feature_extractor=downstream_feat_extractor,
                calc_enh_loss=True,
                enh_loss_type = frontend_loss if 
                            frontend_loss_specified else None,
                bypass_enh_prob=0,
            )
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return model

def init_feature_extractor(config: dict):
    feature_name = config.pop('name')
    feature = feature_classes.get(feature_name, None)
    if feature_name == "stft":
        feature_inv = feature_classes.get("istft", None)
    return (feature(**config), feature_inv(**config)) \
                if feature_name == "stft" else (feature(**config), None)

def init_loss(config: dict):
    loss_name = config.pop("name")
    loss = loss_classes.get(loss_name, None)
    return loss(**config)
    
def init_optimizer(config: dict):
    optimizer_name = config.pop("name")
    optimizer = optim_classes.get(optimizer_name, None)
    return optimizer(**config["parameter"])

def init_scheduler(config: dict):
    scheduler_name = config.pop("name")
    scheduler = scheduler_classes.get(scheduler_name, None)
    return scheduler(**config["parameter"])

if __name__ == '__main__':
    import json
    from collections import defaultdict
    # config = defaultdict()
    # config['joint'] = False
    # config['frontend'] = {}
    # config['frontend']['feature'] = {
    #     'type': "stft",
    #     'n_fft': 512,
    #     'hop_length': 100,
    #     'win_length': 400,
    # }

    # config['frontend']['model'] = {
    #     'name': "dccrn",
    #     'type': "denoise"
    # }
    # config['frontend']['model']['parameter'] = {     
    #     'input_dim': 512,
    #     'num_spk': 1,
    #     'rnn_layer': 2,
    #     'rnn_units': 256,
    #     'masking_mode': 'E',
    #     'use_clstm': False,
    #     'bidirectional': False,
    #     'use_cbn': False,
    #     'kernel_size': 5,
    #     'kernel_num': [32, 64, 128, 256, 256, 256],
    #     'use_builtin_complex': True,
    #     'use_noise_mask': False
    # }
    # with open("/Users/marlowe/workspace/myownspeechtoolbox/shell/enh/conf/enh_dccrn.json", 'w') as f:
    #     json.dump(config, f, ensure_ascii=False, allow_nan=True, indent=2)
    # fp = open("/Users/marlowe/workspace/myownspeechtoolbox/shell/enh/conf/enh_dccrn.json", 'r')
    # config = json.load(fp)
    # joint = config.get('joint', False)
    # config.pop('joint')
    # dccrn = init_model(config, joint)
    # import yaml
    # with open("conf/train_conformer.yaml", 'r') as f:
    #     conformer_config = yaml.load(f, Loader=yaml.FullLoader)
    # conformer = wenet_init_model.init_model(conformer_config)
    with open("/Users/marlowe/workspace/myownspeechtoolbox/shell/enh/conf/enh_dccrn.json", 'r') as f:
        config = json.load(f)
    model = init_model(config, "enh_asr")
    import torchaudio
    wav, sr = torchaudio.load("/Users/marlowe/workspace/myownspeechtoolbox/shell/enh/positive.wav")
    wav_len = wav.shape[0]
    print(wav_len)
    target = [1,2,1,2]
    target_len = 4
    
    