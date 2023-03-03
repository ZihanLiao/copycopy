from model.dccrn import DCCRN
from legos.enh.loss import SiSnr
from tool.feature_extractor import STFT

# Collaborate with wenet
import wenet.utils.init_model as wenet_init_model
# from legos.general.model_pattern import JointModel
model_zoo = {
    'dccrn': DCCRN
}

def init_model(config: dict, joint: bool = False):
    if not joint:
        model_name = config['frontend']['model']['name']
        model_type = config['frontend']['model']['type']
        hyper_params = config['frontend']['model']['parameter']
        model_class = model_zoo.get(model_name, None)
        if not model_class:
            raise TypeError("ERROR: {} is not implemented yet".format(model_name))
        model = model_class(**hyper_params)
    else:
        pass
    return model

def init_loss(loss_name: str):
    if loss_name == "si_snr":
        loss = SiSnr
    return loss
    
def init_scheduler(config: dict):
    scheduler_name = config.get("name")
    if scheduler_name == ""

    return

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
    import yaml
    with open("/Users/marlowe/workspace/myownspeechtoolbox/shell/enh/conf/train_conformer.yaml", 'r') as f:
        conformer_config = yaml.load(f, Loader=yaml.FullLoader)
    conformer_config['input_dim'] = 80
    conformer_config['output_dim'] = 100
    conformer_config['cmvn_file'] = './cmvn'
    conformer_config['is_json_cmvn'] = True
    conformer = wenet_init_model.init_model(conformer_config)