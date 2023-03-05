from typing import Tuple, Optional, List, Union, Dict
from typeguard import check_argument_types

import torch
from torch import nn

from model.dccrn import DCCRN
from wenet.transformer.asr_model import WenetASRModel
from legos.enh.loss import SiSnr
from tool.feature_extractor import STFT, FBank


class EnhAsr(nn.Module):
    
    def __init__(self,
                enh_model: Union[DCCRN, None],
                asr_model: Union[WenetASRModel, None],
                enh_feature_extractor: nn.Module,
                asr_feature_extractor: nn.Module,
                calc_enh_loss: bool,
                enh_loss_type: Optional[SiSnr],
                bypass_enh_prob: float = 0,
                #asr_loss_type: Union[CTC, None]=CTC
                ):
        assert check_argument_types()
        assert calc_enh_loss is True and enh_loss_type is not None
        self.enh_model = enh_model
        self.asr_model = asr_model
        self.enh_feature_extractor = enh_feature_extractor
        self.asr_feature_extractor = asr_feature_extractor
        self.calc_enh_loss = calc_enh_loss
        self.enh_loss_type = enh_loss_type
        self.bypass_enh_prob = bypass_enh_prob
        # self.asr_loss_type = asr_loss_type

    def forward(self,
                speech_mix: torch.Tensor,
                speech_length: torch.Tensor,
                target: str,  
                target_length: torch.Tensor,
                **kwargs):
        # speech to TF feature
        feats, feats_len = self.enh_feature_extractor(speech_mix, speech_length)
        # enh model forward
        enh_feats, enh_feats_len, enh_others = self.enh_model(feats, feats_len)
        # TF feature to speech
        enh_speech, enh_speech_len = self.enh_feature_extractor.inverse(enh_feats, enh_feats_len)
        batch_size = speech_mix.shape[0]

        enh_loss = None
        ref_speech = None
        if self.calc_enh_loss:
            assert "speech_ref" in kwargs
            ref_speech = kwargs["speech_ref"]
            ref_speech.unsqueeze(0)

            # Calculate enhancement loss
            enh_loss = self.enh_loss_type(ref_speech, enh_speech)

        asr_feats, asr_feats_len = self.asr_feature_extractor(enh_speech, enh_speech_len)
        loss_dict = self.asr_model(asr_feats, asr_feats_len,
                                    target, target_length)
        asr_loss = loss_dict['loss']
        loss = None
        loss = enh_loss + asr_loss if self.calc_enh_loss else asr_loss
        return loss

if __name__ == '__main__':
    
    



