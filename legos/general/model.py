from typing import Tuple, Optional, List, Union, Dict
from typeguard import check_argument_types

import torch
from torch import nn

from model.dccrn import DCCRN
import wenet.transformer.asr_model as WenetTransfromer
from legos.general.loss import SiSnr
from tool.feature_extractor import STFT, FBank

WenetASRModel = WenetTransfromer.ASRModel

class EnhAsr(nn.Module):
    
    def __init__(self,
                enh_model: Union[DCCRN, None],
                asr_model: Union[WenetASRModel, None],
                enh_feature_extractor: Tuple[nn.Module, nn.Module],
                asr_feature_extractor: Tuple[nn.Module, Optional[nn.Module]],
                calc_enh_loss: bool,
                enh_loss_type: Optional[SiSnr],
                bypass_enh_prob: float = 0,
                #asr_loss_type: Union[CTC, None]=CTC
                ):
        assert check_argument_types()
        assert calc_enh_loss is True and enh_loss_type is not None
        super().__init__()
        self.enh_model = enh_model
        self.asr_model = asr_model
        self.enh_feature_extractor_for, self.enh_feature_extractor_inv = enh_feature_extractor
        self.enh_inverse = True if self.enh_feature_extractor_inv is not None else False
        self.asr_feature_extractor_for, self.asr_feature_extractor_inv = asr_feature_extractor
        self.asr_inverse = True if self.asr_feature_extractor_inv is not None else False
        self.asr_feature_extractor = asr_feature_extractor
        self.calc_enh_loss = calc_enh_loss
        self.enh_loss_type = enh_loss_type
        self.bypass_enh_prob = bypass_enh_prob
        # self.asr_loss_type = asr_loss_type

    def forward(self,
                speech_mix: torch.Tensor,
                speech_ref: Optional[torch.Tensor],
                speech_length: torch.Tensor,
                target: torch.Tensor,  
                target_length: torch.Tensor) -> torch.Tensor:
        """ Forward
            Args:
                speech_mix: mix of reference speech and noise (B, T, channel)
                speech_ref: reference speech (B, T, channel)
                speech_length: speech length, (B, )
                target: ASR target
                target_length: ASR target length
        """
        speech_mix = speech_mix.squeeze(2) # (B, T)
        speech_ref = speech_ref.squeeze(2) # (B, T)
        # speech to TF feature
        feats, feats_len = self.enh_feature_extractor_for(speech_mix, speech_length)
        # enh model forward
        enh_feats, enh_feats_len, enh_others = self.enh_model(feats, feats_len)
        # TF feature to speech
        enh_speech, enh_speech_len = self.enh_feature_extractor_inv(enh_feats, speech_length)
        # enh_speech, enh_speech_len = self.enh_feature_extractor.inverse(enh_feats, enh_feats_len)
        
        enh_loss = None
        if self.calc_enh_loss:
            speech_ref.unsqueeze(0)
            # Calculate enhancement loss
            enh_loss = self.enh_loss_type(speech_ref, enh_speech)

        asr_feats, asr_feats_len = self.asr_feature_extractor_for(enh_speech, speech_length)
        # print("asr_feats: ", asr_feats)
        # print("asr_feats_len: ",asr_feats_len)
        # print("target: ",target)
        # print("target_length: ", target_length)
        loss_dict = self.asr_model(asr_feats, asr_feats_len,
                                    target, target_length)
        asr_loss = loss_dict['loss']
        loss = None
        loss = enh_loss + asr_loss if self.calc_enh_loss else asr_loss
        return enh_loss, asr_loss, loss




