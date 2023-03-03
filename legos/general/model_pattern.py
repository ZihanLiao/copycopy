import model
from torch import nn
import torch
from torch import nn
from model.dccrn import DCCRN
from legos.enh.loss import SiSnr
from tool.feature_extractor import STFT, Fbank
from typing import Tuple, Optional, List, Union, Dict

class JointModel(nn.Module):
    
    def __init__(self,
                frontend: Union[DCCRN, None],
                downstream: Union[WenetConformer, None],
                frontend_feature_extractor: nn.Module=STFT,
                downstream_feature_extractor: nn.Module=Fbank,
                calc_frontend_loss: bool=True,
                frontend_loss_type: nn.Module=SiSnr,
                bypass_frontend_prob: float = 0,):
        self.frontend = frontend
        self.downstream = downstream
        self.frontend_feature_extractor = frontend_feature_extractor
        self.bypass_enh_prob = bypass_frontend_prob
        self.calc_enh_loss = calc_frontend_loss
        self.enh_loss_type = frontend_loss_type

    def forward(self,
                speech_mix: torch.Tensor, 
                speech_lengths: torch.Tensor,
                additional: Optional[Dict]):
        frontend_feature = self.frontend_feature_extractor(speech_mix, speech_length)
        speech_pre, speech_pre_length, others = self.frontend()

        return


