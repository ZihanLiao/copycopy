import torch
from torch import nn
# import torchaudio_contrib as audio_nn
import torchaudio
from torchaudio.compliance.kaldi import fbank
from utils import istft, is_complex

class STFT(nn.Module):
    
    def __init__(
        self,
        n_fft: int=512, 
        hop_length: int=100, 
        win_length: int=400,
        center: bool=True,
        normalized: bool = False,
        onesided: bool=True
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.center = center
        self.normalize = normalized
        self.onesided = onesided

    def forward(
        self, 
        input: torch.Tensor,
        ilens: torch.Tensor=None
    ):
        output = torch.stft(
            input,
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            win_length=self.win_length, 
            return_complex=True)
        if ilens is not None:
            if self.center:
                pad = self.n_fft // 2
                ilens = ilens + 2 * pad
            olens = (
                    torch.div(
                        ilens - self.n_fft, self.hop_length, rounding_mode="trunc"
                    )
                    + 1
                )
        return output, olens
    
    def inverse(
        self, 
        input: torch.Tensor,
        ilens: torch.Tensor=None
    ):
        """Inverse STFT.

        Args:
            input: Tensor(batch, T, F, 2) or ComplexTensor(batch, T, F)
            ilens: (batch,)
        Returns:
            wavs: (batch, samples)
            ilens: (batch,)
        """
        istft = torch.functional.istft
        if is_complex(input):
            input = torch.stack([input.real, input.imag], dim=-1)
        elif input.shape[-1] != 2:
            raise TypeError("Invalid input type")
        input = input.transpose(1, 2)

        wavs = istft(
            input,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=self.center,
            normalized=self.normalized,
            onesided=self.onesided,
            length=ilens.max() if ilens is not None else ilens,
        )
        return wavs, ilens

# class ISTFT(nn.Module):
#     def __init__(
#         self, 
#         n_fft: int=512, 
#         hop_length: int=256, 
#         win_length: int=512
#         ):
#         super().__init__()
#         self.n_fft, self.hop_length, self.win_length = n_fft, hop_length, win_length

#     def forward(self, speech):
#         B, C, F, T, D = speech.shape
#         x = speech.view(B, F, T, D)
#         speech_istft = istft(speech, hop_length=self.hop_length, length=600)
#         return speech_istft.view(B, C, -1)


class Fbank(nn.Module):

    def __init__(
        self,
        num_mel_bins: int=23,
        frame_length=25,
        frame_shift=10,
        dither=0.0,
        sample_rate=16000
    ):
        super().__init__()
        self.num_mel_bins = num_mel_bins
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.dither = dither
        self.sample_rate = sample_rate
        
    def forward(
        self, 
        speech
    ):
        with torch.no_grad():
            mat = fbank(speech,
                      num_mel_bins=self.num_mel_bins,
                      frame_length=self.frame_length,
                      frame_shift=self.frame_shift,
                      dither=self.dither,
                      energy_floor=0.0,
                      sample_frequency=self.sample_rate)
        return mat