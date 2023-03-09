from typing import Tuple, Optional
import librosa

import torch
from torch import nn
import torchaudio
import torchaudio.compliance.kaldi as kaldi

from utils import is_complex, make_pad_mask

class STFT(nn.Module):
    
    def __init__(
        self,
        n_fft: int=512, 
        win_length: int=400,
        hop_length: int=100, 
        center: bool=True,
        normalized: bool = False,
        onesided: bool=True,
        window: Optional[str]="hann"
    ):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.normalized = normalized
        self.onesided = onesided
        if window is not None and not hasattr(torch, f"{window}_window"):
            raise ValueError(f"{window} window is not implemented")
        self.window = window

    def forward(
        self, 
        input: torch.Tensor,
        ilens: torch.Tensor=None
    ):
        if self.window is not None:
            window_func = getattr(torch, f"{self.window}_window")
            window = window_func(
                self.win_length, dtype=input.dtype, device=input.device
            )
        else:
            window = None
        output = torch.stft(
            input,
            n_fft=self.n_fft, 
            win_length=self.win_length,
            hop_length=self.hop_length,
            center=self.center,
            window=window,
            pad_mode='reflect',
            normalized=self.normalized,
            onesided=self.onesided,
            return_complex=True) # (B, F, T)

        output = output.transpose(1, 2) # (B, F, T) -> (B, T, F)
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
            output.masked_fill_(make_pad_mask(olens, output, 1), 0.0)
        else:
            olens = None
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
            win_length=self.win_length,
            hop_length=self.hop_length,
            center=self.center,
            normalized=self.normalized,
            onesided=self.onesided,
            length=ilens.max() if ilens is not None else ilens,
        )
        return wavs, ilens

class ISTFT(nn.Module):
    
    def __init__(self,
                n_fft: int=512,
                win_length: int=400,
                hop_length: int=100,
                center: bool=True,
                normalized: bool = False,
                onesided: bool=True,
                window: Optional[str]="hann"):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.window = window
        self.center = center
        self.normalized = normalized
        self.onesided = onesided

        self.istft = torch.functional.istft
    
    def forward(self,
                input: torch.Tensor,
                ilens: torch.Tensor):
        if self.window is not None:
            window_func = getattr(torch, f"{self.window}_window")
            if is_complex(input):
                datatype = input.real.dtype
            else:
                datatype = input.dtype
            window = window_func(
                self.win_length, dtype=datatype, device=input.device
            )
        else:
            window = None

        if is_complex(input):
            input = torch.stack([input.real, input.imag], dim=-1)
        elif input.shape[-1] != 2:
            raise TypeError("Invalid input type")
        input = input.transpose(1, 2)
        
        wavs = self.istft(
            input,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=self.center,
            normalized=self.normalized,
            onesided=self.onesided,
            length=ilens.max() if ilens is not None else ilens,
        )
        return wavs, ilens

class KaldiFbank(nn.Module):

    def __init__(self,
                n_mels: int=80,
                frame_len: int=25,
                frame_shift: int=10,
                dither: float=0.0):
        super().__init__()
        self.n_mels = n_mels
        self.frame_len = frame_len
        self.frame_shift = frame_shift
        self.dither = dither

    def forward(self, 
                speech: torch.Tensor,
                sample_rate: int=16000):
        mat = kaldi.fbank(speech,
                        num_mel_bins=self.n_mels,
                        frame_length=self.frame_len,
                        frame_shift=self.frame_shift,
                        dither=self.dither,
                        energy_floor=0.0,
                        sample_frequency=sample_rate)
        return mat

class FBank(torch.nn.Module):
    """Convert STFT to fbank feats

    The arguments is same as librosa.filters.mel

    Args:
        fs: number > 0 [scalar] sampling rate of the incoming signal
        n_fft: int > 0 [scalar] number of FFT components
        n_mels: int > 0 [scalar] number of Mel bands to generate
        fmin: float >= 0 [scalar] lowest frequency (in Hz)
        fmax: float >= 0 [scalar] highest frequency (in Hz).
            If `None`, use `fmax = fs / 2.0`
        htk: use HTK formula instead of Slaney
    """

    def __init__(
        self,
        fs: int=16000,
        n_fft: int=512,
        win_length: int=400,
        hop_length: int=100,
        n_mels: int=80,
        fmin: float=None,
        fmax: float=None,
        htk: bool=False,
        log_base: float=None,
    ):
        super().__init__()
        self.stft = STFT(n_fft=n_fft,
                        win_length=win_length,
                        hop_length=hop_length)
        fmin = 0 if fmin is None else fmin
        fmax = fs / 2 if fmax is None else fmax
        _mel_options = dict(
            sr=fs,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            htk=htk,
        )
        self.mel_options = _mel_options
        self.log_base = log_base

        # Note(kamo): The mel matrix of librosa is different from kaldi.
        melmat = librosa.filters.mel(**_mel_options)
        # melmat: (D2, D1) -> (D1, D2)
        self.register_buffer("melmat", torch.from_numpy(melmat.T).float())

    def extra_repr(self):
        return ", ".join(f"{k}={v}" for k, v in self.mel_options.items())

    def forward(
        self,
        speech: torch.Tensor,
        ilens: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # feat: (B, T, D1) x melmat: (D1, D2) -> mel_feat: (B, T, D2)
        stft, stft_len =self.stft(speech, ilens)
        input_power = stft.real**2 + stft.imag**2
        mel_feat = torch.matmul(input_power, self.melmat)
        mel_feat = torch.clamp(mel_feat, min=1e-10)

        if self.log_base is None:
            logmel_feat = mel_feat.log()
        elif self.log_base == 2.0:
            logmel_feat = mel_feat.log2()
        elif self.log_base == 10.0:
            logmel_feat = mel_feat.log10()
        else:
            logmel_feat = mel_feat.log() / torch.log(self.log_base)

        # Zero padding
        if ilens is not None:
            logmel_feat = logmel_feat.masked_fill(
                make_pad_mask(ilens, logmel_feat, 1), 0.0
            )
        else:
            ilens = stft.new_full(
                [stft.size(0)], fill_value=stft.size(1), dtype=torch.long
            )
        return logmel_feat, stft_len

feature_classes = dict(
    stft=STFT,
    istft=ISTFT,
    fbank=FBank,
    kaldi_fbank=KaldiFbank
)