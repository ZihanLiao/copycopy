import torch
import torch.nn.functional as F
# from legos.enh.loss import *

def si_snr(source, estimate_source, eps=1e-5):
    source = source.squeeze(1)
    estimate_source = estimate_source.squeeze(1)
    B, T = source.size()
    source_energy = torch.sum(source ** 2, dim=1).view(B, 1)  # B , 1
    dot = torch.matmul(estimate_source, source.t())  # B , B
    s_target = torch.matmul(dot, source) / (source_energy + eps)  # B , T
    e_noise = estimate_source - source
    snr = 10 * torch.log10(torch.sum(s_target ** 2, dim=1) / (torch.sum(e_noise ** 2, dim=1) + eps) + eps)  # B , 1
    lo = 0 - torch.mean(snr)
    return lo

class SiSnr(object):
    def __call__(self, source, estimate_source):
        return si_snr(source, estimate_source)

def phasen(source_cspec, estimate_cspec, feat_dim, power):
    """
    https://github.com/huyanxin/phasen/blob/master/model/phasen.py
    
    PHASEN: A Phase-and-Harmonics-Aware Speech Enhancement Network
    https://arxiv.org/pdf/1911.04697.pdf

        Args:
            source_cpsec: STFT (real, imag) shape: (B, F*2,T)
            estimate_cspec: STFT (real, imag) shape: (B, F*2,T)
            feat_dim: n_fft 
            power: power for doing compression
            
    """
    b, d, t = estimate_cspec.size()
    src_spec_mag = torch.sqrt(
                                source_cspec[:, :feat_dim, :]**2
                                +source_cspec[:, feat_dim:, :]**2
                               )
    est_spec_mag = torch.sqrt(
                                estimate_cspec[:, :feat_dim, :]**2
                                +estimate_cspec[:, feat_dim:, :]**2
                               )
    src_spec_mag_cprs = src_spec_mag**power
    est_spec_mag_cprs = est_spec_mag**power

    amp_loss = F.mse_loss(src_spec_mag_cprs, est_spec_mag_cprs)*d
    compress_coeff = (src_spec_mag_cprs/(1e-8+src_spec_mag)).repeat(1,2,1)
    phase_loss = F.mse_loss(
                                src_spec_mag_cprs*compress_coeff,
                                est_spec_mag_cprs*compress_coeff
                            )*d
    all_loss = amp_loss*0.5 + phase_loss*0.5
    return all_loss, amp_loss, phase_loss

class Phasen(object):
    def __call__(self, 
                source_cspec, 
                estimate_cspec, 
                feat_dim=256, 
                power=0.3):
        return phasen(source_cspec, estimate_cspec, feat_dim, power)

loss_classes = dict(
    si_snr=SiSnr,
    phasen=Phasen,
)