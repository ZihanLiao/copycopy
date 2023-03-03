from torch import nn
import torch.nn.functional as F
from typeguard import check_argument_types

class SubBlock(nn.Module):
    """Subblock for the MatchboxNet
        Args:
        
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int
    ):
        assert check_argument_types()
        super().__init__()
        self.depthwise_conv = nn.Conv1d(in_channels, in_channels, kernel_size, groups=in_channels, padding='same')
        self.pointwise_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bnorm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout()

    def forward(self, x, residual=None):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.bnorm(x)
        
        if residual is not None:
            x = x + residual

        x = F.relu(x)
        x = self.dropout(x)
        return x
