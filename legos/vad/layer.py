from torch import nn
import torch.nn.functional as F
from typeguard import check_argument_types

class MatchboxNetSubBlock(nn.Module):
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

class MatchboxNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        R : int = 1
    ):
        assert check_argument_types()
        super().__init__()
        
        self.residual_pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.residual_batchnorm = nn.BatchNorm1d(out_channels)
        self.sub_block = MatchboxNetSubBlock

        self.sub_blocks = nn.ModuleList()
        self.sub_blocks.append(self.sub_block(in_channels, out_channels, kernel_size))
        for _ in range(R-1):
            self.sub_blocks.append(self.sub_block(out_channels, out_channels, kernel_size))

    def forward(self, x):
        residual = self.residual_pointwise(x)
        residual = self.residual_batchnorm(residual)
        for i, layer in enumerate(self.sub_blocks):
            if (i+1) == len(self.sub_blocks): # compute the residual in the final sub-block
                x = layer(x, residual)
            else:
                x = layer(x)

        return x