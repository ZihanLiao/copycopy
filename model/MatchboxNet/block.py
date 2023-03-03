import torch
from torch import nn
from typeguard import check_argument_types
from model.MatchboxNet.subblock import SubBlock

class Block(nn.Module):
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
        self.sub_block = SubBlock

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