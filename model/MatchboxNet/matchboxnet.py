import torch
from torch import nn
from typeguard import check_argument_types
from typing import Optional
import torch.nn.functional as F
from model.MatchboxNet.block import Block

class MatchboxNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        B: int,
        R: int,
        C: int,
        kernel_sizes: Optional[list]
        ):
        assert check_argument_types()
        super().__init__()
        if not kernel_sizes:
            kernel_sizes = [k*2+11 for k in range(1,5+1)]
        else:
            assert len(kernel_sizes) == 5

        self.prologue_conv1 = nn.Conv1d(input_dim, 128, kernel_size=11, stride=2)
        self.prologue_bnorm1 = nn.BatchNorm1d(128)

        self.block = Block
        self.blocks = nn.ModuleList()

        self.blocks.append(self.block(128, C, kernel_sizes[0], R=R))

        for i in range(1, B):
            self.blocks.append(self.block(C, C, kernel_size=kernel_sizes[i], R=R))

        self.epilogue_conv1 = nn.Conv1d(C, 128, kernel_size=29, dilation=2)
        self.epilogue_bnorm1 = nn.BatchNorm1d(128)

        self.epilogue_conv2 = nn.Conv1d(128, 128, kernel_size=1)
        self.epilogue_bnorm2 = nn.BatchNorm1d(128)

        self.epilogue_conv3 = nn.Conv1d(128, output_dim, kernel_size=1)
        self.epilogue_adaptivepool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # prologue block
        x = self.prologue_conv1(x)
        x = self.prologue_bnorm1(x)
        x = F.relu(x)

        # intermediate blocks
        for layer in self.blocks:
            x = layer(x)

        # epilogue blocks
        x = self.epilogue_conv1(x)
        x = self.epilogue_bnorm1(x)

        x = self.epilogue_conv2(x)
        x = self.epilogue_bnorm2(x)

        x = self.epilogue_conv3(x)
        x = self.epilogue_adaptivepool(x)
        x = x.squeeze(2) # (N, 30, 1) > (N, 30)
        x = F.softmax(x, dim=1) # softmax across classes and not batch
        
        return x
