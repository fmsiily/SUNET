""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from unet import spiking_activation
SpikeRelu = spiking_activation.SpikeRelu
from unet.unet_parts_spiking import *


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self,  thresholds, device, clamp_slope, reset,in_channels, out_channels,layer_num, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            SpikeRelu(thresholds[layer_num], layer_num, clamp_slope, device, reset),
            # nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            SpikeRelu(thresholds[layer_num+1], layer_num+1, clamp_slope, device, reset),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, thresholds, device, clamp_slope, reset,in_channels, out_channels,layer_num):
        super().__init__()
        self.avgpool_conv = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            SpikeRelu(thresholds[layer_num], layer_num, clamp_slope, device, reset),
            DoubleConv(thresholds, device, clamp_slope, reset,in_channels, out_channels,layer_num+1)
        )

    def forward(self, x):
        return self.avgpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self,thresholds, device, clamp_slope, reset, in_channels, out_channels, layer_num,bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(thresholds, device, clamp_slope, reset,in_channels, out_channels, layer_num,in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(thresholds, device, clamp_slope, reset,in_channels, out_channels,layer_num)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, thresholds, device, clamp_slope, reset,in_channels, out_channels,layer_num):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.spirelu = SpikeRelu(thresholds[layer_num], layer_num, clamp_slope, device, reset)


    def forward(self, x):
        x = self.conv(x)
        x = self.spirelu(x)
        return x
