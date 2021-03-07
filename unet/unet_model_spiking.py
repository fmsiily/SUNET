import torch
import torch.nn as nn

from unet import spiking_activation
SpikeRelu = spiking_activation.SpikeRelu
from unet.unet_parts_spiking import *

class Unet_nobn_spike(nn.Module ):
    def __init__(self,thresholds, device, clamp_slope, reset, n_channels, n_classes, bilinear=True):
        super(Unet_nobn_spike,self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv( thresholds, device, clamp_slope, reset,n_channels, 64,layer_num=0) #layer 0,1
        self.down1 = Down(thresholds, device, clamp_slope, reset,64, 128,layer_num=2) #2.3.4
        self.down2 = Down(thresholds, device, clamp_slope, reset,128, 256,layer_num=5) #5 6 7
        self.down3 = Down(thresholds, device, clamp_slope, reset,256, 512,layer_num=8) #8 9 10
        factor = 2 if bilinear else 1
        self.down4 = Down(thresholds, device, clamp_slope, reset,512, 1024 // factor,layer_num=11) #11 12 13
        self.up1 = Up(thresholds, device, clamp_slope, reset,1024, 512 // factor,bilinear=True, layer_num=14) #14 15
        self.up2 = Up(thresholds, device, clamp_slope, reset,512, 256 // factor, bilinear=True,layer_num=16) #16 17
        self.up3 = Up(thresholds, device, clamp_slope, reset,256, 128 // factor, bilinear=True,layer_num=18) #18 19
        self.up4 = Up(thresholds, device, clamp_slope, reset,128, 64, bilinear=True,layer_num=20) #20 21
        self.outc = OutConv(thresholds, device, clamp_slope, reset,64, 1,layer_num=22) #22


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


def vgg16_nobn_spike(thresholds, device, clamp_slope,reset, n_channels,num_classes):
    return Unet_nobn_spike(thresholds, device, clamp_slope, reset,n_channels, num_classes)
