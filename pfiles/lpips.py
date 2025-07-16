#

import torch
from torch import nn
import torch.nn.functional as F

from torchvision import models

import numpy as np
from collections import namedtuple

#

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#

def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2, 3], keepdim=keepdim)

#

class vgg16(nn.Module):
    def __init__(self, requires_grad=False):
        super(vgg16, self).__init__()
        
        vgg_pretrained_features = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        
        self.N_slices = 5
        
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
            
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
                
    def forward(self, x):
        
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        
        return out

#

class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])
        
    def forward(self, inp):
        return (inp - self.shift) / self.scale

#

class NetLinlayer(nn.Module):
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinlayer, self).__init__()
        
        layers = [nn.Dropout()] if use_dropout else []
        layers += [nn.Conv2d(in_channels=chn_in, out_channels=chn_out, kernel_size=1, stride=1, padding=0, bias=False)]
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        out = self.model(x)
        return out
    
#

class LPIPS(nn.Module):
    def __init__(self, use_dropout=True):
        super(LPIPS, self).__init__()
        
        self.scaling_layer = ScalingLayer()
        
        self.chns = [64, 128, 256, 512, 512]
        self.L = len(self.chns)
        self.net = vgg16(requires_grad=False)
        
        self.lin0 = NetLinlayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinlayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinlayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinlayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinlayer(self.chns[4], use_dropout=use_dropout)
        
        self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        self.lins = nn.ModuleList(self.lins)
        
        model_path = 'weights/vgg.pth'
        self.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=False)
        
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
        
    def forward(self, in0, in1, normalize=False):
        if normalize:
            in0 = 2 * in0 - 1
            in1 = 2 * in1 - 1
            
        in0_input, in1_input = self.scaling_layer(in0), self.scaling_layer(in1)
        
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        
        for kk in range(self.L):
            feats0[kk], feats1[kk] = F.normalize(outs0[kk], dim=1), F.normalize(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2
            
        res = [spatial_average(self.lins[kk](diffs[kk]), keepdim=True) for kk in range(self.L)]
        val = 0
        
        for l in range(self.L):
            val += res[l]
        return val