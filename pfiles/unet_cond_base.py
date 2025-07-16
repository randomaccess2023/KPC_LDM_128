#

import torch
from torch import nn

from pfiles.blocks import get_time_embedding, DownBlock, MidBlock, UpBlockUNet

#

class UNet(nn.Module):
    def __init__(self, im_channels, cls):
        super(UNet, self).__init__()
        
        self.down_channels = [128, 256, 256, 256]
        self.mid_channels = [256, 256]
        self.t_emb_dim = 256
        self.down_sample = [True, True, False]
        self.num_down_layers = 2
        self.num_mid_layers = 2
        self.num_up_layers = 2
        self.attns = [True, True, True]
        self.norm_channels = 32
        self.num_heads = 16
        self.conv_out_channels = 128
        
        self.num_classes = cls
        
        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-2]
        assert len(self.down_sample) == len(self.down_channels) - 1
        assert len(self.attns) == len(self.down_channels) - 1
        
        self.class_emb = nn.Embedding(self.num_classes, self.t_emb_dim)
        self.conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=3, padding=1)
        
        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim)
        )
        
        self.up_sample = list(reversed(self.down_sample))
        self.downs = nn.ModuleList([])
        for i in range(len(self.down_channels) - 1):
            self.downs.append(DownBlock(self.down_channels[i], self.down_channels[i + 1], self.t_emb_dim,
                                      down_sample=self.down_sample[i], num_heads=self.num_heads,
                                      num_layers=self.num_down_layers, attn=self.attns[i], norm_channels=self.norm_channels))
            
        self.mids = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            self.mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i + 1], self.t_emb_dim,
                                     num_heads=self.num_heads, num_layers=self.num_mid_layers,
                                     norm_channels=self.norm_channels))
        
        self.ups = nn.ModuleList([])
        for i in reversed(range(len(self.down_channels) - 1)):
            self.ups.append(UpBlockUNet(self.down_channels[i] * 2, self.down_channels[i - 1] if i != 0
                                        else self.conv_out_channels, self.t_emb_dim, up_sample=self.down_sample[i],   
                                        num_heads=self.num_heads,num_layers=self.num_up_layers,
                                        norm_channels=self.norm_channels))
            
        self.norm_out = nn.GroupNorm(self.norm_channels, self.conv_out_channels)
        self.conv_out = nn.Conv2d(self.conv_out_channels, im_channels, kernel_size=3, padding=1)
        
    def forward(self, x, t, cond_input=None):
        
        out = self.conv_in(x)
              
        t_emb = get_time_embedding(torch.as_tensor(t).long(), self.t_emb_dim)
        t_emb = self.t_proj(t_emb)
        
        if cond_input is not None:
            t_emb += self.class_emb(cond_input)
        
        down_outs = []
        
        for idx, down in enumerate(self.downs):
            down_outs.append(out)
            out = down(out, t_emb)
        
        for mid in self.mids:
            out = mid(out, t_emb)
            
        for up in self.ups:
            down_out = down_outs.pop()
            out = up(out, down_out, t_emb)
            
        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out)
        
        return out