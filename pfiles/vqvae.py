#

import torch
from torch import nn

from pfiles.blocks import DownBlock, MidBlock, UpBlock


#

class VQVAE(nn.Module):
    def __init__(self, im_channels):
        super(VQVAE, self).__init__()
        
        self.down_channels = [32, 64, 128]
        self.mid_channels = [128, 128]
        self.down_sample = [True, True]
        self.num_down_layers = 1
        self.num_mid_layers = 1
        self.num_up_layers = 1
        
        self.attns = [False, False]
        
        self.z_channels = 16
        self.codebook_size = 128
        self.norm_channels = 32
        self.num_heads = 16
        
        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-1]
        assert len(self.down_sample) == len(self.down_channels) - 1
        assert len(self.attns) == len(self.down_channels) - 1
        
        self.up_sample = list(reversed(self.down_sample))
        self.encoder_conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=3, padding=1)
        
        self.encoder_layers = nn.ModuleList([])
        for i in range(len(self.down_channels) - 1):
            self.encoder_layers.append(DownBlock(self.down_channels[i], self.down_channels[i + 1], t_emb_dim=None,
                                                down_sample=self.down_sample[i], num_heads=self.num_heads,
                                                num_layers=self.num_down_layers, attn=self.attns[i],
                                                norm_channels=self.norm_channels))
        
        self.encoder_mids = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            self.encoder_mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i + 1], t_emb_dim=None,
                                             num_heads=self.num_heads, num_layers=self.num_mid_layers,
                                             norm_channels=self.norm_channels))
            
        self.encoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[-1])
        self.encoder_conv_out = nn.Conv2d(self.down_channels[-1], self.z_channels, kernel_size=3, padding=1)
        
        self.pre_quant_conv = nn.Conv2d(self.z_channels, self.z_channels, kernel_size=1)
        self.embedding = nn.Embedding(self.codebook_size, self.z_channels)
        
        self.post_quant_conv = nn.Conv2d(self.z_channels, self.z_channels, kernel_size=1)
        self.decoder_conv_in = nn.Conv2d(self.z_channels, self.mid_channels[-1], kernel_size=3, padding=1)
        
        self.decoder_mids = nn.ModuleList([])
        for i in reversed(range(1, len(self.mid_channels))):
            self.decoder_mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i - 1], t_emb_dim=None,
                                             num_heads=self.num_heads, num_layers=self.num_mid_layers,
                                             norm_channels=self.norm_channels))
            
        self.decoder_layers = nn.ModuleList([])
        for i in reversed(range(1, len(self.down_channels))):
            self.decoder_layers.append(UpBlock(self.down_channels[i], self.down_channels[i - 1], t_emb_dim=None,
                                              up_sample=self.down_sample[i - 1], num_heads=self.num_heads,
                                              num_layers=self.num_up_layers, attn=self.attns[i - 1],
                                              norm_channels=self.norm_channels))
            
        self.decoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[0])
        self.decoder_conv_out = nn.Conv2d(self.down_channels[0], im_channels, kernel_size=3, padding=1)
        
    def quantize(self, x):
        
        B, C, H, W = x.shape
        
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(x.size(0), -1, x.size(-1))
        
        dist = torch.cdist(x, self.embedding.weight[None, :].repeat((x.size(0), 1, 1)))
        min_encoding_indices = torch.argmin(dist, dim=-1)
        quant_out = torch.index_select(self.embedding.weight, 0, min_encoding_indices.view(-1))
        x = x.reshape((-1, x.size(-1)))
        
        commitment_loss = torch.mean((quant_out.detach() - x) ** 2)
        codebook_loss = torch.mean((quant_out - x.detach()) ** 2)
        quantize_losses = {'codebook_loss': codebook_loss, 'commitment_loss': commitment_loss}
        
        quant_out = x + (quant_out - x).detach()
        quant_out = quant_out.reshape((B, H, W, C)).permute(0, 3, 1, 2)
        min_encoding_indices = min_encoding_indices.reshape((-1, quant_out.size(-2), quant_out.size(-1)))
        return quant_out, quantize_losses, min_encoding_indices
    
    def encode(self, x):
        out = self.encoder_conv_in(x)
        
        for idx, down in enumerate(self.encoder_layers):
            out = down(out)
        for mid in self.encoder_mids:
            out = mid(out)
        
        out = self.encoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.encoder_conv_out(out)
        out = self.pre_quant_conv(out)
        out, quant_losses, _ = self.quantize(out)
        return out, quant_losses
    
    def decode(self, z):
        out = z
        out = self.post_quant_conv(out)
        out = self.decoder_conv_in(out)
        
        for mid in self.decoder_mids:
            out = mid(out)
        for idx, up in enumerate(self.decoder_layers):
            out = up(out)
            
        out = self.decoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.decoder_conv_out(out)
        return out
    
    def forward(self, x):
        z, quant_losses = self.encode(x)
        out = self.decode(z)
        return out, z, quant_losses