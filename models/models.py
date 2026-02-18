import torch
import torch.nn as nn
from .mae import MAE
from .vit import ViT
from .vit import Transformer, Attention, FeedForward
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import einops
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import torch.nn.functional as F
import torch


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1):
        x1 = self.up(x1)

        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        return self.conv(x1)



def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class ERAencoder(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., num_registers = 0):
        super(ERAencoder,self).__init__()

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        self.num_registers = num_registers

        #assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + self.num_registers, dim))

        self.positional_ff = nn.Linear(dim,dim)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, img):
        x = self.to_patch_embedding(img)

        b, n, d = x.shape

        x += self.positional_ff(self.pos_embedding[:, :])

        x = self.transformer(x)

        return x


class ERAdecoder(nn.Module):
    def __init__(self, *, final_image_size, patch_size, dim, depth, heads, mlp_dim, final_channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., encoder_dim=1537):
        super(ERAdecoder,self).__init__()

        image_height, image_width = pair(final_image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = final_channels * patch_height * patch_width
        self.num_registers=0

        transformer_layer = TransformerDecoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout, batch_first = True)
        self.transformer = TransformerDecoder(transformer_layer, num_layers=depth)

        self.enc_to_dec = nn.Linear(encoder_dim, dim)
        self.enc_to_dec_pos = nn.Linear(encoder_dim, dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, 910, dim))

        self.positional_ff = nn.Linear(dim,dim)

    def forward(self, x, memory, encoder_pos_emb):

        memory = self.enc_to_dec(memory)

        memory += self.enc_to_dec_pos(encoder_pos_emb)

        x += self.positional_ff(self.pos_embedding[:,:]) #added for version 3

        x = self.transformer(x, memory)
        
        x = x[:,:910]   #take only the first 910 tokens

        return x


"""
encoder = 
hrrr_decoder
memory_pos_emb
"""

class ERA5UpscalerV2(nn.Module):

    def __init__(self, encoder, hrrr_decoder, hrrr_final_decoder):
        super(ERA5UpscalerV2,self).__init__()

        self.encoder = encoder
        self.to_patch = encoder.to_patch_embedding[0]
        self.patch_to_emb = nn.Sequential(*encoder.to_patch_embedding[1:])
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]

        self.hrrr_decoder = hrrr_decoder
        self.hrrr_final_decoder = hrrr_final_decoder

        self.enc_to_dec = nn.Linear(encoder_dim, self.decoder_dim, requires_grad=False)
        self.enc_to_dec_pos = nn.Linear(encoder_dim, self.decoder_dim, requires_grad=False)

        self.decoder_patches, self.decoder_dim = self.hrrr_decoder.pos_embedding.shape[-2:]
        hrrr_pixel_values_per_patch = 10*10*2

        self.to_pixels = nn.Linear(self.decoder_dim, hrrr_pixel_values_per_patch)

    def forward(self, img, img_target):

        device = img.device

        patches = self.to_patch(img)
        x = self.patch_to_emb(patches)

        """ Encoder """
        x = F.pad(input=x, pad=(0,0,0,self.encoder.num_registers,0,0), mode='constant', value=0)
        b, _, _   = x.shape
        _, n, dim = self.hrrr_decoder.pos_embedding.shape

        x += self.encoder.pos_embedding[:, :]

        memory = self.encoder.transformer(x)
        memory = self.enc_to_dec(memory)
        memory += self.enc_to_dec_pos(self.encoder.pos_embedding)

        """ Decoder """
        """ we need to double check this. decoder effectively has 1056 - 910 = 146 register tokens """
        y = torch.zeros((b,self.decoder_patches,self.decoder_dim)).to(memory.get_device())
        y += self.hrrr_decoder.pos_embedding[:,:]
        y = self.hrrr_decoder.transformer(y, memory)
        y = self.hrrr_final_decoder(y)

        return self.to_pixels(y), img_target

class ERA5Upscaler(nn.Module):

    def __init__(self, encoder, hrrr_decoder):
        super(ERA5Upscaler,self).__init__()

        self.encoder = encoder
        self.to_patch = encoder.to_patch_embedding[0]
        self.patch_to_emb = nn.Sequential(*encoder.to_patch_embedding[1:])
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]

        self.hrrr_decoder = hrrr_decoder

        self.decoder_patches, self.decoder_dim = self.hrrr_decoder.pos_embedding.shape[-2:]
        self.enc_to_dec = nn.Linear(self.decoder_dim,192)
        self.enc_to_dec_pos = nn.Linear(self.decoder_dim,192)

        self.t_token = nn.Parameter(torch.randn(encoder_dim))
        self.tminus1_token = nn.Parameter(torch.randn(encoder_dim))

        #self.unet_mask = nn.Parameter(torch.randn(1,1,self.decoder_dim))
        #self.unet_mask = nn.Parameter(torch.randn(1,1,192))

        #self.decoder = Transformer(dim = 192, depth = 2, heads = 4, dim_head = 64, mlp_dim = 192*4)
        #self.layernorm = torch.nn.LayerNorm(192)

        self.l1 = Up(192,96,bilinear=True)
        #self.l1 = Up(392,96,bilinear=True)
        self.l2 = Up(96,48,bilinear=True)
        self.l3 = Up(48,12,bilinear=True)
        self.l4 = Up(12,2,bilinear=True)

    def forward(self, img, img_target, hrrr_indices, hrrr_anti_indices,two_hours=True):

        device = "cpu" #img.get_device()

        """ Encoder """

        if two_hours:
            img2 = img[:,15:,...]
            img  = img[:,:15,...]

        patches = self.to_patch(img)
        x = self.patch_to_emb(patches)
        x = F.pad(input=x, pad=(0,0,0,self.encoder.num_registers,0,0), mode='constant', value=0)
        b, _, _   = x.shape
        _, n, dim = self.hrrr_decoder.pos_embedding.shape


        if two_hours:
            patches_tminus1 = self.to_patch(img2)
            x_t1 = self.patch_to_emb(patches_tminus1)
            x_t1 = F.pad(input=x_t1, pad=(0,0,0,self.encoder.num_registers,0,0), mode='constant', value=0)

            x    += self.encoder.positional_ff(self.encoder.pos_embedding[:, :])
            x_t1 += self.encoder.positional_ff(self.encoder.pos_embedding[:, :])

            x    += self.t_token
            x_t1 += self.tminus1_token

            x = torch.cat((x_t1, x),dim=1)
        else:
            x += self.encoder.positional_ff(self.encoder.pos_embedding[:, :])

        memory = self.encoder.transformer(x)

        memory = self.hrrr_decoder.enc_to_dec(memory)

        if two_hours:
            memory += self.hrrr_decoder.enc_to_dec_pos(self.encoder.positional_ff(self.encoder.pos_embedding)).repeat(1,2,1)
        else:
            memory += self.hrrr_decoder.enc_to_dec_pos(self.encoder.positional_ff(self.encoder.pos_embedding))

        """ Decoder """
        """ we need to double check this. decoder effectively has 1056 - 910 = 146 register tokens """

        x = torch.zeros((b,n,dim)).to(device)
        x += self.hrrr_decoder.positional_ff(self.hrrr_decoder.pos_embedding[:,:])
        x = self.hrrr_decoder.transformer(x, memory)
        x = self.enc_to_dec(x) + self.enc_to_dec_pos(self.hrrr_decoder.positional_ff(self.hrrr_decoder.pos_embedding))
        
        x = x[:,:910]   #take only the first 910 tokens #was commented

        #y = torch.zeros((b, 1742, dim)).to(device)
        y = torch.zeros((b, 1742, 192)).to(device)

        batchrange = torch.arange(b, device = device)[:, None]

        y[batchrange, hrrr_indices.to(device)] = x
        y[batchrange, hrrr_anti_indices.to(device)] = 1e-5*2*(torch.randn(b,hrrr_anti_indices.shape[0],192).to(device)-0.5)

        y = torch.nn.functional.relu(y)

        y = einops.rearrange(y, 'b (h w) dim -> b dim h w',h=67, w=26)
        y = self.l1(y)
        y = self.l2(y)
        y = self.l3(y)
        y = self.l4(y)

        y = torch.nn.functional.interpolate(y, size=(670,260), scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False)

        y_full = y

        y = einops.rearrange(y, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',c=2,p1=10, p2=10)
        y = y[batchrange, hrrr_indices]

        return y, img_target, y_full


if __name__ == '__main__':

    enc = ERAencoder(image_size=(240,440),
	       patch_size=(15,22),
	       num_classes=192,
           channels=12,
	       dim=768,
	       depth=8,
	       heads=4,
	       mlp_dim=3)

    mae = MAE(encoder=enc, decoder_dim=192)
    
    sample = torch.rand(10,12,240,440)

    y = mae(sample)

