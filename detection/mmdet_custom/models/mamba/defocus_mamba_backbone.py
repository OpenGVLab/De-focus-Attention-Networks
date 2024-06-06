# Copyright (c) Shanghai AI Lab. All rights reserved.
import logging
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from timm.models.layers import trunc_normal_
from torch.nn.init import normal_
from torchvision.transforms.functional import resize

from mmdet.registry import MODELS
from mmengine.runner.checkpoint import load_checkpoint
from mmengine.model.base_module import BaseModule, ModuleList
from mmengine.utils import to_2tuple
from mmcv.cnn.bricks.transformer import PatchEmbed, AdaptivePadding
from mmcv.cnn import build_conv_layer, build_norm_layer

from .mamba_block import MambaBlock, MambaMixer
from .mamba_config import get_config

_logger = logging.getLogger(__name__)


@MODELS.register_module()
class Defocus_Mamba_Backbone(BaseModule):
    def __init__(self, mamba_config_dict=None,
                 img_size=224, patch_size=16, in_chans=3, drop_rate=0.,
                 pretrained=None, use_fp32=False, PatchChange=False, pretrain_size=224,
                 *args, **kwargs):
        super().__init__()
        # Params Assignment Here
        if (pretrained is not None) and (not isinstance(pretrained, str)):
            print("WARNING: 'pretrained' in Defocus_Mamba_Backbone should be a string")
            pretrained = None
        self.pretrained = pretrained
        self.use_fp32 = use_fp32
        mamba_config = get_config(mamba_config_dict)
        self.mamba_config = mamba_config
        embed_dims = mamba_config.hidden_size
        self.patch_size = patch_size
        
        # Patch Embedding
        if isinstance(img_size,int):
            img_size = (img_size, img_size)
        if PatchChange:
            self.patch_embed = PatchEmbedChangeRes(
                in_channels=in_chans,
                input_size=img_size,
                embed_dims=embed_dims,
                conv_type='Conv2d',
                kernel_size=patch_size,
                stride=patch_size,
                bias=True,
                pretrain_size=pretrain_size)
        else:
            self.patch_embed = PatchEmbed(
                in_channels=in_chans,
                input_size=img_size,
                embed_dims=embed_dims,
                conv_type='Conv2d',
                kernel_size=patch_size,
                stride=patch_size,
                bias=True)
        self.patch_resolution = self.patch_embed.init_out_size
        
        # Other params
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))

        dpr = [x.item() for x in torch.linspace(0, mamba_config.drop_path_rate, mamba_config.depth)]
        self.blocks = nn.ModuleList([MambaBlock(mamba_config, i, dpr[i]) for i in range(mamba_config.depth)])            
        self.norm = nn.LayerNorm(embed_dims)
        
        # Initialization and Pretrain Loading
        self.apply(self._init_weights)
        self.apply(self._init_mamba_weights)
        
        # Load Pretrained Mamba Model
        if self.pretrained is not None:
            print('load pretrained model')
            msg = load_checkpoint(self, self.pretrained, map_location='cpu', logger=_logger) #, strict=False, revise_keys=[(r'^backbone.','')]

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

        if hasattr(self, 'cls_token'):
            trunc_normal_(self.cls_token, std=.02)
            
    def _init_mamba_weights(self, module):
        if isinstance(module, MambaMixer):
            dt_init_std = self.mamba_config.time_step_rank**-0.5 * self.mamba_config.time_step_scale
            nn.init.uniform_(module.dt_proj.weight, -dt_init_std, dt_init_std)

            dt = torch.exp(
                torch.linspace(start=math.log(self.mamba_config.time_step_min),
                                end=math.log(self.mamba_config.time_step_max),
                                steps=self.mamba_config.intermediate_size)
            ).clamp(min=self.mamba_config.time_step_floor)

            # # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                module.dt_proj.bias.copy_(inv_dt)
            module.dt_proj.bias._no_reinit = True

        if isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.mamba_config.initializer_range)

        if self.mamba_config.rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            for name, p in module.named_parameters():
                if name in ["out_proj.weight"]:
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                    # We need to reinit p since this code could be called multiple times
                    # Having just p *= scale would repeatedly scale it down
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(self.mamba_config.depth)
   
        if isinstance(module, MambaMixer) and hasattr(module, 'freq_proj'):
            nn.init.constant_(module.freq_proj.weight, 0)
            freq = (100**(-torch.arange(1, module.ssm_state_size+1)/module.ssm_state_size) - 100**(-1)) / (1 - 100**(-1))
            with torch.no_grad():
                module.freq_proj.bias.copy_(freq)
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x, samplelist=None):
        B, C, H, W = x.shape
        x, patch_resolution = self.patch_embed(x)
        B, L, C = x.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((x, cls_tokens), dim=1)

        with torch.autocast(device_type="cuda", enabled=(not self.use_fp32)):
            for i , blk in enumerate(self.blocks):
                x = blk(x)
        x = self.norm(x)
        
        out = x[:, :-1, :]
        if isinstance(self.patch_embed,PatchEmbedChangeRes):
            out = self.patch_embed.re_shape(out, patch_resolution)
        else:
            out = out.permute(0, 2, 1).contiguous().view(B, C, math.ceil(H/self.patch_size), math.ceil(W/self.patch_size))   # turn out into [B,C,H,W]
        return out


class PatchEmbedChangeRes(BaseModule):
    def __init__(self,
                 in_channels=3,
                 embed_dims=768,
                 conv_type='Conv2d',
                 kernel_size=16,
                 stride=16,
                 padding='corner',
                 dilation=1,
                 bias=True,
                 norm_cfg=None,
                 input_size=None,
                 init_cfg=None,
                 pretrain_size=224):
        super().__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        if stride is None:
            stride = kernel_size

        kernel_size = to_2tuple(kernel_size)
        self.kernel_size = kernel_size
        stride = to_2tuple(stride)
        self.stride = stride
        dilation = to_2tuple(dilation)
        self.dilation = dilation

        if isinstance(padding, str):
            self.adaptive_padding = AdaptivePadding(
                kernel_size=(pretrain_size, pretrain_size),
                stride=(pretrain_size, pretrain_size),
                dilation=dilation,
                padding=padding)
            # disable the padding of conv
            padding = 0
            self.pretrain_size = pretrain_size
        else:
            self.adaptive_padding = None
        padding = to_2tuple(padding)
        self.padding = padding

        self.projection = build_conv_layer(
            dict(type=conv_type),
            in_channels=in_channels,
            out_channels=embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias)

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm = None

        if input_size:
            input_size = to_2tuple(input_size)
            # `init_out_size` would be used outside to
            # calculate the num_patches
            # e.g. when `use_abs_pos_embed` outside
            self.init_input_size = input_size
            self.init_out_size = self.out_shape(input_size)
        else:
            self.init_input_size = None
            self.init_out_size = None
            
    def out_shape(self, size, kernel_size=16):
        input_size = to_2tuple(size)

        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        h_out = (input_size[0] - 1) // kernel_size + 1
        w_out = (input_size[1] - 1) // kernel_size + 1
        return (h_out, w_out)

    def forward(self, x):
        out_shape = self.out_shape((x.shape[2], x.shape[3]))
        if self.adaptive_padding:
            x = self.adaptive_padding(x)

        x = self.projection(x)
        B, C, H, W = x.shape

        n_pretrain_patch = self.pretrain_size // self.kernel_size[0]
        x = x.view(B, C, H//n_pretrain_patch, n_pretrain_patch, W//n_pretrain_patch, n_pretrain_patch)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, -1, C)

        if self.norm is not None:
            x = self.norm(x)
        return x, out_shape

    def re_shape(self, out, patch_clip):
        n_pretrain_patch = self.pretrain_size // self.kernel_size[0]
        patch_size = ( 
            math.ceil(patch_clip[0]/n_pretrain_patch)*n_pretrain_patch, 
            math.ceil(patch_clip[1]/n_pretrain_patch)*n_pretrain_patch
        )  # = math.ceil
        assert out.shape[1] == patch_size[0]*patch_size[1]
        
        B, L, C = out.shape
        out = out.permute(0, 2, 1).contiguous()

        out = out.view(B, C, patch_size[0]//n_pretrain_patch, patch_size[1]//n_pretrain_patch, n_pretrain_patch, n_pretrain_patch)
        out = out.permute(0, 1, 2, 4, 3, 5).reshape(B, C, patch_size[0], patch_size[1])
        out = out[:, :, :patch_clip[0], :patch_clip[1]]
        return out