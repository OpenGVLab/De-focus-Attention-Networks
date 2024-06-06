import os
import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from functools import partial
from easydict import EasyDict as edict

from timm.models.vision_transformer import PatchEmbed , _cfg
from timm.models.layers import DropPath, to_2tuple, trunc_normal_, drop_path
from timm.models.registry import register_model

from .mamba_block import MambaBlock, MambaMixer


class PatchEmbedChangeRes(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
            ratio=2,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.ratio = ratio

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        if self.flatten:
            x = x.view(B, C, self.ratio, H//self.ratio, self.ratio, W//self.ratio)
            x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, -1, C)
        x = self.norm(x)
        return x


class DefocusAttentionNetwork(nn.Module):
    """ 
    De-focus Attention Network
    """
    def __init__(self, img_size=224,  patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, global_pool=None,
                 block_layers=MambaBlock, Patch_layer=PatchEmbed, act_layer=nn.GELU,
                 args=None, **kwargs):
        super().__init__()

        self.args = args
        self.dropout_rate = drop_rate

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        if args.MODEL.patch_embed_change_res:
            self.patch_embed = PatchEmbedChangeRes(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, ratio=2)
        else:
            self.patch_embed = Patch_layer(
                    img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches

        if 'cls' in self.args.MODEL.head_type:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        if self.args.MODEL.use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        if args.MODEL.TYPE in ['vit']:
            dpr = [x.item() for x in torch.linspace(0, args.MODEL.DROP_PATH_RATE, depth)]
            self.blocks = nn.ModuleList([
                block_layers(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=0.0, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    act_layer=act_layer,Attention_block=Attention_block,Mlp_block=Mlp_block,init_values=init_scale,args=args)
                for i in range(depth)])
        elif args.MODEL.TYPE == 'mamba':
            # convert to mamba config
            mamba_config = edict({
                    'hidden_size': embed_dim,
                    'num_layers': args.MODEL.depth,
                    'use_rope': args.MODEL.use_rope,
                })
            for key, value in args.MODEL.MAMBA.items():
                mamba_config[key] = value
            self.mamba_config = mamba_config

            dpr = [x.item() for x in torch.linspace(0, args.MODEL.drop_path_rate, args.MODEL.depth)]
            self.blocks = nn.ModuleList([block_layers(mamba_config, i, dpr[i], use_checkpoint=args.TRAIN.USE_CHECKPOINT) for i in range(args.MODEL.depth)])            

        self.norm = norm_layer(embed_dim)

        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        if 'IN1k' in self.args.MODEL.head_type:
            self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
            if self.args.MODEL.use_aux_loss:
                self.head2 = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        elif 'IN22k' in self.args.MODEL.head_type:
            self.head = nn.Sequential(
                nn.Linear(embed_dim, 3*embed_dim),
                nn.Tanh(),
                nn.Linear(3*embed_dim, num_classes)
            )
            if self.args.MODEL.use_aux_loss:
                self.head2 = nn.Sequential(
                    nn.Linear(embed_dim, 3*embed_dim),
                    nn.Tanh(),
                    nn.Linear(3*embed_dim, num_classes)
                )

        if hasattr(self, 'pos_embed'):
            trunc_normal_(self.pos_embed, std=.02)
        if hasattr(self, 'cls_token'):
            trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        self.apply(self._init_mamba_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
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
                        p /= math.sqrt(self.mamba_config.num_layers)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head
    
    def get_num_layers(self):
        return len(self.blocks)
    
    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, drop_path_rate=0.0, tblogger=None):
        images = x
        x = self.patch_embed(x)
        patch_embed = x.clone().detach()

        B, L, C = x.shape

        if self.args.MODEL.use_pos_embed:
            x = x + self.pos_embed

        if 'cls' in self.args.MODEL.head_type:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((x, cls_tokens), dim=1)

        if drop_path_rate > 0:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.args.MODEL.depth)]
        else:
            dpr = [x.item() for x in torch.linspace(0, self.args.MODEL.drop_path_rate, self.args.MODEL.depth)]

        for i , blk in enumerate(self.blocks):
            x = blk(x, dpr[i])

        x = self.norm(x)

        if 'cls' in self.args.MODEL.head_type:
            if self.args.MODEL.use_aux_loss:
                out = x
            else:
                out = x[:, -1]
        elif 'avg' in self.args.MODEL.head_type:
            out = x.mean(1)

        return out

    def forward(self, x, drop_path_rate=0.0, tblogger=None):
        x = self.forward_features(x, drop_path_rate, tblogger)
        
        if self.dropout_rate:
            x = F.dropout(x, p=float(self.dropout_rate), training=self.training)

        if self.args.MODEL.use_aux_loss:
            x1 = self.head(x[:, -1:])
            x2 = self.head2(x[:, :-1])
            x = torch.cat([x2, x1], dim=1)
        else:
            x = self.head(x)

        return x


@register_model
def defocus_deit_small(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = DefocusAttentionNetwork(
        img_size = img_size, patch_size=16, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Block, **kwargs)
    return model


@register_model
def defocus_deit_base(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = DefocusAttentionNetwork(
        img_size = img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Block, **kwargs)
    return model


@register_model
def defocus_mamba_small(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = DefocusAttentionNetwork(
        img_size = img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=MambaBlock, **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def defocus_mamba_base(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = DefocusAttentionNetwork(
        img_size = img_size, patch_size=16, embed_dim=768, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=MambaBlock, **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def defocus_mamba_large(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = DefocusAttentionNetwork(
        img_size = img_size, patch_size=16, embed_dim=1024, depth=48, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=MambaBlock, **kwargs)
    model.default_cfg = _cfg()

    return model