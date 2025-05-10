import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import DropPath
from typing import Optional
from torch import nn, Tensor
from models.modeling.audio_backbone.CED.layers import Mlp
from models.modeling.audio_backbone.CED.build_CED import Block


def permute_and_flatten(layer, N, A, C, H, W):
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2).contiguous()
    layer = layer.reshape(N, -1, C)
    return layer


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X"""
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


class OneStreamAttention(nn.Module):
    def __init__(self, v_dim, a_dim, embed_dim, num_heads, dropout=0.1):
        """
        v_dim: visual feature dimension
        a_dim: audio feature dimension
        embed_dim: embedding dimension
        num_heads: number of heads
        """
        super(OneStreamAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.v_dim = v_dim
        self.a_dim = a_dim

        assert (
            self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.n_layers = 3
        self.attn_block = nn.ModuleList()
        for i in range(self.n_layers):
            self.attn_block.append(Block(dim = embed_dim, num_heads= num_heads)) 
        self.norm = nn.LayerNorm(embed_dim)


    def forward(self, v, a, pos_v=None, pos_a=None):
        _, visual_len, _ = v.size()  # * bs*5, 56*56, 256
        _, audio_len, _ = a.size()
        # total_size = tgt_h * tgt_w + audio_len

        vis = v + pos_v
        aud = a + pos_a

        # vis = vis.flatten(2).transpose(1, 2) 
        vis_aud = torch.cat([vis, aud], dim = 1)
        for i in range(self.n_layers):
            vis_aud = self.attn_block[i](vis_aud) 
             
        vis_aud = self.norm(vis_aud) 
        
        out_v, out_a = torch.split(vis_aud, [visual_len, audio_len], dim = 1)

        return out_v, out_a



class OneStreamAttentionBlock(nn.Module):
    def __init__(
        self,
        visual_features_names,
        vision_dim_list,
        audio_dim,
        embed_dim,
        num_heads,
        hidden_dim=None,
        dropout=0.1,
        drop_path=0.0,
        init_values=1e-4,
    ):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super(OneStreamAttentionBlock, self).__init__()

        self.visual_features_names = visual_features_names

        # pre layer norm
        self.layer_norm_v_list = nn.ModuleList()
        self.layer_norm_a_list = nn.ModuleList()
        self.attn_list = nn.ModuleList()

        self.gamma_v_list = nn.ParameterList()
        for vision_dim in vision_dim_list:
            self.layer_norm_v_list.append(nn.LayerNorm(vision_dim))
            self.layer_norm_a_list.append(nn.LayerNorm(audio_dim))
            self.attn_list.append(
                OneStreamAttention(
                    v_dim=vision_dim,
                    a_dim=audio_dim,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
            )
            # add layer scale for training stability
            self.gamma_v_list.append(nn.Parameter(init_values * torch.ones((vision_dim)), requires_grad=True))

        # add layer scale for training stability
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.gamma_a = nn.Parameter(init_values * torch.ones((audio_dim)), requires_grad=True)

    def forward(self, visual_features, audio_feature, pos_a=None, pos_v=None):
        size_per_level = []
        new_v_list, new_a_list = [], []
        for num_level, feature_name in enumerate(self.visual_features_names):
            feat_per_level = visual_features[feature_name]
            bs, c, h, w = feat_per_level.shape 
            size_per_level.append([h, w])
            visual_feature_flatten = permute_and_flatten(feat_per_level, bs, 1, c, h, w)  
            # * start fusion
            new_v, new_a = self.single_attention_call(v=visual_feature_flatten, a=audio_feature, level=num_level, pos_a=pos_a, pos_v=pos_v)

            new_v = new_v.transpose(1, 2).contiguous()
            new_v_list.append(new_v)
            new_a_list.append(new_a)
  
        new_a = torch.stack(new_a_list, dim=1)  
        audio_feature = torch.mean(new_a, dim=1)  
     
        for num_level, (h, w) in enumerate(size_per_level):
            new_v_per_level = new_v_list[num_level].view(bs, -1, h, w).contiguous()
            visual_features[self.visual_features_names[num_level]] = new_v_per_level

        return visual_features, audio_feature

    def single_attention_call(self, v, a, level, pos_v=None, pos_a=None):
        """
        Args:
            v: visual feature
            a: audio feature
        """
        v = self.layer_norm_v_list[level](v)  
        a = self.layer_norm_a_list[level](a)  

        delta_v, delta_a = self.attn_list[level](v, a, pos_v, pos_a)  
        v = v + self.drop_path(self.gamma_v_list[level] * delta_v)
        a = a + self.drop_path(self.gamma_a * delta_a)
        return v, a 

