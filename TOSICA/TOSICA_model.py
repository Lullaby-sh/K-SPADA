from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn
import copy
from .customized_linear import CustomizedLinear
from einops import rearrange
import random
import numpy as np
import pandas as pd

from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import math

class GCN(nn.Module):
    def __init__(self, num_genes, embed_dim=301, use_position_encode=False, num_heads=4, embed_dim_heads=48):
        super(GCN, self).__init__()
        self.gcn = GCNConv(num_genes, embed_dim)
        # self.linear = nn.Linear(num_genes, embed_dim)
        self.embed_dim = embed_dim
        self.use_position_encode = use_position_encode
        print('use_position_encode:', use_position_encode)
        print('embed_dim_heads:', embed_dim_heads)
        print('num_heads:', num_heads)
        self.num_heads = num_heads
        self.embed_dim_heads = embed_dim_heads
        self.norm = nn.LayerNorm(num_genes)

    def patch_position_encode(self, coordinate, count):
        device = 'cuda:0'
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        # 创建与count相同形状的零张量作为position_encode
        position_encode = torch.zeros_like(count).to(device)
        feature_dim = self.embed_dim
        for i in range(int(feature_dim / 2)):
            if i % 2 == 0:
                position_encode[:, i] = torch.sin(
                    coordinate[:, 0] / (torch.exp((4 * i) / feature_dim * torch.log(torch.tensor([10000.0], dtype=torch.float32)))).to(device)).to(device)
            else:
                position_encode[:, i] = torch.cos(
                    coordinate[:, 0] / (torch.exp((4 * i) / feature_dim * torch.log(torch.tensor([10000.0], dtype=torch.float32)))).to(device)).to(device)

        for j in range(int(feature_dim / 2), feature_dim):
            if j % 2 == 0:
                position_encode[:, j] = torch.sin(coordinate[:, 1] / (
                    torch.exp((4 * (j - feature_dim / 2)) / feature_dim * torch.log(torch.tensor([10000.0], dtype=torch.float32)))).to(device)).to(device)
            else:
                position_encode[:, j] = torch.cos(coordinate[:, 1] / (
                    torch.exp((4 * (j - feature_dim / 2)) / feature_dim * torch.log(torch.tensor([10000.0], dtype=torch.float32))).to(device))).to(device)
        return position_encode

    def forward(self, x, edge_index, edge_weight, coordinate):
        # edge_index = edge_index.reshape(2, -1)
        # edge_weight = edge_weight.reshape(-1)
        # x = self.linear(x)
        # x = F.relu(x)
        # x = self.norm(x)
        x = self.gcn(x, edge_index, edge_weight).to(torch.float32)
        x = F.relu(x)

        # 如果模型处于训练状态，添加高斯噪声
        # if self.training:
        #     noise = torch.randn_like(x) * 1  # 0.1是噪声的标准差，你可以根据需要调整
        #     x = x + noise

        embeddings = x
        if self.use_position_encode:
            position_encode = self.patch_position_encode(coordinate, x)
            x = x + position_encode
            x = F.relu(x)
        x = x.unsqueeze(1)
        x = x.unsqueeze(3)
        x = x.repeat(1, self.num_heads, 1, self.embed_dim_heads // self.num_heads)
        return x, embeddings


class CommunicateGCN(nn.Module):
    def __init__(self, num_genes, embed_dim=301, nums_heads=4, embed_dim_heads=48):
        super(CommunicateGCN, self).__init__()
        self.gcn = GCNConv(num_genes, embed_dim)
        self.embed_dim = embed_dim
        self.num_heads = nums_heads
        self.embed_dim_heads = embed_dim_heads
    
    def forward(self, x, edge_index, edge_weight):
        x = self.gcn(x, edge_index, edge_weight).to(torch.float32)
        x = F.relu(x)
        x = x.unsqueeze(1)
        x = x.unsqueeze(3)
        x = x.repeat(1, self.num_heads, 1, self.embed_dim_heads // self.num_heads)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    # Returns a tensor filled with random numbers from a uniform distribution on the interval
    # [0,1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    # 为了保持期望值不变，除以keep_prob, * 运算：逐元素乘，矩阵乘 matmul
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class FeatureEmbed(nn.Module):
    def __init__(self, num_genes, mask, embed_dim=192, fe_bias=True, norm_layer=None):
        super().__init__()
        self.num_genes = num_genes
        self.num_patches = mask.shape[1]
        self.embed_dim = embed_dim
        # embed_dim 的意思是迭代运行48次，需要48个在axis=1方向concat的mask矩阵
        mask = np.repeat(mask,embed_dim,axis=1)
        self.mask = mask
        self.fe = CustomizedLinear(self.mask)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    def forward(self, x):
        num_cells = x.shape[0]
        x = rearrange(self.fe(x), 'h (w c) -> h c w ', c=self.num_patches)
        x = self.norm(x)
        # x: (8, 300, 48)
        return x

class Attention(nn.Module):
    def __init__(self,
                 dim, 
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
    def forward(self, x, latent_graph):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q + latent_graph
        k = k + latent_graph
        # v = v + latent_graph
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn = attn + latent_graph
        attn = attn.softmax(dim=-1)
        weights = attn
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, weights

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features 
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self,
                 dim, 
                 num_heads,
                 mlp_ratio=4., 
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0., 
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)
    def forward(self, x, latent_graph):
        #x = x + self.drop_path(self.attn(self.norm1(x)))
        hhh, weights = self.attn(self.norm1(x), latent_graph)
        x = x + self.drop_path(hhh)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, weights

def get_weight(att_mat):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # att_mat = torch.stack(att_mat).squeeze(1)
    #print(att_mat.size())
    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=2)
    #print(att_mat.size())
    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(3))
    aug_att_mat = att_mat.to(device) + residual_att.to(device)
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
    #print(aug_att_mat.size())
    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size()).to(device)
    joint_attentions[0] = aug_att_mat[0]
    
    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

    #print(joint_attentions.size())
    # Attention from the output token to the input space.
    v = joint_attentions[-1]
    #print(v.size())
    v = v[:,0,1:]
    #print(v.size())
    return v

class Transformer(nn.Module):
    def __init__(self, num_classes, num_genes, mask, fe_bias=True,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=FeatureEmbed, norm_layer=None,
                 act_layer=None, use_gcn=False):
        """
        Args:
            num_classes (int): number of classes for classification head
            num_genes (int): number of feature of input(expData) 
            embed_dim (int): embedding dimension
            depth (int): depth of transformer 
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate 
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): feature embed layer
            norm_layer: (nn.Module): normalization layer
        """
        super(Transformer, self).__init__()
        self.use_gcn = use_gcn
        if use_gcn:
            self.gcn = GCN(num_genes, 301, use_position_encode=False, num_heads=num_heads, embed_dim_heads=embed_dim)
            # self.communicate_gcn = CommunicateGCN(num_genes, embed_dim=embed_dim, nums_heads=num_heads, embed_dim_heads=embed_dim)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.feature_embed = embed_layer(num_genes, mask = mask, embed_dim=embed_dim, fe_bias=fe_bias)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]
        #self.blocks = nn.Sequential(*[
        #    Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #          drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
        #          norm_layer=norm_layer, act_layer=act_layer)
        #    for i in range(depth)
        #])
        self.blocks = nn.ModuleList()
        for i in range(depth):
            layer = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                          drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                          norm_layer=norm_layer, act_layer=act_layer)
            self.blocks.append(copy.deepcopy(layer))
        self.norm = norm_layer(embed_dim)
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()
        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()
        # Weight init
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x, latent_graph):
        x = self.feature_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None: #ViT中就是None
            x = torch.cat((cls_token, x), dim=1) 
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        # attn_weights = []
        tem = x
        for layer_block in self.blocks:
            # tem, weights = layer_block(tem)
            tem, _ = layer_block(tem, latent_graph)
            # attn_weights.append(weights)
        x = self.norm(tem)
        # attn_weights = get_weight(attn_weights)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0]), x # ,attn_weights
        else:
            return x[:, 0], x[:, 1]# , attn_weights
        
    def forward(self, x, edge_index, edge_weight, coordinate):
        # latent, attn_weights = self.forward_features(x)
        if self.use_gcn:
            # _ 表示的是 gcn 所生成的嵌入，现在改为 attention 的 output 作为 node 的嵌入
            latent_graph, _ = self.gcn(x, edge_index, edge_weight, coordinate)

            # 加上来自通讯图的结构偏置项
            # latent_communicate_graph = self.communicate_gcn(latent_graph, edge_index, edge_weight)
            # latent_graph = latent_graph + 0.5 * latent_communicate_graph

        latent, embeddings = self.forward_features(x, latent_graph)
        # latent_graph = torch.squeeze(latent_graph, dim=3)
        # latent_graph = torch.squeeze(latent_graph, dim=1)
        # latent = latent + latent_graph
        embeddings = embeddings.reshape(embeddings.shape[0], -1)
        if self.head_dist is not None: 
            latent, latent_dist = self.head(latent[0]), self.head_dist(latent[1])
            if self.training and not torch.jit.is_scripting():
                return latent, latent_dist
            else:
                return (latent+latent_dist) / 2
        else:
            pre = self.head(latent) 
        return latent, pre, embeddings # , attn_weights, 

def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)  

def scTrans_model(num_classes, num_genes, mask, embed_dim=48,depth=2,num_heads=4,has_logits: bool = True, use_gcn=False):
    model = Transformer(num_classes=num_classes,
                        num_genes=num_genes, 
                        mask = mask,
                        embed_dim=embed_dim,
                        depth=depth,
                        num_heads=num_heads,
                        drop_ratio=0.5, attn_drop_ratio=0.5, drop_path_ratio=0.5,
                        representation_size=embed_dim if has_logits else None, use_gcn=use_gcn)
    return model

