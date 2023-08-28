import PIL
import time, json
import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import torch.nn.init as init
from einops import rearrange, repeat
import collections
import torch.nn as nn


def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# 等于 PreNorm
class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# 等于 FeedForward
class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class CrossAttention(nn.Module):
    def __init__(self, q_dim, kv_dim, dim, heads=8, droupout=0.1) -> None:
        '''
            给定q和kv, 利用attention的方式返回对q的新空间编码new_q
            其中q的输入维度为(batch, seq, q_dim), 最终输出维度为(batch, seq, dim)
        '''
        super().__init__()
        self.heads = heads
        self.scale = kv_dim ** -0.5 #1/sqrt(dim)

        self.to_q = nn.Linear(q_dim, dim, bias=True) # dim = heads * per_dim
        self.to_k = nn.Linear(kv_dim, dim, bias=True)
        self.to_v = nn.Linear(kv_dim, dim, bias=True)

        self.nn1 = nn.Linear(dim, dim)
        self.do1 = nn.Dropout(droupout)



    def forward(self, x, y, mask=None):
        # x shape is (batch, seq1, q_dim)
        # y shape is (batch, seq2, kv_dim)
        b, n, _, h = *x.shape, self.heads
        by, ny, _, hy= *x.shape, self.heads
        assert b == by

        # q,k,v获取
        qheads, kheads, vheads = self.to_q(x), self.to_k(y), self.to_v(y) # qheads,kvheads shape all is (batch, seq, dim)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (qheads, kheads, vheads))  # split into multi head attentions
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)  # follow the softmax,q,d,v equation in the paper

        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # product of v times whatever inside softmax
        out = rearrange(out, 'b h n d -> b n (h d)')  # concat heads into one matrix, ready for next encoder block
        out = self.nn1(out)
        out = self.do1(out) # (batch, seq1, dim)
        return out

class Attention(nn.Module):

    def __init__(self, dim, heads, dim_heads, dropout):
        super().__init__()
        inner_dim = dim_heads * heads
        self.heads = heads
        self.scale = dim_heads ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x, mask = None):
        # x:[b,n,dim]
        b, n, _, h = *x.shape, self.heads

        # get qkv tuple:([b,n,head_num*head_dim],[...],[...])
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # split q,k,v from [b,n,head_num*head_dim] -> [b,head_num,n,head_dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # transpose(k) * q / sqrt(head_dim) -> [b,head_num,n,n]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        # mask value: -inf
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        # softmax normalization -> attention matrix
        attn = dots.softmax(dim=-1)
        # value * attention matrix -> output
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        # cat all output -> [b, n, head_num*head_dim]
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Attention(dim, heads=heads, dim_heads=dim_heads, dropout=dropout))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attention, mlp in self.layers:
            x = attention(x, mask=mask)  # go to attention
            x = mlp(x)  # go to MLP_Block
        return x

class CrossTransformer(nn.Module):
    def __init__(self, dim, heads, mlp_dim, drouput) -> None:
        '''
            输入x和y, 将x在y空间中进行transformer的encoder生成x的新表示 new_x
            输入x和y, 将y在x空间中进行transformer的encoder生成y的新表示 new_y
            输入x和y维度应该相同 否则无法做residule 输入dim,输出dim
            x shape(batch, seq1, dim)
            y shape(batch, seq2, dim)
        '''
        super().__init__()
        # CrossTransformer 目前只支持one layer, 即depth=1
        self.norm_x = nn.LayerNorm(dim)
        self.norm_y = nn.LayerNorm(dim)
        self.norm_x2 = nn.LayerNorm(dim)
        self.norm_y2 = nn.LayerNorm(dim)

        self.cross_attention_x = CrossAttention(dim, dim, dim, heads=heads, droupout=drouput)
        self.cross_attention_y = CrossAttention(dim, dim, dim, heads=heads, droupout=drouput)

        self.mlp_x = MLP_Block(dim, mlp_dim, dropout=drouput)
        self.mlp_y = MLP_Block(dim, mlp_dim, dropout=drouput)

    def forward(self, x, y, mask=None):
        assert mask==None
        # x和y会分别作为q以及对应的kv进行cross-transformer
        #1. 保留shortcut
        shortcut_x = x
        shortcut_y = y

        #2. 做prenorm
        x = self.norm_x(x)
        y = self.norm_y(y)

        #3. 分别做cross-attention
        x = self.cross_attention_x(x, y, mask=mask)
        y = self.cross_attention_y(y, x, mask=mask)

        #4. short cut收
        x = shortcut_x + x
        y = shortcut_y + y

        #5. 做mlp 和 residual
        x = x + self.mlp_x(self.norm_x2(x))
        y = y + self.mlp_y(self.norm_y2(y))

        return x, y



class SE(nn.Module):

    def __init__(self, in_chnls, ratio):
        super(SE, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Conv2d(in_chnls, in_chnls//ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls//ratio, in_chnls, 1, 1, 0)

    def forward(self, x):
        out = self.squeeze(x)
        out = self.compress(out)
        out = F.relu(out)
        out = self.excitation(out)
        return F.sigmoid(out)



class HSINet(nn.Module):
    def __init__(self, params):
        super(HSINet, self).__init__()
        self.params = params
        net_params = params['net']
        data_params = params['data']

        num_classes = data_params.get("num_classes", 16)
        patch_size = data_params.get("patch_size", 13)
        self.spectral_size = data_params.get("spectral_size", 200)

        depth = net_params.get("depth", 1)
        heads = net_params.get("heads", 8)
        mlp_dim = net_params.get("mlp_dim", 8)
        dropout = net_params.get("dropout", 0)
        conv2d_out = 64
        dim = net_params.get("dim", 64)
        dim_heads = dim
        mlp_head_dim = dim
        
        image_size = patch_size * patch_size

        self.pixel_patch_embedding = nn.Linear(conv2d_out, dim)

        self.local_trans_pixel = Transformer(dim=dim, depth=depth, heads=heads, dim_heads=dim_heads, mlp_dim=mlp_dim, dropout=dropout)
        self.new_image_size = image_size
        self.pixel_pos_embedding = nn.Parameter(torch.randn(1, self.new_image_size+1, dim))
        self.pixel_pos_scale = nn.Parameter(torch.ones(1) * 0.01)

        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=self.spectral_size, out_channels=conv2d_out, kernel_size=(3, 3), padding=(1,1)),
            nn.BatchNorm2d(conv2d_out),
            nn.ReLU(),
            # featuremap 是在这之后加一层channel上的压缩
            # nn.Conv2d(in_channels=conv2d_out,out_channels=dim,kernel_size=3,padding=1),
            # nn.BatchNorm2d(dim),
            # nn.ReLU()
        )

        self.senet = SE(conv2d_out, 5)

        self.cls_token_pixel = nn.Parameter(torch.randn(1, 1, dim))
        self.to_latent_pixel = nn.Identity()

        self.mlp_head =nn.Linear(dim, num_classes)
        torch.nn.init.xavier_uniform_(self.mlp_head.weight)
        torch.nn.init.normal_(self.mlp_head.bias, std=1e-6)
        self.dropout = nn.Dropout(0.1)

        linear_dim = dim * 2
        self.classifier_mlp = nn.Sequential(
            nn.Linear(dim, linear_dim),
            nn.BatchNorm1d(linear_dim),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(linear_dim, num_classes),
        )

    def encoder_block(self, x):
        '''
        x: (batch, s, w, h), s=spectral, w=weigth, h=height
        '''
        x_pixel = x 

        b, s, w, h = x_pixel.shape
        img = w * h
        x_pixel = self.conv2d_features(x_pixel)


        scale = self.senet(x_pixel)
        # print('scale shape is ', scale.shape)
        # print('pixel shape is ', x_pixel.shape)
        # x_pixel = x_pixel * scale#(batch, image_size, dim)

        #1. reshape
        x_pixel = rearrange(x_pixel, 'b s w h-> b (w h) s') # (batch, w*h, s)

        #2. patch_embedding
        # x_pixel = self.pixel_patch_embedding(x_pixel)

        #3. local transformer
        cls_tokens_pixel = self.cls_token_pixel.expand(x_pixel.shape[0], -1, -1)
        x_pixel = torch.cat((cls_tokens_pixel, x_pixel), dim = 1) #[b,image+1,dim]
        x_pixel = x_pixel + self.pixel_pos_embedding[:,:] * self.pixel_pos_scale
        # x_pixel = x_pixel + self.pixel_pos_embedding[:,:] 
        # x_pixel = self.dropout(x_pixel)

        x_pixel = self.local_trans_pixel(x_pixel) #(batch, image_size+1, dim)

        logit_pixel = self.to_latent_pixel(x_pixel[:,0])

        logit_x = logit_pixel 
        reduce_x = torch.mean(x_pixel, dim=1)
        
        return logit_x, reduce_x

    def forward(self, x,left=None,right=None):
        '''
        x: (batch, s, w, h), s=spectral, w=weigth, h=height

        '''
        logit_x, _ = self.encoder_block(x)
        mean_left, mean_right = None, None
        if left is not None and right is not None:
            _, mean_left = self.encoder_block(left)
            _, mean_right = self.encoder_block(right)

        # return  self.mlp_head(logit_x), mean_left, mean_right 
        return  self.classifier_mlp(logit_x), mean_left, mean_right 