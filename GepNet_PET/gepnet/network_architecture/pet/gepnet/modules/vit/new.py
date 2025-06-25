import torch
import torch.nn as nn
from torch.nn import functional as F

from ..cnn import *
from torch.nn.init import trunc_normal_
import math
class ADLA(nn.Module):
    def __init__(self, dim, num_patches, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=1, adla_num=343, **kwargs):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_patches = num_patches
        print(self.num_patches)
        window_size = (int(math.ceil(pow(self.num_patches, 1/3))), int(math.ceil(pow(self.num_patches, 1/3))), int(math.ceil(pow(self.num_patches, 1/3))))
        print("window_size:", window_size)
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv3d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.adla_num = adla_num
        self.dwc = nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=(3, 3, 3), padding=1, groups=dim)
        self.an_bias = nn.Parameter(torch.zeros(num_heads, adla_num, 7, 7, 7))
        self.na_bias = nn.Parameter(torch.zeros(num_heads, adla_num, 7, 7, 7))

        self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, adla_num, window_size[0] // sr_ratio, 1, 1))
        self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, adla_num, 1, window_size[1] // sr_ratio, 1))
        self.ad_bias = nn.Parameter(torch.zeros(1, num_heads, adla_num, 1, 1, window_size[2] // sr_ratio))

        self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window_size[0], 1, 1, adla_num))
        self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window_size[1], 1, adla_num))
        self.da_bias = nn.Parameter(torch.zeros(1, num_heads, 1, 1, window_size[2], adla_num))
        trunc_normal_(self.an_bias, std=.02)
        trunc_normal_(self.na_bias, std=.02)
        trunc_normal_(self.ah_bias, std=.02)
        trunc_normal_(self.aw_bias, std=.02)
        trunc_normal_(self.ad_bias, std=.02)
        trunc_normal_(self.ha_bias, std=.02)
        trunc_normal_(self.wa_bias, std=.02)
        trunc_normal_(self.da_bias, std=.02)
        pool_size = int(math.ceil(pow(adla_num, 1/3)))
        print("pool_size:", pool_size)
        self.pool = nn.AdaptiveAvgPool3d(output_size=(pool_size, pool_size, pool_size))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, H, W, D):
        b, n, c = x.shape  # 1,4096,32
        num_heads = self.num_heads  # 8
        head_dim = c // num_heads   #  4
        q = self.q(x)   # 1,4096,32

        if self.sr_ratio > 1:  #  1
            x_ = x.permute(0, 2, 1).reshape(b, c, H, W)
            x_ = self.sr(x_).reshape(b, c, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
        else:
            # 1,256,32 -> 2,1,256,32
            kv = self.kv(x).reshape(b, -1, 2, c).permute(2, 0, 1, 3) #2,1,4096,32
        k, v = kv[0], kv[1]  # 1,4096,32
        #  1,256,32 -> 1,16,16,32 -> 1,32,16,16 -> 1,32,7,7 -> 1,32,49 -> 1,49 32
        adla_tokens = self.pool(q.reshape(b, H, W, D, c).permute(0, 4, 1, 2, 3)).reshape(b, c, -1).permute(0, 2, 1) #1,343,32
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)   # 1,4096,32 -> 1,8,4096,4
        k = k.reshape(b, n // self.sr_ratio ** 2, num_heads, head_dim).permute(0, 2, 1, 3)  # 1,8,4096,4
        v = v.reshape(b, n // self.sr_ratio ** 2, num_heads, head_dim).permute(0, 2, 1, 3)  # 1,8,4096,4
        adla_tokens = adla_tokens.reshape(b, self.adla_num, num_heads, head_dim).permute(0, 2, 1, 3)  # 1,8,343,4

        kv_size = (self.window_size[0] // self.sr_ratio, self.window_size[1] // self.sr_ratio, self.window_size[2] // self.sr_ratio)  # (16,16,16)
        position_bias1 = nn.functional.interpolate(self.an_bias, size=kv_size, mode='trilinear')  # 8,343,16,16,16
        position_bias1 = position_bias1.reshape(1, num_heads, self.adla_num, -1).repeat(b, 1, 1, 1)  # 1，8，343，4096

        position_bias2 = (self.ah_bias + self.aw_bias + + self.ad_bias).reshape(1, num_heads, self.adla_num, -1).repeat(b, 1, 1, 1)  # 1，8，343，4096
        position_bias = position_bias1 + position_bias2  # 1，8，343，4096
        adla_attn = self.softmax((adla_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
        adla_attn = self.attn_drop(adla_attn)  # 1，8，343，4096
        adla_v = adla_attn @ v   # 1,8,343,4

        adla_bias1 = nn.functional.interpolate(self.na_bias, size=self.window_size, mode='trilinear')  # 8，343，16，16,16
        adla_bias1 = adla_bias1.reshape(1, num_heads, self.adla_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1) # 1,8,4096,343
        adla_bias2 = (self.ha_bias + self.wa_bias + self.da_bias).reshape(1, num_heads, -1, self.adla_num).repeat(b, 1, 1, 1)# 1,8,4096,343
        adla_bias = adla_bias1 + adla_bias2 # 1,8,4096,343
        q_attn = self.softmax((q * self.scale) @ adla_tokens.transpose(-2, -1) + adla_bias)# 1,8,4096,343
        q_attn = self.attn_drop(q_attn)  # 1,8,4096,343
        x = q_attn @ adla_v   # 1,8,4096,4

        x = x.transpose(1, 2).reshape(b, n, c)  # 1，4096，32
        v = v.transpose(1, 2).reshape(b, H // self.sr_ratio, W // self.sr_ratio, D // self.sr_ratio, c).permute(0, 4, 1, 2, 3)  # 1，32，16，16,16
        if self.sr_ratio > 1:
            v = nn.functional.interpolate(v, size=(H, W, D), mode='trilinear')   # 1，32，16，16
        x = x + self.dwc(v).permute(0, 2, 3, 4, 1).reshape(b, n, c) #1,4096,32

        x = self.proj(x) # 1，4096,32
        x = self.proj_drop(x) # 1，4096,32
        return x

class AttBlock_ViT_Parallel_DLKA3D(nn.Module):
    """
    A transformer block, based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    With LKA and spatial attention
    """

    def __init__(
        self,
        vit_block: nn.Module,
        lka_block: nn.Module,
        input_size: int,
        hidden_size: int,
        proj_size: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        *args,
        **kwargs
    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage.
            hidden_size: dimension of hidden layer.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            pos_embed: bool argument to determine if positional embedding is used.

        """

        super().__init__()
        # print(f"Using {epa_block}")

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm = nn.LayerNorm(hidden_size)
        self.bnorm = nn.BatchNorm3d(hidden_size)

        self.gamma = nn.Parameter(torch.ones(hidden_size, 1, 1, 1), requires_grad=True)
        self.attn = vit_block(
            dim=hidden_size,
            num_patches=input_size,
        )

        self.delta = nn.Parameter(torch.ones(hidden_size, 1, 1, 1), requires_grad=True)
        self.lka = lka_block(d_model=hidden_size)

        self.conv3 = UnetResBlock(
            3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch"
        )
        if dropout_rate:
            self.conv1 = nn.Sequential(
                nn.Dropout3d(dropout_rate, False),
                nn.Conv3d(hidden_size, hidden_size, 1),
            )
        else:
            self.conv1 = nn.Conv3d(hidden_size, hidden_size, 1)

        self.pos_embed = nn.Parameter(1e-6 + torch.zeros(1, input_size, hidden_size))

    def vit_attn(self, x):
        B, C, H, W, D = x.shape
        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)
        x = x + self.pos_embed
        attn = self.attn(self.norm(x), H, W, D)
        return attn.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)

    def forward(self, x):
        x_lka = self.delta * self.lka(x)
        x_vit = self.gamma * self.vit_attn(x)
        x = x * (1 + x_lka + x_vit)
        x = self.bnorm(x)
        x = x + self.conv1(self.conv3(x))
        return x


class ChannelAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads=4,
        qkv_bias=False,
        use_norm=False,
        use_temperature=False,
        dropout=0,
        *args,
        **kwargs
    ):
        super().__init__()

        self.num_heads = num_heads
        self.use_norm = use_norm
        self.use_dropout = dropout
        self.use_temperature = use_temperature

        if dropout:
            self.dropout = nn.Dropout(dropout)

        # qkvv are 3 linear layers (query_shared, key_shared, value_spatial, value_channel)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.out = nn.Linear(hidden_size, hidden_size)

        if use_norm:
            self.norm = nn.LayerNorm(hidden_size)
        if use_temperature:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

    def vit_attention(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, v_CA = qkv[0], qkv[1], qkv[2]

        query = query.transpose(-2, -1)
        key = key.transpose(-2, -1)
        v_CA = v_CA.transpose(-2, -1)

        query = torch.nn.functional.normalize(query, dim=-1)
        key = torch.nn.functional.normalize(key, dim=-1)

        attn_CA = query @ key.transpose(-2, -1)
        if self.use_temperature:
            attn_CA *= self.temperature
        attn_CA = attn_CA.softmax(dim=-1)
        if self.use_dropout:
            attn_CA = self.dropout(attn_CA)
        x_CA = (attn_CA @ v_CA).permute(0, 3, 1, 2).reshape(B, N, C)
        return self.norm(x_CA) if self.use_norm else x_CA

    def forward(self, x, B_in, C_in, H, W, D):
        x = self.vit_attention(x, H, W, D)  # 1,4096,32
        return self.out(x)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"temperature"}


class SpatialAttention(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_heads=4,
        qkv_bias=False,
        proj_size=8**3,
        use_norm=False,
        use_temperature=False,
        dropout=0,
        *args,
        **kwargs
    ):
        super().__init__()
        self.num_heads = num_heads
        self.use_norm = use_norm
        self.use_dropout = dropout
        self.use_temperature = use_temperature

        # E and F are projection matrices with shared weights used in spatial attention module to project
        # keys and values from HWD-dimension to P-dimension
        self.E = self.F = nn.Linear(input_size, proj_size)

        # qkvv are 3 linear layers (query_shared, key_shared, value_spatial, value_channlka)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.out = nn.Linear(hidden_size, hidden_size)

        if dropout:
            self.dropout = nn.Dropout(dropout)
        if use_temperature:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        if use_norm:
            self.norm = nn.LayerNorm(hidden_size)

    def vit_attention(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, v_SA = qkv[0], qkv[1], qkv[2]

        query = query.transpose(-2, -1)
        key = key.transpose(-2, -1)
        v_SA = v_SA.transpose(-2, -1)

        k_projected = self.E(key)
        v_SA_projected = self.F(v_SA)

        query = torch.nn.functional.normalize(query, dim=-1)


        attn_SA = query.permute(0, 1, 3, 2) @ k_projected
        if self.use_temperature:
            attn_SA *= self.temperature
        attn_SA = attn_SA.softmax(dim=-1)
        if self.use_dropout:
            attn_SA = self.dropout(attn_SA)
        x_SA = (
            (attn_SA @ v_SA_projected.transpose(-2, -1))
            .permute(0, 3, 1, 2)
            .reshape(B, N, C)
        )
        return self.norm(x_SA) if self.use_norm else x_SA

    def forward(self, x, B_in, C_in, H, W, D):
        x = self.vit_attention(x)
        x = self.out(x)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"temperature"}


####################(Sequential)#####################
HybAttn_DLKA3D_Parallel_SpatialViT = partial(
    AttBlock_ViT_Parallel_DLKA3D,
    vit_block=ADLA,
    lka_block=DLKA3D_Block_onTensor,
)

HybAttn_DLKA3D_Parallel_ChannelViT = partial(
    AttBlock_ViT_Parallel_DLKA3D,
    vit_block=ADLA,
    lka_block=DLKA3D_Block_onTensor,
)
