import torch.nn as nn
import math
from timm.layers import trunc_normal_
import torch
from ultralytics.nn.modules.block import PSABlock, C2PSA, C2f, C3, Bottleneck
from ultralytics.nn.modules.conv import Conv


# 定义一个通道压缩注意力模块类
class ChannelReductionAttention(nn.Module):
    def __init__(self, dim1, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., pool_ratio=2):
        super().__init__()

        # 确保dim1可以被head数量整除
        assert dim1 % num_heads == 0, f"dim {dim1} should be divided by num_heads {num_heads}."

        self.dim1 = dim1
        self.pool_ratio = pool_ratio  # 用于池化的比例
        self.num_heads = num_heads  # 注意力头数
        head_dim = dim1 // num_heads  # 每个注意力头的维度

        # 设置缩放因子，如果未提供qk_scale，则使用head_dim的倒数平方根
        self.scale = qk_scale or head_dim ** -0.5

        # 定义查询（q）、键（k）、值（v）的线性层
        self.q = nn.Linear(dim1, self.num_heads, bias=qkv_bias)
        self.k = nn.Linear(dim1, self.num_heads, bias=qkv_bias)
        self.v = nn.Linear(dim1, dim1, bias=qkv_bias)

        # 定义注意力和投影的dropout层
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim1, dim1)
        self.proj_drop = nn.Dropout(proj_drop)

        # 定义池化和卷积操作，平均池化降低空间维度，卷积保持通道数
        self.pool = nn.AvgPool2d(pool_ratio, pool_ratio)
        self.sr = nn.Conv2d(dim1, dim1, kernel_size=1, stride=1)  # 1x1卷积保持输入和输出通道一致

        # 定义LayerNorm和激活函数
        self.norm = nn.LayerNorm(dim1)
        self.act = nn.GELU()

        # 初始化权重
        self.apply(self._init_weights)

    # 定义初始化函数，适用于线性层、LayerNorm和卷积层
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)  # 截断正态分布初始化
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)  # 偏置初始化为0
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)  # LayerNorm的偏置初始化为0
            nn.init.constant_(m.weight, 1.0)  # LayerNorm的权重初始化为1
        elif isinstance(m, nn.Conv2d):
            # 使用Kaiming方法初始化卷积层
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()  # 偏置初始化为0

    # 前向传播过程
    def forward(self, x):

        n_, _, h_, w_ = x.shape

        x = x.flatten(2).transpose(1, 2)

        B, N, C = x.shape  # 获取batch大小，序列长度和通道数

        # 计算查询q，将输入x通过线性层生成多头的查询向量
        q = self.q(x).reshape(B, N, self.num_heads).permute(0, 2, 1).unsqueeze(-1)

        # 将输入x调整为卷积所需的形状，并通过池化和卷积层处理
        x_ = x.permute(0, 2, 1).reshape(B, C, h_, w_)
        x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)

        # 归一化并激活处理后的x_
        x_ = self.norm(x_)
        x_ = self.act(x_)

        # 计算键k和值v，类似于查询q的过程
        k = self.k(x_).reshape(B, -1, self.num_heads).permute(0, 2, 1).unsqueeze(-1)
        v = self.v(x_).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # 计算注意力得分，使用缩放因子进行缩放，然后应用softmax
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)  # 加入dropout

        # 将注意力分数和v相乘，得到注意力加权输出
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # 投影输出并添加投影的dropout
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x.permute(0, 2, 1).reshape(n_, -1, h_, w_)
        return x


class PSABlock_CRA(PSABlock):

    def __init__(self, c, qk_dim =16 , pdim=32, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__(c)
        self.ffn = ChannelReductionAttention(c)


class C2PSA_CRA(C2PSA):

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__(c1, c2)
        assert c1 == c2
        self.c = int(c1 * e)
        self.m = nn.Sequential(*(PSABlock_CRA(self.c, qk_dim =16 , pdim=32) for _ in range(n)))


class Bottleneck_CRA(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = ChannelReductionAttention(c_)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck_CRA(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

# 在c3k=True时，使用C3k2_PCFN特征融合，为false的时候我们使用普通的Bottleneck提取特征
class C3k2_CRA(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


if __name__ == '__main__':
    CRA = ChannelReductionAttention(256)
    #创建一个输入张量
    batch_size = 8
    input_tensor=torch.randn(batch_size, 256, 64, 64 )
    #运行模型并打印输入和输出的形状
    output_tensor =CRA(input_tensor)
    print("Input shape:",input_tensor.shape)
    print("0utput shape:",output_tensor.shape)
