import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文题目：Dual-domain strip attention for image restoration
# 论文链接：https://www.sciencedirect.com/science/article/pii/S0893608023006974
# 官方github：https://github.com/c-yn/DSANet/blob/main/Desnowing/models/layers.py
# 代码改进者：一勺汤

class spatial_strip_att(nn.Module):
    def __init__(self, dim, kernel=5, group=2, H=True) -> None:
        """
        初始化空间条带注意力模块。

        参数:
        dim (int): 输入特征图的通道数。
        kernel (int): 卷积核的大小，默认为 5。
        group (int): 分组卷积的组数，默认为 2。
        H (bool): 若为 True，则在水平方向上应用注意力；若为 False，则在垂直方向上应用注意力，默认为 True。
        """
        super().__init__()
        self.k = kernel
        pad = kernel // 2
        # 根据 H 的值确定卷积核的形状
        self.kernel = (1, kernel) if H else (kernel, 1)
        # 根据 H 的值确定填充的形状
        self.padding = (kernel//2, 1) if H else (1, kernel//2)

        self.group = group
        # 根据 H 的值选择不同的填充方式
        self.pad = nn.ReflectionPad2d((pad, pad, 0, 0)) if H else nn.ReflectionPad2d((0, 0, pad, pad))
        # 1x1 卷积用于生成滤波器
        self.conv = nn.Conv2d(dim, group * kernel, kernel_size=1, stride=1, bias=False)
        # 自适应平均池化，将特征图池化为 1x1 大小
        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        # 激活函数，用于生成滤波器的权重
        self.filter_act = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播过程。

        参数:
        x (torch.Tensor): 输入的特征图，形状为 (n, c, h, w)。

        返回:
        torch.Tensor: 经过注意力机制处理后的特征图。
        """
        # 对输入特征图进行自适应平均池化
        filter = self.ap(x)
        # 通过 1x1 卷积生成滤波器
        filter = self.conv(filter)
        n, c, h, w = x.shape
        # 对输入特征图进行填充和展开操作
        x = F.unfold(self.pad(x), kernel_size=self.kernel).reshape(n, self.group, c // self.group, self.k, h * w)

        n, c1, p, q = filter.shape
        # 对滤波器进行形状调整
        filter = filter.reshape(n, c1 // self.k, self.k, p * q).unsqueeze(2)
        # 对滤波器应用激活函数
        filter = self.filter_act(filter)

        # 计算注意力加权后的特征图
        out = torch.sum(x * filter, dim=3).reshape(n, c, h, w)
        return out


class SSA(nn.Module):
    def __init__(self, dim, group=1, kernel=7) -> None:
        """
        初始化空间条带注意力模块的组合模块。

        参数:
        dim (int): 输入特征图的通道数。
        group (int): 分组卷积的组数。
        kernel (int): 卷积核的大小。
        """
        super().__init__()
        # 水平方向的空间条带注意力模块
        self.H_spatial_att = spatial_strip_att(dim, group=group, kernel=kernel)
        # 垂直方向的空间条带注意力模块
        self.W_spatial_att = spatial_strip_att(dim, group=group, kernel=kernel, H=False)
        # 可学习的缩放因子
        self.gamma = nn.Parameter(torch.zeros(dim, 1, 1))
        # 可学习的偏移因子
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        """
        前向传播过程。

        参数:
        x (torch.Tensor): 输入的特征图，形状为 (n, c, h, w)。

        返回:
        torch.Tensor: 经过组合注意力机制处理后的特征图。
        """
        # 先通过水平方向的注意力模块
        out = self.H_spatial_att(x)
        # 再通过垂直方向的注意力模块
        out = self.W_spatial_att(out)
        # 应用缩放和偏移因子
        return self.gamma * out + x * self.beta


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class Bottleneck_SSA(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = SSA(c_)
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
        self.m = nn.Sequential(*(Bottleneck_SSA(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

# 在c3k=True时，使用Bottleneck_LLSKM特征融合，为false的时候我们使用普通的Bottleneck提取特征
class C3k2_SSA(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


if __name__ == "__main__":
    # 定义输入特征图的参数
    dim = 64
    group = 2
    kernel = 5
    # 创建 SSA 模型实例
    model = SSA(dim)
    # 生成随机输入特征图
    x = torch.randn(1, dim, 27, 32)
    # 进行前向传播
    output = model(x)
    print("输出特征图的形状:", output.shape)