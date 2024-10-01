import torch
import torch.nn as nn
import MinkowskiEngine as ME
from typing import Optional

def assign_feats(sp: ME.SparseTensor, x: torch.Tensor) -> ME.SparseTensor:
    """
    Assign features to a MinkowskiEngine SparseTensor.

    :param sp: SparseTensor to which features are assigned.
    :type sp: ME.SparseTensor
    :param x: Features to assign.
    :type x: torch.Tensor
    :return: SparseTensor with assigned features.
    :rtype: ME.SparseTensor
    """
    return ME.SparseTensor(
        features=x.float(),
        coordinate_map_key=sp.coordinate_map_key,
        coordinate_manager=sp.coordinate_manager,
    )

class MinkConvBN(nn.Module):
    """
    Minkowski Convolution followed by Batch Normalization.

    :param in_channels: Number of input channels.
    :type in_channels: int
    :param out_channels: Number of output channels.
    :type out_channels: int
    :param kernel_size: Size of the convolution kernel. Defaults to `3`.
    :type kernel_size: int, optional
    :param stride: Stride of the convolution. Defaults to `1`.
    :type stride: int, optional
    :param dilation: Dilation rate of the convolution. Defaults to `1`.
    :type dilation: int, optional
    :param bias: Whether to use a bias term. Defaults to `False`.
    :type bias: bool, optional
    :param dimension: Dimension of the convolution. Defaults to `3`.
    :type dimension: int, optional
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = False,
        dimension: int = 3,
    ) -> None:
        super().__init__()
        self.conv_layers = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                bias=bias,
                dimension=dimension,
            ),
            ME.MinkowskiBatchNorm(out_channels),
        )

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        """
        Forward pass of the MinkConvBN module.

        :param x: Input SparseTensor.
        :type x: ME.SparseTensor
        :return: Output SparseTensor.
        :rtype: ME.SparseTensor
        """
        return self.conv_layers(x)

class MinkConvBNRelu(nn.Module):
    """
    Minkowski Convolution followed by Batch Normalization and ReLU activation.

    :param in_channels: Number of input channels.
    :type in_channels: int
    :param out_channels: Number of output channels.
    :type out_channels: int
    :param kernel_size: Size of the convolution kernel. Defaults to `3`.
    :type kernel_size: int, optional
    :param stride: Stride of the convolution. Defaults to `1`.
    :type stride: int, optional
    :param dilation: Dilation rate of the convolution. Defaults to `1`.
    :type dilation: int, optional
    :param bias: Whether to use a bias term. Defaults to `False`.
    :type bias: bool, optional
    :param dimension: Dimension of the convolution. Defaults to `3`.
    :type dimension: int, optional
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = False,
        dimension: int = 3,
    ) -> None:
        super().__init__()
        self.conv_layers = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                bias=bias,
                dimension=dimension,
            ),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True),
        )

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        """
        Forward pass of the MinkConvBNRelu module.

        :param x: Input SparseTensor.
        :type x: ME.SparseTensor
        :return: Output SparseTensor.
        :rtype: ME.SparseTensor
        """
        x = self.conv_layers(x)
        if x.F.dtype == torch.float16:
            x = assign_feats(x, x.F.float())
        return x

class MinkDeConvBNRelu(nn.Module):
    """
    Minkowski Deconvolution followed by Batch Normalization and ReLU activation.

    :param in_channels: Number of input channels.
    :type in_channels: int
    :param out_channels: Number of output channels.
    :type out_channels: int
    :param kernel_size: Size of the deconvolution kernel.
    :type kernel_size: int
    :param stride: Stride of the deconvolution.
    :type stride: int
    :param dilation: Dilation rate of the deconvolution. Defaults to `1`.
    :type dilation: int, optional
    :param bias: Whether to use a bias term. Defaults to `False`.
    :type bias: bool, optional
    :param dimension: Dimension of the deconvolution. Defaults to `3`.
    :type dimension: int, optional
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int = 1,
        bias: bool = False,
        dimension: int = 3,
    ) -> None:
        super().__init__()
        self.conv_layers = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                bias=bias,
                dimension=dimension,
            ),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(),
        )

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        """
        Forward pass of the MinkDeConvBNRelu module.

        :param x: Input SparseTensor.
        :type x: ME.SparseTensor
        :return: Output SparseTensor.
        :rtype: ME.SparseTensor
        """
        return self.conv_layers(x)

class MinkResBlock(nn.Module):
    """
    Minkowski Residual Block.

    :param in_channels: Number of input channels.
    :type in_channels: int
    :param out_channels: Number of output channels.
    :type out_channels: int
    :param stride: Stride of the convolution. Defaults to `1`.
    :type stride: int, optional
    :param dilation: Dilation rate of the convolution. Defaults to `1`.
    :type dilation: int, optional
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dilation: int = 1) -> None:
        super().__init__()
        self.conv1 = ME.MinkowskiConvolution(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            bias=False,
            dimension=3,
        )
        self.norm1 = ME.MinkowskiBatchNorm(out_channels)
        self.conv2 = ME.MinkowskiConvolution(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            dilation=dilation,
            bias=False,
            dimension=3,
        )
        self.norm2 = ME.MinkowskiBatchNorm(out_channels)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        """
        Forward pass of the MinkResBlock module.

        :param x: Input SparseTensor.
        :type x: ME.SparseTensor
        :return: Output SparseTensor.
        :rtype: ME.SparseTensor
        """
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out += residual
        out = self.relu(out)
        return out

class SparseTensorLinear(nn.Module):
    """
    Linear layer for SparseTensor.

    :param in_channels: Number of input channels.
    :type in_channels: int
    :param out_channels: Number of output channels.
    :type out_channels: int
    :param bias: Whether to use a bias term. Defaults to `False`.
    :type bias: bool, optional
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False) -> None:
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, sp: ME.SparseTensor) -> ME.SparseTensor:
        """
        Forward pass of the SparseTensorLinear module.

        :param sp: Input SparseTensor.
        :type sp: ME.SparseTensor
        :return: Output SparseTensor.
        :rtype: ME.SparseTensor
        """
        x = self.linear(sp.F)
        return assign_feats(sp, x.float())

class SparseTensorLayerNorm(nn.Module):
    """
    Layer Normalization for SparseTensor.

    :param dim: Dimension of the input tensor.
    :type dim: int
    """
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, sp: ME.SparseTensor) -> ME.SparseTensor:
        """
        Forward pass of the SparseTensorLayerNorm module.

        :param sp: Input SparseTensor.
        :type sp: ME.SparseTensor
        :return: Output SparseTensor.
        :rtype: ME.SparseTensor
        """
        x = self.norm(sp.F)
        return assign_feats(sp, x.float())

class MinkResBlock_v2(nn.Module):
    """
    Minkowski Residual Block version 2.

    :param in_channels: Number of input channels.
    :type in_channels: int
    :param out_channels: Number of output channels.
    :type out_channels: int
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        d_2 = out_channels // 4
        self.conv1 = nn.Sequential(
            SparseTensorLinear(in_channels, d_2, bias=False),
            ME.MinkowskiBatchNorm(d_2),
            ME.MinkowskiReLU(),
        )
        self.unary_2 = nn.Sequential(
            SparseTensorLinear(d_2, out_channels, bias=False),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(),
        )
        self.spconv = ME.MinkowskiConvolution(
            in_channels=d_2,
            out_channels=d_2,
            kernel_size=5,
            stride=1,
            dilation=1,
            bias=False,
            dimension=3,
        )
        if in_channels != out_channels:
            self.shortcut_op = nn.Sequential(
                SparseTensorLinear(in_channels, out_channels, bias=False),
                ME.MinkowskiBatchNorm(out_channels),
            )
        else:
            self.shortcut_op = nn.Identity()

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        """
        Forward pass of the MinkResBlock_v2 module.

        :param x: Input SparseTensor.
        :type x: ME.SparseTensor
        :return: Output SparseTensor.
        :rtype: ME.SparseTensor
        """
        shortcut = x
        x = self.conv1(x)
        x = self.spconv(x)
        x = self.unary_2(x)
        shortcut = self.shortcut_op(shortcut)
        x += shortcut
        return x

class MinkResBlock_BottleNeck(nn.Module):
    """
    Minkowski Residual Block with Bottleneck.

    :param in_channels: Number of input channels.
    :type in_channels: int
    :param out_channels: Number of output channels.
    :type out_channels: int
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        bottle_neck = out_channels // 4
        self.conv1x1a = MinkConvBNRelu(
            in_channels, bottle_neck, kernel_size=1, stride=1
        )
        self.conv3x3 = MinkConvBNRelu(bottle_neck, bottle_neck, kernel_size=3, stride=1)
        self.conv1x1b = MinkConvBN(bottle_neck, out_channels, kernel_size=1, stride=1)
        if in_channels != out_channels:
            self.conv1x1c = MinkConvBN(
                in_channels, out_channels, kernel_size=1, stride=1
            )
        else:
            self.conv1x1c = None
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        """
        Forward pass of the MinkResBlock_BottleNeck module.

        :param x: Input SparseTensor.
        :type x: ME.SparseTensor
        :return: Output SparseTensor.
        :rtype: ME.SparseTensor
        """
        residual = x
        out = self.conv1x1a(x)
        out = self.conv3x3(out)
        out = self.conv1x1b(out)
        if self.conv1x1c is not None:
            residual = self.conv1x1c(residual)
        out = self.relu(out + residual)
        return out