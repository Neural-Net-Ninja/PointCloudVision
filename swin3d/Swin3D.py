import os
import torch
import torch.nn as nn
from typing import List, Literal, Optional, Tuple, Union

from .base_architecture import BaseArchitecture
from .Swin_3D.Mink_layers import MinkConvBNRelu, MinkResBlock
from Swin_3D.Swin3D_layers import GridDownsample, GridKNNDownsample, BasicLayer, Upsample


class Swin3DUNet(BaseArchitecture):
    """
    Swin3DUNet architecture for 3D point cloud processing.

    :param num_classes: Number of output classes.
    :type num_classes: int
    :param in_channels: Number of input channels. Defaults to `None`.
    :type in_channels: int, optional
    :param num_layers: Number of layers in the network. Defaults to `None`.
    :type num_layers: int, optional
    :param depths: List of depths for each layer. Defaults to `None`.
    :type depths: List[int], optional
    :param channels: List of channels for each layer. Defaults to `None`.
    :type channels: List[int], optional
    :param num_heads: List of number of heads for each layer. Defaults to `None`.
    :type num_heads: List[int], optional
    :param window_sizes: List of window sizes for each layer. Defaults to `None`.
    :type window_sizes: List[int], optional
    :param quant_size: Quantization size. Defaults to `None`.
    :type quant_size: int, optional
    :param drop_path_rate: Drop path rate for stochastic depth. Defaults to `None`.
    :type drop_path_rate: float, optional
    :param up_k: Upsampling factor. Defaults to `None`.
    :type up_k: int, optional
    :param stem_transformer: Whether to use a transformer in the stem layer. Defaults to `True`.
    :type stem_transformer: bool, optional
    :param first_down_stride: Stride for the first downsampling layer. Defaults to `None`.
    :type first_down_stride: int, optional
    :param upsample: Upsampling method. Defaults to `'linear'`.
    :type upsample: Literal['linear'], optional
    :param knn_down: Whether to use KNN downsampling. Defaults to `True`.
    :type knn_down: bool, optional
    :param cRSE: Coordinate Reference System Encoding. Defaults to `'XYZ_RGB'`.
    :type cRSE: Literal['XYZ_RGB'], optional
    :param fp16_mode: Floating point precision mode. Defaults to `None`.
    :type fp16_mode: int, optional
    :param version: Version of the architecture. Defaults to `'1.0.0'`.
    :type version: str, optional
    :param kwargs: Additional keyword arguments.
    :type kwargs: dict
    """
    def __init__(self,
                 num_classes: int,
                 in_channels: int = None,
                 num_layers: int = None,
                 depths: Optional[List[int]] = None,
                 channels: Optional[List[int]] = None,
                 num_heads: Optional[List[int]] = None,
                 window_sizes: Optional[List[int]] = None,
                 quant_size: int = None,
                 drop_path_rate: float = None,
                 up_k: int = None,
                 stem_transformer: bool = True,
                 first_down_stride: int = None,
                 upsample: Literal['linear'] = 'linear',
                 knn_down: bool = True,
                 cRSE: Literal['XYZ_RGB'] = 'XYZ_RGB',
                 fp16_mode: int = None,
                 version: str = '1.0.0',
                 **kwargs
                 ) -> None:
        super().__init__(input_size="variable", version=version)
        self.config['in_channels'] = 9 if in_channels is None else in_channels
        self.config['num_layers'] = 5 if num_layers is None else num_layers
        self.config['depths'] = [2, 4, 9, 4, 4] if depths is None else depths
        self.config['channels'] = [48, 96, 192, 384, 384] if channels is None else channels
        self.config['num_heads'] = [6, 6, 12, 24, 24] if num_heads is None else num_heads
        self.config['window_sizes'] = [5, 7, 7, 7, 7] if window_sizes is None else window_sizes
        self.config['quant_size'] = 4 if quant_size is None else quant_size
        self.config['up_k'] = 3 if up_k is None else up_k
        self.config['first_down_stride'] = 3 if first_down_stride is None else first_down_stride
        self.config['upsample'] = upsample
        self.config['cRSE'] = cRSE
        self.config['fp16_mode'] = 2 if fp16_mode is None else fp16_mode
        self.drop_path_rate = 0.0 if drop_path_rate is None else drop_path_rate

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        downsample = GridKNNDownsample if knn_down else GridDownsample

        self.cRSE = cRSE
        if stem_transformer:
            self.stem_layer = MinkConvBNRelu(
                in_channels=in_channels,
                out_channels=channels[0],
                kernel_size=3,
                stride=1,
            )
            self.layer_start = 0
        else:
            self.stem_layer = nn.Sequential(
                MinkConvBNRelu(
                    in_channels=in_channels,
                    out_channels=channels[0],
                    kernel_size=3,
                    stride=1,
                ),
                MinkResBlock(
                    in_channels=channels[0],
                    out_channels=channels[0]
                )
            )
            self.downsample = downsample(
                channels[0],
                channels[1],
                kernel_size=first_down_stride,
                stride=first_down_stride
            )
            self.layer_start = 1

        self.layers = nn.ModuleList([
            BasicLayer(
                dim=channels[i],
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_sizes[i],
                quant_size=quant_size,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=downsample if i < num_layers - 1 else None,
                down_stride=first_down_stride if i == 0 else 2,
                out_channels=channels[i + 1] if i < num_layers - 1 else None,
                cRSE=cRSE,
                fp16_mode=fp16_mode) for i in range(self.layer_start, num_layers)])

        up_attn = 'attn' in upsample

        self.upsamples = nn.ModuleList([
            Upsample(channels[i], channels[i - 1], num_heads[i - 1], window_sizes[i - 1], quant_size, attn=up_attn,
                     up_k=up_k, cRSE=cRSE, fp16_mode=fp16_mode)
            for i in range(num_layers - 1, 0, -1)])

        self.classifier = nn.Sequential(
            nn.Linear(channels[0], channels[0]),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
            nn.Linear(channels[0], num_classes)
        )
        self.num_classes = num_classes
        self.init_weights()

    def forward(self, sp: torch.Tensor, coords_sp: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Swin3DUNet.

        :param sp: MinkowskiEngine SparseTensor for feature input.
        :type sp: torch.Tensor
        :param coords_sp: MinkowskiEngine SparseTensor for position and feature embedding.
        :type coords_sp: torch.Tensor
        :return: Output tensor.
        :rtype: torch.Tensor
        """
        sp_stack = []
        coords_sp_stack = []
        sp = self.stem_layer(sp)
        if self.layer_start > 0:
            sp_stack.append(sp)
            coords_sp_stack.append(coords_sp)
            sp, coords_sp = self.downsample(sp, coords_sp)

        for i, layer in enumerate(self.layers):
            coords_sp_stack.append(coords_sp)
            sp, sp_down, coords_sp = layer(sp, coords_sp)
            sp_stack.append(sp)
            assert (coords_sp.C == sp_down.C).all()
            sp = sp_down

        sp = sp_stack.pop()
        coords_sp = coords_sp_stack.pop()
        for i, upsample in enumerate(self.upsamples):
            sp_i = sp_stack.pop()
            coords_sp_i = coords_sp_stack.pop()
            sp = upsample(sp, coords_sp, sp_i, coords_sp_i)
            coords_sp = coords_sp_i

        output = self.classifier(sp.F)
        return output

    def init_weights(self) -> None:
        """
        Initialize the weights in the backbone.
        """
        def _init_weights(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

    def load_pretrained_model(self, ckpt: str, skip_first_conv: bool = True, verbose: bool = True) -> None:
        """
        Load a pretrained model from a checkpoint.

        :param ckpt: Path to the checkpoint file.
        :type ckpt: str
        :param skip_first_conv: Whether to skip the first convolutional layer. Defaults to `True`.
        :type skip_first_conv: bool, optional
        :param verbose: Whether to print verbose messages. Defaults to `True`.
        :type verbose: bool, optional
        """
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            weights = checkpoint['state_dict']
            matched_weights = load_state_with_same_shape(self, weights, skip_first_conv=skip_first_conv, verbose=verbose)
            self.load_state_dict(matched_weights, strict=False)
            if verbose:
                print(f"=> loaded weight '{ckpt}'")
        else:
            if verbose:
                print(f"=> no weight found at '{ckpt}'")


class Swin3DEncoder(nn.Module):
    """
    Swin3DEncoder architecture for 3D point cloud processing.

    :param depths: List of depths for each layer.
    :type depths: List[int]
    :param channels: List of channels for each layer.
    :type channels: List[int]
    :param num_heads: List of number of heads for each layer.
    :type num_heads: List[int]
    :param window_sizes: List of window sizes for each layer.
    :type window_sizes: List[int]
    :param quant_size: Quantization size.
    :type quant_size: int
    :param drop_path_rate: Drop path rate for stochastic depth. Defaults to `0.2`.
    :type drop_path_rate: float, optional
    :param num_layers: Number of layers in the network. Defaults to `5`.
    :type num_layers: int, optional
    :param stem_transformer: Whether to use a transformer in the stem layer. Defaults to `True`.
    :type stem_transformer: bool, optional
    :param first_down_stride: Stride for the first downsampling layer. Defaults to `2`.
    :type first_down_stride: int, optional
    :param knn_down: Whether to use KNN downsampling. Defaults to `True`.
    :type knn_down: bool, optional
    :param in_channels: Number of input channels. Defaults to `6`.
    :type in_channels: int, optional
    :param cRSE: Coordinate Reference System Encoding. Defaults to `'XYZ_RGB'`.
    :type cRSE: Literal['XYZ_RGB'], optional
    :param fp16_mode: Floating point precision mode. Defaults to `0`.
    :type fp16_mode: int, optional
    """
    def __init__(self,
                 depths: List[int],
                 channels: List[int],
                 num_heads: List[int],
                 window_sizes: List[int],
                 quant_size: int,
                 drop_path_rate: float = 0.2,
                 num_layers: int = 5,
                 stem_transformer: bool = True,
                 first_down_stride: int = 2,
                 knn_down: bool = True,
                 in_channels: int = 6,
                 cRSE: Literal['XYZ_RGB'] = 'XYZ_RGB',
                 fp16_mode: int = 0) -> None:
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        downsample = GridKNNDownsample if knn_down else GridDownsample

        self.cRSE = cRSE
        if stem_transformer:
            self.stem_layer = MinkConvBNRelu(
                in_channels=in_channels,
                out_channels=channels[0],
                kernel_size=3,
                stride=1,
            )
            self.layer_start = 0
        else:
            self.stem_layer = nn.Sequential(
                MinkConvBNRelu(
                    in_channels=in_channels,
                    out_channels=channels[0],
                    kernel_size=3,
                    stride=1,
                ),
                MinkResBlock(
                    in_channels=channels[0],
                    out_channels=channels[0]
                )
            )
            self.downsample = downsample(
                channels[0],
                channels[1],
                kernel_size=first_down_stride,
                stride=first_down_stride
            )
            self.layer_start = 1

        self.layers = nn.ModuleList([
            BasicLayer(
                dim=channels[i],
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_sizes[i],
                quant_size=quant_size,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=downsample if i < num_layers - 1 else None,
                down_stride=first_down_stride if i == 0 else 2,
                out_channels=channels[i + 1] if i < num_layers - 1 else None,
                cRSE=cRSE,
                fp16_mode=fp16_mode) for i in range(self.layer_start, num_layers)])

        self.init_weights()

    def forward(self, sp: torch.Tensor, coords_sp: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass of the Swin3DEncoder.

        :param sp: MinkowskiEngine SparseTensor for feature input.
        :type sp: torch.Tensor
        :param coords_sp: MinkowskiEngine SparseTensor for position and feature embedding.
        :type coords_sp: torch.Tensor
        :return: List of output tensors.
        :rtype: List[torch.Tensor]
        """
        sp_stack = []
        coords_sp_stack = []
        sp = self.stem_layer(sp)
        if self.layer_start > 0:
            sp_stack.append(sp)
            coords_sp_stack.append(coords_sp)
            sp, coords_sp = self.downsample(sp, coords_sp)

        for i, layer in enumerate(self.layers):
            coords_sp_stack.append(coords_sp)
            sp, sp_down, coords_sp = layer(sp, coords_sp)
            sp_stack.append(sp)
            assert (coords_sp.C == sp_down.C).all()
            sp = sp_down
        return sp_stack

    def init_weights(self) -> None:
        """
        Initialize the weights in the backbone.
        """
        def _init_weights(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

    def load_pretrained_model(self, ckpt: str, skip_first_conv: bool = True, verbose: bool = True) -> None:
        """
        Load a pretrained model from a checkpoint.

        :param ckpt: Path to the checkpoint file.
        :type ckpt: str
        :param skip_first_conv: Whether to skip the first convolutional layer. Defaults to `True`.
        :type skip_first_conv: bool, optional
        :param verbose: Whether to print verbose messages. Defaults to `True`.
        :type verbose: bool, optional
        """
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            weights = checkpoint['state_dict']
            matched_weights = load_state_with_same_shape(self, weights, skip_first_conv=skip_first_conv, verbose=verbose)
            self.load_state_dict(matched_weights, strict=False)
            if verbose:
                print(f"=> loaded weight '{ckpt}'")
        else:
            if verbose:
                print(f"=> no weight found at '{ckpt}'")