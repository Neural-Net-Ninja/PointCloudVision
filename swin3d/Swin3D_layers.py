import torch
import torch.nn as nn
import MinkowskiEngine as ME
from typing import Optional, Tuple, List, Union

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

class Mlp(nn.Module):
    """
    Multi-layer Perceptron (MLP) module.

    :param in_features: Number of input features.
    :type in_features: int
    :param hidden_features: Number of hidden features. Defaults to `None`.
    :type hidden_features: Optional[int], optional
    :param out_features: Number of output features. Defaults to `None`.
    :type out_features: Optional[int], optional
    :param act_layer: Activation layer. Defaults to `nn.GELU`.
    :type act_layer: nn.Module, optional
    :param drop: Dropout rate. Defaults to `0.0`.
    :type drop: float, optional
    """
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Mlp module.

        :param x: Input tensor.
        :type x: torch.Tensor
        :return: Output tensor.
        :rtype: torch.Tensor
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class GridCoordsDown(nn.Module):
    """
    Downsample the grid coordinates and keep the nearest point to the average point of the downsampled grid.

    :param stride: Stride for downsampling.
    :type stride: int
    """
    def __init__(self, stride: int) -> None:
        super().__init__()
        self.stride = stride
        self.avg_pool = ME.MinkowskiAvgPooling(
            kernel_size=self.stride, stride=self.stride, dimension=3
        )
        self.unpool = ME.MinkowskiPoolingTranspose(
            kernel_size=stride, stride=stride, dimension=3
        )
        self.max_pool = ME.MinkowskiMaxPooling(
            kernel_size=self.stride, stride=self.stride, dimension=3
        )

    def forward(self, coords_sp: ME.SparseTensor, sp: ME.SparseTensor, return_map: bool = False) -> Union[ME.SparseTensor, Tuple[ME.SparseTensor, torch.Tensor]]:
        """
        Forward pass of the GridCoordsDown module.

        :param coords_sp: SparseTensor for coordinates.
        :type coords_sp: ME.SparseTensor
        :param sp: SparseTensor for features.
        :type sp: ME.SparseTensor
        :param return_map: Whether to return the downsample map. Defaults to `False`.
        :type return_map: bool, optional
        :return: Downsampled SparseTensor or a tuple of downsampled SparseTensor and downsample map.
        :rtype: Union[ME.SparseTensor, Tuple[ME.SparseTensor, torch.Tensor]]
        """
        device = sp.C.device
        N = sp.shape[0]
        avg_coords_sp = self.avg_pool(coords_sp)
        dist_sp = self.unpool(avg_coords_sp) - coords_sp
        dist = dist_sp.F
        dist = -torch.sqrt((dist**2).sum(dim=1)).unsqueeze(1)
        dist_sp = assign_feats(dist_sp, dist)
        min_dist_sp = self.max_pool(dist_sp)
        map_pair = sp.coordinate_manager.kernel_map(
            dist_sp.coordinate_map_key,
            min_dist_sp.coordinate_map_key,
            stride=self.stride,
            kernel_size=self.stride,
            is_pool=True,
        )[0]
        in_map, out_map = map_pair
        broad_min_dist_sp = self.unpool(min_dist_sp)
        mask = (broad_min_dist_sp.F == dist_sp.F).squeeze(1)
        in_map = in_map[mask].long()
        out_map = out_map[mask].long()
        downsample_map = torch.zeros(N, dtype=torch.long, device=device) - 1
        downsample_map[out_map] = in_map
        assert (downsample_map >= 0).all()
        assert (dist_sp.F[downsample_map] == min_dist_sp.F).all()
        new_coords = coords_sp.F[downsample_map]
        new_coords_sp = assign_feats(sp, new_coords)
        if return_map:
            return new_coords_sp, downsample_map
        else:
            return new_coords_sp

def get_offset(batch: torch.Tensor) -> torch.Tensor:
    """
    Get the offset for each batch.

    :param batch: Batch tensor.
    :type batch: torch.Tensor
    :return: Offset tensor.
    :rtype: torch.Tensor
    """
    offset = []
    bs = batch.max() + 1
    for i in range(bs):
        offset.append(torch.sum(batch == i))
    offset = torch.cuda.IntTensor(offset)
    offset = offset.cumsum(dim=0).int()
    return offset

class GridDownsample(nn.Module):
    """
    Downsample voxel using stride and grid maxpooling with kernel size.

    :param in_channels: Number of input channels.
    :type in_channels: int
    :param out_channels: Number of output channels.
    :type out_channels: int
    :param kernel_size: Size of the kernel. Defaults to `2`.
    :type kernel_size: int, optional
    :param stride: Stride for downsampling. Defaults to `2`.
    :type stride: int, optional
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sp_pool = ME.MinkowskiMaxPooling(
            kernel_size=kernel_size, stride=stride, dimension=3
        )
        self.coords_pool = GridCoordsDown(stride=stride)
        self.norm = SparseTensorLayerNorm(in_channels)
        self.linear = SparseTensorLinear(in_channels, out_channels)

    def forward(self, sp: ME.SparseTensor, coords_sp: ME.SparseTensor) -> Tuple[ME.SparseTensor, ME.SparseTensor]:
        """
        Forward pass of the GridDownsample module.

        :param sp: SparseTensor for features.
        :type sp: ME.SparseTensor
        :param coords_sp: SparseTensor for coordinates.
        :type coords_sp: ME.SparseTensor
        :return: Tuple of downsampled SparseTensors for features and coordinates.
        :rtype: Tuple[ME.SparseTensor, ME.SparseTensor]
        """
        sp_down = self.sp_pool(self.linear(self.norm(sp)))
        coords_sp_down = self.coords_pool(coords_sp, sp_down)
        return sp_down, coords_sp_down

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, stride={self.stride}, in_channels={self.in_channels}, out_channels={self.out_channels}"

class GridKNNDownsample(nn.Module):
    """
    Downsample voxel using stride and KNN maxpooling.

    :param in_channels: Number of input channels.
    :type in_channels: int
    :param out_channels: Number of output channels.
    :type out_channels: int
    :param kernel_size: Size of the kernel. Defaults to `2`.
    :type kernel_size: int, optional
    :param stride: Stride for downsampling. Defaults to `2`.
    :type stride: int, optional
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = 16
        self.sp_pool = ME.MinkowskiMaxPooling(
            kernel_size=stride, stride=stride, dimension=3
        )
        self.coords_pool = GridCoordsDown(stride=stride)
        self.norm = nn.LayerNorm(in_channels)
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.pool = nn.MaxPool1d(self.k)

    def forward(self, sp: ME.SparseTensor, coords_sp: ME.SparseTensor) -> Tuple[ME.SparseTensor, ME.SparseTensor]:
        """
        Forward pass of the GridKNNDownsample module.

        :param sp: SparseTensor for features.
        :type sp: ME.SparseTensor
        :param coords_sp: SparseTensor for coordinates.
        :type coords_sp: ME.SparseTensor
        :return: Tuple of downsampled SparseTensors for features and coordinates.
        :rtype: Tuple[ME.SparseTensor, ME.SparseTensor]
        """
        sp_down = self.sp_pool(sp)
        coords_sp_down = self.coords_pool(coords_sp, sp_down)
        offset = get_offset(sp.C[:, 0])
        n_offset = get_offset(sp_down.C[:, 0])

        xyz = coords_sp.F[:, 1:4].detach().contiguous()
        n_xyz = coords_sp_down.F[:, 1:4].detach().contiguous()
        feats = query_knn_feature(self.k, xyz, n_xyz, sp.F, offset, n_offset)
        m, k, c = feats.shape
        feats = (
            self.linear(self.norm(feats.view(m * k, c)).view(m, k, c))
            .transpose(1, 2)
            .contiguous()
        )
        feats = self.pool(feats).squeeze(-1)
        sp = assign_feats(sp_down, feats.float())
        coords_sp = coords_sp_down
        return sp, coords_sp

    def extra_repr(self) -> str:
        return f"kernel_size={self.k}, stride={self.stride}, in_channels={self.in_channels}, out_channels={self.out_channels}"

class Upsample(nn.Module):
    """
    Upsample using trilinear interpolation followed by attention block.

    :param in_channels: Number of input channels.
    :type in_channels: int
    :param out_channels: Number of output channels.
    :type out_channels: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param window_size: Size of the window.
    :type window_size: int
    :param quant_size: Quantization size.
    :type quant_size: int
    :param attn: Whether to use attention. Defaults to `True`.
    :type attn: bool, optional
    :param up_k: Number of nearest neighbors for upsampling. Defaults to `3`.
    :type up_k: int, optional
    :param cRSE: Coordinate Reference System Encoding. Defaults to `'XYZ_RGB'`.
    :type cRSE: str, optional
    :param fp16_mode: Floating point precision mode. Defaults to `0`.
    :type fp16_mode: int, optional
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int,
        window_size: int,
        quant_size: int,
        attn: bool = True,
        up_k: int = 3,
        cRSE: str = "XYZ_RGB",
        fp16_mode: int = 0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.linear1 = nn.Sequential(
            nn.LayerNorm(out_channels), nn.Linear(out_channels, out_channels)
        )
        self.linear2 = nn.Sequential(
            nn.LayerNorm(in_channels), nn.Linear(in_channels, out_channels)
        )
        self.up_k = up_k
        self.attn = attn and window_size > 0
        if self.attn:
            self.attn_block = WindowAttention(
                dim=out_channels,
                window_size=window_size,
                quant_size=quant_size,
                num_heads=num_heads,
                cRSE=cRSE,
                fp16_mode=fp16_mode,
            )

    def forward(self, sp: ME.SparseTensor, coords_sp: ME.SparseTensor, sp_up: ME.SparseTensor, coords_sp_up: ME.SparseTensor) -> ME.SparseTensor:
        """
        Forward pass of the Upsample module.

        :param sp: SparseTensor for features.
        :type sp: ME.SparseTensor
        :param coords_sp: SparseTensor for coordinates.
        :type coords_sp: ME.SparseTensor
        :param sp_up: SparseTensor for upsampled features.
        :type sp_up: ME.SparseTensor
        :param coords_sp_up: SparseTensor for upsampled coordinates.
        :type coords_sp_up: ME.SparseTensor
        :return: Upsampled SparseTensor.
        :rtype: ME.SparseTensor
        """
        feats = sp.F
        support_feats = sp_up.F
        xyz = coords_sp.F[:, 1:4].detach().contiguous()
        support_xyz = coords_sp_up.F[:, 1:4].detach().contiguous()
        offset = get_offset(sp.C[:, 0])
        support_offset = get_offset(sp_up.C[:, 0])

        feats = self.linear1(support_feats) + knn_linear_interpolation(
            xyz, support_xyz, self.linear2(feats), offset, support_offset, K=self.up_k
        )
        sp_up = assign_feats(sp_up, feats)
        if self.attn:
            attn_args = (coords_sp_up, sp_up)
            sp_up = self.attn_block(sp_up, attn_args)
        return sp_up

    def extra_repr(self) -> str:
        return f"up_k={self.up_k}, in_channels={self.in_channels}, out_channels={self.out_channels}, attn={self.attn}"

class WindowAttention(nn.Module):
    """
    Window based multi-head self attention (W-MSA) module with cRSE.
    Designed for sparse structure. It supports both shifted and non-shifted windows.

    :param dim: Number of input channels.
    :type dim: int
    :param window_size: Size of the window.
    :type window_size: Tuple[int, int]
    :param quant_size: Quantization size.
    :type quant_size: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param qkv_bias: Whether to add a learnable bias to query, key, value. Defaults to `True`.
    :type qkv_bias: bool, optional
    :param qk_scale: Override default qk scale of head_dim ** -0.5 if set. Defaults to `None`.
    :type qk_scale: Optional[float], optional
    :param attn_drop: Dropout ratio of attention weight. Defaults to `0.0`.
    :type attn_drop: float, optional
    :param proj_drop: Dropout ratio of output. Defaults to `0.0`.
    :type proj_drop: float, optional
    :param cRSE: Coordinate Reference System Encoding. Defaults to `'XYZ_RGB'`.
    :type cRSE: str, optional
    :param fp16_mode: Floating point precision mode. Defaults to `0`.
    :type fp16_mode: int, optional
    """
    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int],
        quant_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        cRSE: str = "XYZ_RGB",
        fp16_mode: int = 0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.fp16_mode = fp16_mode

        table_offsets = []
        self.cRSE = cRSE
        if "XYZ" in cRSE:
            table_offsets.append(0)
        if "RGB" in cRSE:
            table_offsets.append(1)
        if "NORM" in cRSE:
            table_offsets.append(2)

        self.table_offsets = table_offsets

        self.quant_size = quant_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, feats: torch.Tensor, attn_args: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Forward function for the attention mechanism.

        :param feats: Input features with shape (N, C).
        :type feats: torch.Tensor
        :param attn_args: Arguments for computing attention, including offsets, indices, and coordinates.
        :type attn_args: Tuple[torch.Tensor, ...]
        :return: Updated features after applying the attention mechanism.
        :rtype: torch.Tensor
        """
        num_v, _ = feats.shape
        num_sc = self.dim // self.num_heads

        (
            x_offset,
            y_offset,
            m2w_indices,
            w_sizes,
            w2n_indices,
            n2n_indices,
            w2m_indices,
            n_coords,
        ) = attn_args

        # Query, Key, Value
        qkv = self.qkv(feats)
        qkv = (
            qkv.reshape(num_v, 3, self.num_heads, num_sc)
            .permute(1, 0, 2, 3)
            .contiguous()
        )
        query, key, value = qkv[0], qkv[1], qkv[2]  # [N, num_heads, C//num_heads]
        query = query * self.scale

        table_offsets = torch.IntTensor(self.table_offsets).cuda()
        query_table, key_table, value_table = [], [], []
        n_cRSE = []
        if "XYZ" in self.cRSE:
            n_xyz = n_coords[:, 0:3]
            n_xyz = n_xyz * self.quant_size
            n_cRSE.append(n_xyz)
            query_table.append(self.query_xyz_table.view(-1))
            key_table.append(self.key_xyz_table.view(-1))
            value_table.append(self.value_xyz_table.view(-1))
        if "RGB" in self.cRSE:
            n_rgb = n_coords[:, 3:6]
            n_rgb = n_rgb * self.color_quant_size
            n_cRSE.append(n_rgb)
            query_table.append(self.query_rgb_table.view(-1))
            key_table.append(self.key_rgb_table.view(-1))
            value_table.append(self.value_rgb_table.view(-1))
        if "NORM" in self.cRSE:
            n_norm = n_coords[:, 6:9]
            n_norm = n_norm * self.normal_quant_size
            n_cRSE.append(n_norm)
            query_table.append(self.query_norm_table.view(-1))
            key_table.append(self.key_norm_table.view(-1))
            value_table.append(self.value_norm_table.view(-1))

        n_cRSE = torch.cat(n_cRSE, dim=1)

        indices = [m2w_indices, w_sizes, w2m_indices, w2n_indices, n2n_indices, n_cRSE]
        query_table = torch.cat(query_table)
        key_table = torch.cat(key_table)
        value_table = torch.cat(value_table)

        if self.fp16_mode == 0:
            # do not use fp16
            # cast q,k,v to fp32 in forward and backward
            fp16_mode = PrecisionMode.HALF_NONE
        elif self.fp16_mode == 1:
            # use fp16 only in forward
            fp16_mode = PrecisionMode.HALF_FORWARD
        elif self.fp16_mode == 2:
            # use fp16 both in forward and backward
            fp16_mode = PrecisionMode.HALF_ALL

        updated_values = SelfAttnAIOFunction.apply(
            query,
            key,
            value,
            query_table,
            key_table,
            value_table,
            table_offsets,
            indices,
            PosEmb.SEPARATE,
            TableDims.D0,
            IndexMode.INDIRECT,
            fp16_mode,
        )

        updated_values = updated_values.flatten(1)
        updated_feats = updated_values.view(num_v, self.dim)

        updated_feats = self.proj(updated_feats)
        updated_feats = self.proj_drop(updated_feats)  # [N, C]

        return updated_feats

import torch
import torch.nn as nn
import MinkowskiEngine as ME
from typing import Optional, Tuple, List, Union

class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block for 3D point cloud processing.

    :param dim: Number of input channels.
    :type dim: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param window_size: Local window size.
    :type window_size: int
    :param quant_size: Quantization size for finer cRSE table.
    :type quant_size: int
    :param drop_path: Stochastic depth rate. Defaults to `0.0`.
    :type drop_path: float, optional
    :param mlp_ratio: Ratio of MLP hidden dim to embedding dim. Defaults to `4.0`.
    :type mlp_ratio: float, optional
    :param qkv_bias: If True, add a learnable bias to query, key, value. Defaults to `True`.
    :type qkv_bias: bool, optional
    :param qk_scale: Override default qk scale of head_dim ** -0.5 if set. Defaults to `None`.
    :type qk_scale: Optional[float], optional
    :param act_layer: Activation layer. Defaults to `nn.GELU`.
    :type act_layer: nn.Module, optional
    :param norm_layer: Normalization layer. Defaults to `nn.LayerNorm`.
    :type norm_layer: nn.Module, optional
    :param cRSE: cRSE mode. Defaults to `'XYZ_RGB'`.
    :type cRSE: str, optional
    :param fp16_mode: fp16 mode for attention module. Defaults to `0`.
    :type fp16_mode: int, optional
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        quant_size: int,
        drop_path: float = 0.0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        cRSE: str = "XYZ_RGB",
        fp16_mode: int = 0,
    ) -> None:
        super().__init__()
        self.window_size = window_size

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
            quant_size=quant_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            cRSE=cRSE,
            fp16_mode=fp16_mode,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer
        )

    def forward(self, feats: torch.Tensor, attn_args: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Forward pass of the SwinTransformerBlock.

        :param feats: Input features with shape (N, C).
        :type feats: torch.Tensor
        :param attn_args: Arguments for computing attention.
        :type attn_args: Tuple[torch.Tensor, ...]
        :return: Updated features after applying the attention mechanism.
        :rtype: torch.Tensor
        """
        short_cut = feats
        feats = self.norm1(feats)
        feats = self.attn(feats, attn_args)  # [N, C]

        feats = short_cut + self.drop_path(feats)
        feats = feats + self.drop_path(self.mlp(self.norm2(feats)))

        return feats


class BasicLayer(nn.Module):
    """
    A basic Swin3D layer for one stage.

    :param dim: Number of input channels.
    :type dim: int
    :param depth: Number of blocks.
    :type depth: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param window_size: Local window size.
    :type window_size: int
    :param quant_size: Quantization size for finer cRSE table.
    :type quant_size: int
    :param out_channels: Number of output channels. Defaults to `None`.
    :type out_channels: Optional[int], optional
    :param mlp_ratio: Ratio of MLP hidden dim to embedding dim. Defaults to `4.0`.
    :type mlp_ratio: float, optional
    :param qkv_bias: If True, add a learnable bias to query, key, value. Defaults to `True`.
    :type qkv_bias: bool, optional
    :param qk_scale: Override default qk scale of head_dim ** -0.5 if set. Defaults to `None`.
    :type qk_scale: Optional[float], optional
    :param drop_path: Stochastic depth rate. Defaults to `0.0`.
    :type drop_path: Union[float, List[float]], optional
    :param norm_layer: Normalization layer. Defaults to `nn.LayerNorm`.
    :type norm_layer: nn.Module, optional
    :param downsample: Downsample layer at the end of the layer. Defaults to `None`.
    :type downsample: Optional[nn.Module], optional
    :param down_stride: Stride for downsampling. Defaults to `2`.
    :type down_stride: int, optional
    :param cRSE: cRSE mode. Defaults to `'XYZ_RGB'`.
    :type cRSE: str, optional
    :param fp16_mode: fp16 mode for attention module. Defaults to `0`.
    :type fp16_mode: int, optional
    """
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        quant_size: int,
        out_channels: Optional[int] = None,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_path: Union[float, List[float]] = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        downsample: Optional[nn.Module] = None,
        down_stride: int = 2,
        cRSE: str = "XYZ_RGB",
        fp16_mode: int = 0,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.depth = depth
        self.dim = dim
        self.num_heads = num_heads
        self.quant_size = quant_size
        self.cRSE = cRSE
        self.fp16_mode = fp16_mode

        self.shift_size = window_size // 2
        # build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim,
                    num_heads,
                    window_size,
                    quant_size,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    norm_layer=norm_layer,
                    cRSE=cRSE,
                    fp16_mode=fp16_mode,
                )
                for i in range(depth)
            ]
        )

        self.pool = ME.MinkowskiMaxPooling(
            kernel_size=self.window_size, stride=self.window_size, dimension=3
        )

        if downsample is not None:
            if out_channels is None:
                out_channels = dim * 2
            self.downsample = downsample(
                dim, out_channels, kernel_size=down_stride, stride=down_stride
            )
        else:
            self.downsample = None

    def get_map_pair(self, sp: ME.SparseTensor) -> Tuple[torch.Tensor, int]:
        """
        Use Minkowski pool to calculate windows and get the mapping from voxel to window.

        :param sp: Input SparseTensor.
        :type sp: ME.SparseTensor
        :return: Mapping pair and number of windows.
        :rtype: Tuple[torch.Tensor, int]
        """
        window_size = [self.window_size] * 3
        pool_sp = self.pool(sp)
        windows = pool_sp.C
        window_N = windows.shape[0]

        stride_in = sp.coordinate_map_key.get_tensor_stride()
        x, y, z = [
            torch.arange(window_size[i], device=self.device) * stride_in[i]
            for i in range(3)
        ]
        x, y, z = torch.meshgrid(x, y, z)
        i = torch.zeros_like(x, device=self.device)
        local_window = torch.stack([i, x, y, z], dim=-1).flatten(0, -2)
        all_windows = windows.unsqueeze(1) + local_window.unsqueeze(0)
        all_windows = all_windows.flatten(0, -2).int()
        cm = sp.coordinate_manager
        query_key, (map, inverse_map) = cm.insert_and_map(
            all_windows, tensor_stride=stride_in
        )
        map_pair = cm.kernel_map(query_key, sp.coordinate_map_key, kernel_size=1)[0]
        return map_pair, window_N

    def get_window_mapping(self, sp: ME.SparseTensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate the relationship in the window.

        :param sp: Input SparseTensor.
        :type sp: ME.SparseTensor
        :return: Window mapping information.
        :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        """
        map_pair, window_N = self.get_map_pair(sp)
        window_size = self.window_size
        nW = window_size**3
        in_map, out_map = map_pair
        in_map, sort_idx = torch.sort(in_map)
        out_map = out_map[sort_idx]
        sort_idx = out_map.long()
        inv_sort_idx = torch.zeros_like(sort_idx)
        inv_sort_idx[sort_idx] = torch.arange(
            sort_idx.shape[0], dtype=sort_idx.dtype, device=self.device
        )
        N = window_N * nW
        v2w_mask = torch.zeros(N, dtype=torch.bool, device=self.device)
        w_id = (
            torch.arange(window_N, dtype=torch.long, device=self.device)
            .unsqueeze(1)
            .repeat(1, nW)
            .view(-1)
        )
        w_w_id = (
            torch.arange(nW, dtype=torch.long, device=self.device)
            .unsqueeze(0)
            .repeat(window_N, 1)
            .view(-1)
        )
        v2w_mask[in_map.long()] = True
        nempty_num = v2w_mask.view(-1, nW).sum(dim=-1)
        w_id = w_id[in_map.long()]
        w_w_id = w_w_id[in_map.long()]
        w_w_xyz = torch.stack(
            [
                w_w_id // window_size // window_size,
                w_w_id // window_size % window_size,
                w_w_id % window_size,
            ],
            dim=-1,
        )
        return w_w_id, w_w_xyz, nempty_num, sort_idx, inv_sort_idx

    def get_index01(self, sp: ME.SparseTensor, local_xyz: torch.Tensor, colors: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Calculate the arguments for sparse attention.

        :param sp: Input SparseTensor.
        :type sp: ME.SparseTensor
        :param local_xyz: Local XYZ coordinates.
        :type local_xyz: torch.Tensor
        :param colors: Color information.
        :type colors: torch.Tensor
        :return: Arguments for sparse attention.
        :rtype: Tuple[torch.Tensor, ...]
        """
        (
            w_w_id,
            w_w_xyz,
            nempty_num,
            n2n_indices,
            inv_sort_idx,
        ) = self.get_window_mapping(sp)
        local_xyz = local_xyz[n2n_indices]
        colors = colors[n2n_indices]
        # recover the relative pos in the voxel
        n_coords = w_w_xyz + local_xyz
        n_coords = torch.cat([n_coords, colors], dim=1)
        (
            x_offset,
            y_offset,
            m2w_indices,
            w_sizes,
            w2n_indices,
            w2m_indices,
        ) = sparse_self_attention(w_w_id, nempty_num, protocol="v2")
        return (
            x_offset,
            y_offset,
            m2w_indices,
            w_sizes,
            w2n_indices,
            n2n_indices,
            w2m_indices,
            n_coords,
        )

    def get_shifted_sp(self, sp: ME.SparseTensor) -> ME.SparseTensor:
        """
        Get the shifted sparse tensor for shift-window.

        :param sp: Input SparseTensor.
        :type sp: ME.SparseTensor
        :return: Shifted SparseTensor.
        :rtype: ME.SparseTensor
        """
        stride_in = sp.coordinate_map_key.get_tensor_stride()
        shift_size = self.shift_size * stride_in[0]
        shifted_C = sp.C.clone()
        shifted_C[:, 1:] += shift_size
        shifted_sp = ME.SparseTensor(
            features=sp.F,
            coordinates=shifted_C,
            device=self.device,
            tensor_stride=stride_in,
        )
        return shifted_sp

    def get_window_pos(self, sp: ME.SparseTensor) -> torch.Tensor:
        """
        Get the window position.

        :param sp: Input SparseTensor.
        :type sp: ME.SparseTensor
        :return: Window position.
        :rtype: torch.Tensor
        """
        stride_in = sp.coordinate_map_key.get_tensor_stride()
        return (sp.C[:, 1:] / stride_in[0]) % self.window_size

    def forward(self, sp: ME.SparseTensor, coords_sp: ME.SparseTensor) -> Tuple[ME.SparseTensor, ME.SparseTensor, ME.SparseTensor]:
        """
        Forward pass of the BasicLayer.

        :param sp: Input SparseTensor.
        :type sp: ME.SparseTensor
        :param coords_sp: Coordinate SparseTensor.
        :type coords_sp: ME.SparseTensor
        :return: Updated SparseTensors.
        :rtype: Tuple[ME.SparseTensor, ME.SparseTensor, ME.SparseTensor]
        """
        colors = coords_sp.F[:, 4:]
        xyz = coords_sp.F[:, :4]
        local_xyz = (xyz - coords_sp.C)[
            :, 1:
        ] / coords_sp.coordinate_map_key.get_tensor_stride()[0]
        self.device = sp.device
        sp_shift = self.get_shifted_sp(sp)

        attn_args = self.get_index01(sp, local_xyz, colors)
        attn_args_shift = self.get_index01(sp_shift, local_xyz, colors)

        feats = sp.F
        for i, blk in enumerate(self.blocks):
            attn_args_blk = attn_args if i % 2 == 0 else attn_args_shift
            feats = blk(feats, attn_args_blk)  # [N, C]

        sp = assign_feats(sp, feats)
        if self.downsample is not None:
            sp_down, coords_sp = self.downsample(sp, coords_sp)
            return sp, sp_down, coords_sp
        else:
            return sp, sp, coords_sp

    def extra_repr(self) -> str:
        """
        Extra representation of the BasicLayer.

        :return: String representation of the BasicLayer.
        :rtype: str
        """
        return f"window_size={self.window_size}, depth={self.depth}, channel={self.dim}, num_heads={self.num_heads}, quant_size={self.quant_size}, cRSE={self.cRSE}, fp16_mode={self.fp16_mode}"