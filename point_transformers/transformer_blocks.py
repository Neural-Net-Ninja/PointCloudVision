__all__ = ['Block', 'BlockSequence']

from collections import OrderedDict
from typing import Any, List, Union

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from pcnn.models.blocks import BatchNorm, DropPath
from pcnn.operations import knn_search
from .grouped_vector_attention import GroupedVectorAttentionV1, GroupedVectorAttentionV2
from .sequential import PointModule, PointSequential
from .serialized_attention import SerializedAttention


class Block(nn.Module):
    """
    Block for Point Transformers with Grouped Vector Attention.

    :param feature_dim: The number of input and output feature channels.
    :type feature_dim: integer
    :param groups: The number of groups for grouped vector attention.
    :type groups: integer
    :param use_attention_v2: Whether to use Grouped Vector Attention V1 or Grouped Vector Attention V2.
    :type use_attention_v2: bool, optional
    :param p_drop_path: Drop path rate for stochastic depth. Defaults to `0`.
    :type p_drop_path: float, optional
    :param enable_checkpoint: Whether to enable gradient checkpointing for the grouped vector attention module.
        Checkpointing works by trading compute for memory. Rather than storing the intermediate activations of the
        grouped vector attention module for computing backward, they are instead recomputed in backward pass.
        Defaults to `False`.
    :type enable_checkpoint: bool, optional
    :param attention_parameters: Parameters to be forwarded to the used attention type. Should be empty for Grouped
        Vector Attention V1 and can include `p_dropout: float`, `value_bias: bool`, `pos_encoding_multiplier: bool`,
        `pos_encoding_bias: bool`, and `use_grouped_linear: bool` for Grouped Vector Attention V2.
    :type attention_parameters: bool or float, optional
    """
    def __init__(self,
                 feature_dim: int,
                 groups: int,
                 use_attention_v2: bool = False,
                 p_drop_path: float = 0.,
                 enable_checkpoint: bool = False,
                 **attention_parameters: Any) -> None:
        super().__init__()
        self.grouped_attention: nn.Module
        if use_attention_v2:
            self.grouped_attention = GroupedVectorAttentionV2(feature_dim,
                                                              groups,
                                                              **attention_parameters)
        else:
            self.grouped_attention = GroupedVectorAttentionV1(feature_dim,
                                                              groups)

        self.feature_projection = nn.Sequential(OrderedDict([
                ('linear', nn.Linear(feature_dim, feature_dim, bias=False)),
                ('bn', BatchNorm(feature_dim)),
                ('relu', nn.ReLU(inplace=True))
        ]))

        self.bn_1 = BatchNorm(feature_dim)
        self.bn_2 = BatchNorm(feature_dim)
        self.linear = nn.Linear(feature_dim, feature_dim, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.enable_checkpoint = enable_checkpoint
        self.drop_path = DropPath(p_drop_path)

    def forward(self,
                coords: torch.Tensor,
                features: torch.Tensor,
                neighbor_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the block.

        :param coords: Input coordinates of shape `(N, 3)`, where `N = number of points`.
        :type coords: torch.Tensor
        :param features: Input features of shape `(N, D)`, where `N = number of points`, and
            `D = number of feature channels`.
        :type features: torch.Tensor
        :param neighbor_indices: Indices of the neighbors of each point of shape `(N, K)`, where `N = number of points`,
            and `K = number of neighbors`.
        :type neighbor_indices: torch.Tensor
        :return: Output features of shape `(N, D)`, where `N = number of points` and `D = number of feature channels`.
        :rtype: torch.Tensor
        """
        # Save the identity tensor for the skip connection.
        identity = features

        # Pass the input through the first fully-connected layer.
        features = self.feature_projection(features)

        # Pass the input through the attention layer.
        if not self.enable_checkpoint:
            # If checkpointing is not enabled, apply the attention layer directly.
            features = self.grouped_attention(coords, features, neighbor_indices)
        else:
            # If checkpointing is enabled, apply the attention layer using checkpointing.
            features = checkpoint(self.grouped_attention, coords, features, neighbor_indices)

        features = self.relu(self.bn_1(features))

        # Pass the input through the second fully-connected layer.
        features = self.bn_2(self.linear(features))

        # Apply the skip connection and the dropout layer.
        features = identity + self.drop_path(features)
        features = self.relu(features)

        return features


class BlockSequence(nn.Module):
    """
    Block Sequence for Point Transformers.

    :param num_layers: Number of grouped vector attention (GVA) blocks.
    :type num_layers: integer
    :param feature_dim: The number of input and output feature channels.
    :type feature_dim: integer
    :param groups: The number of groups for grouped vector attention.
    :type groups: integer
    :param use_attention_v2: Whether to use Grouped Vector Attention V1 or Grouped Vector Attention V2.
    :type use_attention_v2: bool, optional
    :param num_neighbors: The number of neighbors for grouped vector attention. Defaults to `16`.
    :type num_neighbors: integer, optional
    :param enable_checkpoint: Whether to enable checkpointing for the grouped vector attention module.
        Defaults to `False`.
    :type enable_checkpoint: bool, optional
    :param p_drop_path: Drop path rate for stochastic depth. A list of values can be provided to specify different drop
        path rates for each block of the sequence. Defaults to `0`.
    :type p_drop_path: float or List[float], optional
    :param attention_parameters: Parameters to be forwarded to the used attention type. Should be empty for Grouped
        Vector Attention V1 and can include `p_dropout: float`, `value_bias: bool`, `pos_encoding_multiplier: bool`,
        `pos_encoding_bias: bool`, and `use_grouped_linear: bool` for Grouped Vector Attention V2.
    :type attention_parameters:bool or float, optional
    """
    def __init__(self,
                 num_layers: int,
                 feature_dim: int,
                 groups: int,
                 use_attention_v2: bool = False,
                 num_neighbors: int = 16,
                 enable_checkpoint: bool = False,
                 p_drop_path: Union[float, List[float]] = 0.,
                 **attention_parameters: Any) -> None:
        super().__init__()
        self.num_neighbors = num_neighbors
        if isinstance(p_drop_path, list):
            p_drop_paths = p_drop_path
            assert len(p_drop_paths) == num_layers, 'The number of drop path rates must be equal to the number of \
                hidden layers.'
        elif isinstance(p_drop_path, float):
            p_drop_paths = [p_drop_path for _ in range(num_layers)]

        blocks = []
        for layer in range(num_layers):
            block = Block(feature_dim,
                          groups,
                          use_attention_v2=use_attention_v2,
                          p_drop_path=p_drop_paths[layer],
                          enable_checkpoint=enable_checkpoint,
                          **attention_parameters)
            blocks.append((f'block_{layer}', block))
        self.blocks = torch.nn.Sequential(OrderedDict(blocks))

    def forward(self,
                coords: torch.Tensor,
                features: torch.Tensor,
                batch_indices: torch.Tensor,
                point_cloud_sizes: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the block sequence.

        :param coords: Input coordinates of shape `(N, 3)`, where `N = number of points`.
        :type coords: torch.Tensor
        :param features: Input features of shape `(N, D)`, where `N = number of points`, and
            `D = number of feature channels`.
        :type features: torch.Tensor
        :param batch_indices: Indices indicating to which input point cloud each point in the batch belongs. Must have
            shape `(N)` where `N = number of points`.
        :type batch_indices: torch.Tensor
        :param point_cloud_sizes: Number of points in each point cloud in the batch. Must have shape `(B)` where
            `B = batch size`.
        :type point_cloud_sizes: torch.Tensor
        :return: Output features of shape `(N, D)`, where `N = number of points`, and `D = number of feature channels`.
        :rtype: torch.Tensor
        """
        # Retrieve indices of kNN for each point
        neighbor_indices, _ = knn_search(coords, coords, batch_indices, batch_indices, point_cloud_sizes,
                                         point_cloud_sizes, k=self.num_neighbors, return_sorted=False)

        # Pass the input through the block sequence.
        for block in self.blocks:
            features = block(coords, features, neighbor_indices)

        return features

    from typing import Optional, Type

class Block_v3(PointModule):
    """
    A transformer block version 3 for processing point cloud data.
    
    Attributes:
        channels (int): Number of channels in the input and output.
        num_heads (int): Number of attention heads.
        patch_size (int): Size of each patch.
        mlp_ratio (float): Ratio for determining the hidden layer size of the MLP.
        qkv_bias (bool): Whether to include bias in the QKV computation.
        qk_scale (Optional[float]): Scaling factor for QK computation. If None, scale is set dynamically.
        attn_drop (float): Dropout rate for attention weights.
        proj_drop (float): Dropout rate for output projection.
        drop_path (float): Dropout rate for paths.
        norm_layer (Type[nn.Module]): Normalization layer class.
        act_layer (Type[nn.Module]): Activation layer class.
        pre_norm (bool): Whether to use pre-normalization.
        order_index (int): Order index for serialized attention.
        cpe_indice_key (Optional[str]): Key for CPE indices.
        enable_rpe (bool): Whether to enable relative position encoding.
        enable_flash (bool): Whether to enable FLASH attention.
        upcast_attention (bool): Whether to upcast attention computation.
        upcast_softmax (bool): Whether to upcast softmax computation.
    """
    def __init__(
        self,
        channels: int,
        num_heads: int,
        patch_size: int = 48,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        pre_norm: bool = True,
        order_index: int = 0,
        cpe_indice_key: Optional[str] = None,
        enable_rpe: bool = False,
        enable_flash: bool = True,
        upcast_attention: bool = True,
        upcast_softmax: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm

        self.cpe = PointSequential(
            spconv.SubMConv3d(
                channels,
                channels,
                kernel_size=3,
                bias=True,
                indice_key=cpe_indice_key,
            ),
            nn.Linear(channels, channels),
            norm_layer(channels),
        )

        self.norm1 = PointSequential(norm_layer(channels))
        self.attn = SerializedAttention(
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
            )
        )
        self.drop_path = PointSequential(
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, point: Point) -> Point:
        """
        Forward pass of the Block_v3.
        
        Args:
            point (Point): Input point cloud data.
            
        Returns:
            Point: Processed point cloud data.
        """
        # Shortcut connection
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat

        if self.pre_norm:
            point = self.norm1(point)
        point = self.drop_path(self.attn(point))
        point.feat = shortcut + point.feat

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point = self.drop_path(self.mlp(point))
        point.feat = shortcut + point.feat

        point.sparse_conv_feat.replace_feature(point.feat)

        return point