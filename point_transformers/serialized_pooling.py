import math
import torch
import torch.nn as nn
import torch_scatter
from typing import Dict, Optional, Type, Callable
from ..models.point_transformers.sequential import PointModule, PointSequential
from ..models.point_transformers.point import Point  # Assuming there's a Point class defined somewhere

class SerializedPooling(PointModule):
    """
    A module for serialized pooling of point cloud data.

    Attributes:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int): Pooling stride.
        reduce (str): Reduction method ('sum', 'mean', 'min', 'max').
        shuffle_orders (bool): Whether to shuffle orders.
        traceable (bool): If true, records parent and cluster information for tracing.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        norm_layer: Optional[Type[nn.Module]] = None,
        act_layer: Optional[Callable[[], nn.Module]] = None,
        reduce: str = "max",
        shuffle_orders: bool = True,
        traceable: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if stride != 2 ** (math.ceil(math.log2(stride))):
            raise ValueError("Stride must be a power of 2.")
        self.stride = stride

        if reduce not in ["sum", "mean", "min", "max"]:
            raise ValueError("Reduce must be one of 'sum', 'mean', 'min', 'max'.")
        self.reduce = reduce

        self.shuffle_orders = shuffle_orders
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)
        self.norm = PointSequential(norm_layer(out_channels)) if norm_layer is not None else None
        self.act = PointSequential(act_layer()) if act_layer is not None else None

    def forward(self, point: Point) -> Point:
        """
        Forward pass of the SerializedPooling module.

        Parameters:
            point (Point): Input point cloud data.

        Returns:
            Point: Pooled point cloud data.
        """
        pooling_depth = (math.ceil(math.log2(self.stride)) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0

        code = point.serialized_code >> (pooling_depth * 3)
        code_, cluster, counts = torch.unique(
            code[0], sorted=True, return_inverse=True, return_counts=True
        )

        _, indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        head_indices = indices[idx_ptr[:-1]]

        code = code[:, head_indices]
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1, index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(code.shape[0], 1)
        )

        if self.shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code, order, inverse = code[perm], order[perm], inverse[perm]

        point_dict = {
            "feat": torch_scatter.segment_csr(self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce),
            "coord": torch_scatter.segment_csr(point.coord[indices], idx_ptr, reduce="mean"),
            "grid_coord": point.grid_coord[head_indices] >> pooling_depth,
            "serialized_code": code,
            "serialized_order": order,
            "serialized_inverse": inverse,
            "serialized_depth": point.serialized_depth - pooling_depth,
            "batch": point.batch[head_indices],
        }

        for key in ["condition", "context"]:
            if key in point.keys():
                point_dict[key] = point[key]

        if self.traceable:
            point_dict.update({"pooling_inverse": cluster, "pooling_parent": point})

        point = Point(point_dict)
        if self.norm is not None:
            point = self.norm(point)
        if self.act is not None:
            point = self.act(point)
        point.sparsify()
        return point