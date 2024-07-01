from typing import Optional, Type, Callable
import spconv.pytorch as spconv
from .sequential import PointModule, PointSequential
from .point import Point  

class Embedding(PointModule):
    """
    A module for embedding point cloud data using a convolutional neural network.

    This module applies a 3D sparse convolution to the input point cloud data to produce an embedded representation.
    It supports optional normalization and activation layers after the convolution.

    Attributes:
        in_channels (int): The number of input channels in the point cloud data.
        embed_channels (int): The number of output channels after embedding.
        stem (PointSequential): A sequential container of modules through which points are processed.

    Parameters:
        in_channels (int): The number of input channels in the point cloud data.
        embed_channels (int): The number of output channels after embedding.
        norm_layer (Optional[Type[torch.nn.Module]]): The type of normalization layer to use. If None, no normalization is applied.
        act_layer (Optional[Callable[[], torch.nn.Module]]): A callable that returns an instance of the activation layer to use. If None, no activation is applied.
    """

    def __init__(
        self,
        in_channels: int,
        embed_channels: int,
        norm_layer: Optional[Type[torch.nn.Module]] = None,
        act_layer: Optional[Callable[[], torch.nn.Module]] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels

        # Initialize the sequential container with the convolution layer
        self.stem = PointSequential(
            conv=spconv.SubMConv3d(
                in_channels,
                embed_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            )
        )

        # Optionally add normalization and activation layers
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

    def forward(self, point: Point) -> Point:
        """
        Forward pass of the Embedding module.

        Processes the input point cloud data through the convolutional network to produce an embedded representation.

        Parameters:
            point (Point): The input point cloud data.

        Returns:
            Point: The embedded point cloud data.
        """
        point = self.stem(point)
        return point