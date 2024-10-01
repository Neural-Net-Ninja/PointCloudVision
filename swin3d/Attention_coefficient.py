import enum
from typing import Any, List

import torch
from torch import Tensor
from torch.autograd.function import Function
import Swin3D.sparse_dl.attn_cuda as attn_module


class PosEmb(enum.Enum):
    NONE = enum.auto()
    LINEAR = enum.auto()
    SEPARATE = enum.auto()


class TableDims(enum.Enum):
    WHCD = enum.auto()
    DWHC = enum.auto()
    D0 = enum.auto()


class IndexMode(enum.Enum):
    DIRECT = enum.auto()
    INDIRECT = enum.auto()


class PrecisionMode(enum.Enum):
    HALF_ALL = enum.auto()
    HALF_FORWARD = enum.auto()
    HALF_NONE = enum.auto()


class AttnBaseFunction(Function):
    """
    Base class for attention functions with utility methods for casting layout and type.
    """
    @staticmethod
    def auto_cast_layout(table: Tensor, table_dims: TableDims, reverse: bool = False) -> Tensor:
        """
        Cast layout to native supported one.

        :param table: Input tensor.
        :type table: Tensor
        :param table_dims: Dimensions of the table.
        :type table_dims: TableDims
        :param reverse: Whether to reverse the layout. Defaults to `False`.
        :type reverse: bool, optional
        :return: Tensor with cast layout.
        :rtype: Tensor
        """
        if table_dims == TableDims.WHCD and not reverse:
            table = torch.permute(table, [3, 0, 1, 2]).contiguous()
        elif table_dims == TableDims.WHCD and reverse:
            table = torch.permute(table, [1, 2, 3, 0]).contiguous()
        return table

    @staticmethod
    def auto_cast_type(src_tensor: Tensor, dst_tensor: Tensor) -> Tensor:
        """
        Cast type to native supported one.

        :param src_tensor: Source tensor.
        :type src_tensor: Tensor
        :param dst_tensor: Destination tensor.
        :type dst_tensor: Tensor
        :return: Tensor with cast type.
        :rtype: Tensor
        """
        if src_tensor.type() == dst_tensor.type():
            return src_tensor
        else:
            return src_tensor.type(dst_tensor.type())

    @staticmethod
    def cast_layout_and_type(tensor: Tensor, dst_tensor: Tensor, table_dims: TableDims, reverse: bool = False) -> Tensor:
        """
        Cast the layout and type.

        :param tensor: Input tensor.
        :type tensor: Tensor
        :param dst_tensor: Destination tensor.
        :type dst_tensor: Tensor
        :param table_dims: Dimensions of the table.
        :type table_dims: TableDims
        :param reverse: Whether to reverse the layout. Defaults to `False`.
        :type reverse: bool, optional
        :return: Tensor with cast layout and type.
        :rtype: Tensor
        """
        tensor = __class__.auto_cast_type(tensor, dst_tensor)
        tensor = __class__.auto_cast_layout(tensor, table_dims, reverse)
        return tensor

    @staticmethod
    def padding_out_grads(grads: List[Tensor], num_inputs: int) -> Tuple[Tensor, ...]:
        """
        Pad the gradients to match the number of inputs.

        :param grads: List of gradients.
        :type grads: List[Tensor]
        :param num_inputs: Number of inputs.
        :type num_inputs: int
        :return: Tuple of padded gradients.
        :rtype: Tuple[Tensor, ...]
        """
        padding_grads = [None] * (num_inputs - len(grads))
        return (*grads, *padding_grads)


class AttnCalCoffFunction(AttnBaseFunction):
    """
    Function for calculating attention coefficients.
    """
    @staticmethod
    def forward(ctx: Any, raw_query_feats: Tensor, raw_key_feats: Tensor, query_table: Tensor, key_table: Tensor,
                m2w_indices: Tensor, w_elems: Tensor, w2m_indices: Tensor, w2n_indices: Tensor, n2n_indices: Tensor,
                n_coords: Tensor, pe: PosEmb = PosEmb.SEPARATE, table_dim: TableDims = TableDims.WHCD) -> Tensor:
        """
        Forward pass for calculating attention coefficients.

        :param ctx: Context object.
        :type ctx: Any
        :param raw_query_feats: Raw query features.
        :type raw_query_feats: Tensor
        :param raw_key_feats: Raw key features.
        :type raw_key_feats: Tensor
        :param query_table: Query table.
        :type query_table: Tensor
        :param key_table: Key table.
        :type key_table: Tensor
        :param m2w_indices: Indices for m2w.
        :type m2w_indices: Tensor
        :param w_elems: Elements for w.
        :type w_elems: Tensor
        :param w2m_indices: Indices for w2m.
        :type w2m_indices: Tensor
        :param w2n_indices: Indices for w2n.
        :type w2n_indices: Tensor
        :param n2n_indices: Indices for n2n.
        :type n2n_indices: Tensor
        :param n_coords: Coordinates for n.
        :type n_coords: Tensor
        :param pe: Positional embedding. Defaults to `PosEmb.SEPARATE`.
        :type pe: PosEmb, optional
        :param table_dim: Table dimensions. Defaults to `TableDims.WHCD`.
        :type table_dim: TableDims, optional
        :return: Calculated attention coefficients.
        :rtype: Tensor
        """
        ctx.save_for_backward(raw_query_feats, raw_key_feats, query_table, key_table, m2w_indices, w_elems, w2m_indices,
                              w2n_indices, n2n_indices, n_coords)
        setattr(ctx, 'table_dim', table_dim)
        setattr(ctx, 'pe', pe)
        query_table = __class__.cast_layout_and_type(query_table, raw_query_feats, table_dim)
        key_table = __class__.cast_layout_and_type(key_table, raw_key_feats, table_dim)
        coff, = attn_module.self_attn_cal_coff_indir_forward(
            raw_query_feats, raw_key_feats, query_table, key_table, m2w_indices, w_elems,
            w2m_indices, w2n_indices, n2n_indices, n_coords, pe.value)
        return coff

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Tuple[Tensor, ...]:
        """
        Backward pass for calculating gradients.

        :param ctx: Context object.
        :type ctx: Any
        :param grad_outputs: Gradients from the forward pass.
        :type grad_outputs: Any
        :return: Gradients for the input tensors.
        :rtype: Tuple[Tensor, ...]
        """
        raw_query_feats, raw_key_feats, query_table, key_table, m2w_indices, w_elems, w2m_indices, w2n_indices, \
            n2n_indices, n_coords = ctx.saved_tensors
        table_dim: TableDims = getattr(ctx, 'table_dim')
        pe: PosEmb = getattr(ctx, "pe")
        coff_grads, = grad_outputs
        query_table_fwd = __class__.cast_layout_and_type(query_table, raw_query_feats, table_dim)
        key_table_fwd = __class__.cast_layout_and_type(key_table, raw_key_feats, table_dim)
        raw_query_grads, raw_key_grads, query_table_grads, key_table_grads = \
            attn_module.self_attn_cal_coff_indir_backward(
                coff_grads, raw_query_feats, raw_key_feats, query_table_fwd, key_table_fwd,
                m2w_indices, w_elems, w2m_indices, w2n_indices, n2n_indices, n_coords, pe.value
            )
        query_table_grads = __class__.cast_layout_and_type(query_table_grads, query_table, table_dim, True)
        key_table_grads = __class__.cast_layout_and_type(key_table_grads, key_table, table_dim, True)

        return raw_query_grads, raw_key_grads, query_table_grads, key_table_grads, None, None, None, None, None, None


class AttnApplyCoffFunction(AttnBaseFunction):
    """
    Function for applying attention coefficients.
    """
    @staticmethod
    def forward(ctx: Any, raw_value_feats: Tensor, coff_norm: Tensor, value_table: Tensor, m2w_indices: Tensor,
                w_elems: Tensor, w2m_indices: Tensor, w2n_indices: Tensor, n2n_indices: Tensor, n_coords: Tensor,
                pe: PosEmb = PosEmb.SEPARATE, table_dim: TableDims = TableDims.WHCD) -> Tensor:
        """
        Forward pass for applying attention coefficients.

        :param ctx: Context object.
        :type ctx: Any
        :param raw_value_feats: Raw value features.
        :type raw_value_feats: Tensor
        :param coff_norm: Normalized coefficients.
        :type coff_norm: Tensor
        :param value_table: Value table.
        :type value_table: Tensor
        :param m2w_indices: Indices for m2w.
        :type m2w_indices: Tensor
        :param w_elems: Elements for w.
        :type w_elems: Tensor
        :param w2m_indices: Indices for w2m.
        :type w2m_indices: Tensor
        :param w2n_indices: Indices for w2n.
        :type w2n_indices: Tensor
        :param n2n_indices: Indices for n2n.
        :type n2n_indices: Tensor
        :param n_coords: Coordinates for n.
        :type n_coords: Tensor
        :param pe: Positional embedding. Defaults to `PosEmb.SEPARATE`.
        :type pe: PosEmb, optional
        :param table_dim: Table dimensions. Defaults to `TableDims.WHCD`.
        :type table_dim: TableDims, optional
        :return: Updated value features.
        :rtype: Tensor
        """
        ctx.save_for_backward(raw_value_feats, coff_norm, value_table, m2w_indices, w_elems, w2m_indices, w2n_indices,
                              n2n_indices, n_coords)
        setattr(ctx, 'table_dim', table_dim)
        setattr(ctx, 'pe', pe)
        value_table = __class__.cast_layout_and_type(value_table, raw_value_feats, table_dim)
        updated_value_feats, = attn_module.self_attn_apply_coff_indir_forward(raw_value_feats,
                                                                              coff_norm, value_table, m2w_indices,
                                                                              w_elems, w2m_indices, w2n_indices,
                                                                              n2n_indices, n_coords, pe.value)
        return updated_value_feats

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Tuple[Tensor, ...]:
        """
        Backward pass for calculating gradients.

        :param ctx: Context object.
        :type ctx: Any
        :param grad_outputs: Gradients from the forward pass.
        :type grad_outputs: Any
        :return: Gradients for the input tensors.
        :rtype: Tuple[Tensor, ...]
        """
        raw_value_feats, coff_norm, value_table, m2w_indices, w_elems, w2m_indices, w2n_indices, n2n_indices, \
            n_coords = ctx.saved_tensors
        table_dim: TableDims = getattr(ctx, 'table_dim')
        pe: PosEmb = getattr(ctx, "pe")
        updated_value_grads, = grad_outputs
        value_table_fwd = __class__.cast_layout_and_type(value_table, raw_value_feats, table_dim)
        raw_value_grads, coff_norm_grads, value_table_grads = \
            attn_module.self_attn_apply_coff_indir_backward(
                updated_value_grads, raw_value_feats, coff_norm, value_table_fwd, m2w_indices,
                w_elems, w2m_indices, w2n_indices, n2n_indices, n_coords, pe.value
            )
        value_table_grads = __class__.cast_layout_and_type(value_table_grads, value_table, table_dim, True)
        return raw_value_grads, coff_norm_grads, value_table_grads, None, None, None, None, None, None


class SelfAttnAIOFunction(AttnBaseFunction):
    """
    Function of all-in-one self-attention.
    """
    @staticmethod
    def forward(ctx: Any, raw_query_feats: Tensor, raw_key_feats: Tensor, raw_value_feats: Tensor, query_table: Tensor,
                key_table: Tensor, value_table: Tensor, table_offsets: Tensor, indices: List[Tensor], pe: PosEmb,
                table_dim: TableDims, mode: IndexMode, precision: PrecisionMode) -> Tensor:
        """
        Forward pass for all-in-one self-attention.

        :param ctx: Context object.
        :type ctx: Any
        :param raw_query_feats: Raw query features.
        :type raw_query_feats: Tensor
        :param raw_key_feats: Raw key features.
        :type raw_key_feats: Tensor
        :param raw_value_feats: Raw value features.
        :type raw_value_feats: Tensor
        :param query_table: Query table.
        :type query_table: Tensor
        :param key_table: Key table.
        :type key_table: Tensor
        :param value_table: Value table.
        :type value_table: Tensor
        :param table_offsets: Table offsets.
        :type table_offsets: Tensor
        :param indices: List of indices.
        :type indices: List[Tensor]
        :param pe: Positional embedding.
        :type pe: PosEmb
        :param table_dim: Table dimensions.
        :type table_dim: TableDims
        :param mode: Index mode.
        :type mode: IndexMode
        :param precision: Precision mode.
        :type precision: PrecisionMode
        :return: Normalized attention features.
        :rtype: Tensor
        """
        assert table_dim == TableDims.D0
        setattr(ctx, 'table_dim', table_dim)
        setattr(ctx, 'pe', pe)
        setattr(ctx, 'mode', mode)
        setattr(ctx, 'precision', precision)

        qkv_feats = [raw_query_feats, raw_key_feats, raw_value_feats]
        qkv_tables = [query_table, key_table, value_table]

        if torch.is_autocast_enabled() and precision != PrecisionMode.HALF_NONE:
            c_dtype = torch.get_autocast_gpu_dtype()
        else:
            c_dtype = torch.float32

        c_qkv_feats = [_f.type(c_dtype) for _f in qkv_feats]
        c_qkv_tables = [_t.type(c_dtype) for _t in qkv_tables]

        coff_rmax = attn_module.cal_max_coffs(c_qkv_feats, c_qkv_tables, table_offsets, indices, mode.value, pe.value)
        raw_attn_feats, sum_coffs = attn_module.self_attn_forward(c_qkv_feats, c_qkv_tables, coff_rmax, table_offsets,
                                                                  indices, mode.value, pe.value)
        norm_attn_feats = raw_attn_feats / sum_coffs

        bp_feats = [sum_coffs, coff_rmax, norm_attn_feats]
        backward_tensors = [*bp_feats, *qkv_feats, *qkv_tables, table_offsets, *indices]
        ctx.save_for_backward(*backward_tensors)
        return norm_attn_feats

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Tuple[Tensor, ...]:
        """
        Backward pass for calculating gradients.

        :param ctx: Context object.
        :type ctx: Any
        :param grad_outputs: Gradients from the forward pass.
        :type grad_outputs: Any
        :return: Gradients for the input tensors.
        :rtype: Tuple[Tensor, ...]
        """
        norm_attn_grads, = grad_outputs
        sum_coffs, coff_rmax, norm_attn_feats = ctx.saved_tensors[:3]
        raw_query_feats, raw_key_feats, raw_value_feats = ctx.saved_tensors[3:6]
        query_table, key_table, value_table = ctx.saved_tensors[6:9]
        table_offsets = ctx.saved_tensors[9]
        indices = ctx.saved_tensors[10:]

        pos_emb: PosEmb = getattr(ctx, "pe")
        mode: IndexMode = getattr(ctx, 'mode')
        precision: PrecisionMode = getattr(ctx, 'precision')

        qkv_feats = [raw_query_feats, raw_key_feats, raw_value_feats]
        qkv_tables = [query_table, key_table, value_table]

        if precision == PrecisionMode.HALF_ALL:
            c_dtype = norm_attn_grads.dtype
        else:
            c_dtype = torch.float32
            r_dtype = norm_attn_grads.dtype
            norm_attn_feats = norm_attn_feats.type(torch.float32)
            sum_coffs = sum_coffs.type(torch.float32)
            norm_attn_grads = norm_attn_grads.type(torch.float32)

        c_qkv_feats = [_f.type(c_dtype) for _f in qkv_feats]
        c_qkv_tables = [_t.type(c_dtype) for _t in qkv_tables]

        raw_attn_feats = norm_attn_feats * sum_coffs
        exp_sum_grads = attn_module.cal_exp_sum_grads(norm_attn_grads, raw_attn_feats, sum_coffs, c_qkv_feats[-1],
                                                      value_table)
        grads = attn_module.self_attn_indir_backward(norm_attn_grads, exp_sum_grads, raw_attn_feats, sum_coffs,
                                                     coff_rmax, *c_qkv_feats, *c_qkv_tables, table_offsets, indices,
                                                     mode.value, pos_emb.value)

        for t_grad, t_in in zip(grads[3:6], qkv_tables):
            t_grad = t_grad.type(t_in.type())

        if precision == PrecisionMode.HALF_FORWARD:
            for t_grad in grads[:3]:
                t_grad = t_grad.type(r_dtype)
        return __class__.padding_out_grads(grads, 12)

class SelfAttnAIOModule(torch.nn.Module):
    """
    Self-Attention All-In-One Module for 3D point cloud processing.

    :param pe: Positional embedding.
    :type pe: PosEmb
    :param table_dim: Dimensions of the table.
    :type table_dim: TableDims
    :param mode: Index mode.
    :type mode: IndexMode
    """
    def __init__(self, pe: PosEmb, table_dim: TableDims, mode: IndexMode) -> None:
        super().__init__()
        self.pe = pe
        self.table_dim = table_dim
        self.mode = mode

    def forward(self, 
                raw_query_feats: torch.Tensor, 
                raw_key_feats: torch.Tensor, 
                raw_value_feats: torch.Tensor, 
                query_table: torch.Tensor, 
                key_table: torch.Tensor, 
                value_table: torch.Tensor, 
                indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Self-Attention All-In-One Module.

        :param raw_query_feats: Raw query features.
        :type raw_query_feats: torch.Tensor
        :param raw_key_feats: Raw key features.
        :type raw_key_feats: torch.Tensor
        :param raw_value_feats: Raw value features.
        :type raw_value_feats: torch.Tensor
        :param query_table: Query table.
        :type query_table: torch.Tensor
        :param key_table: Key table.
        :type key_table: torch.Tensor
        :param value_table: Value table.
        :type value_table: torch.Tensor
        :param indices: Indices for the attention mechanism.
        :type indices: torch.Tensor
        :return: Normalized attention features.
        :rtype: torch.Tensor
        """
        norm_attn_feats = SelfAttnAIOFunction.apply(
            raw_query_feats, raw_key_feats, raw_value_feats, query_table, key_table, 
            value_table, indices, self.pe, self.table_dim, self.mode)
        return norm_attn_feats
