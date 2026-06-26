"""
Decoupled inference framework for Transolver-3.

Implements the two-stage inference pipeline from:
  "Transolver-3: Scaling Up Transformer Solvers to Industrial-Scale Geometries"

  Stage 1 - Physical state caching (PhysicalStateCachingModel):
    Processes the full mesh in memory-compatible chunks, layer-by-layer,
    to build the physical state cache s_cache = {s_out'(l)}_{l=1..L}.
    This captures the global physics structure of the simulation without
    loading the entire mesh into GPU memory at once.

  Stage 2 - Full mesh decoding (FullMeshDecodingModel):
    Uses the precomputed physical state cache to run inference on any
    mesh coordinate without recomputing global attention from scratch.

Both models share the same parameter structure as
Transolver_chunk_opt_matrix_mul.Model, so the same trained checkpoint
can be loaded into either via load_state_dict().
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from models.Transolver_chunk_opt_matrix_mul import Transolver_block, MLP, _as_feature_tensor

trunc_normal_ = nn.init.trunc_normal_


class _CachingBlock(Transolver_block):
    """
    Transolver block used during physical state caching (Stage 1).

    When physical_state is None, computes global physics-state statistics
    from the current chunk batch and returns them as unnormalized accumulators
    (num, den) so the caller can aggregate across batches before normalizing.
    When physical_state is provided (layers already cached), uses it directly.
    """
    def forward_chunks(self, fx_list, physical_state=None, eps=1e-5, use_checkpoint=True):
        if physical_state is None:
            global_num, global_den = None, None
            for fxk in fx_list:
                uk = self.ln_1(fxk)
                if use_checkpoint:
                    num_k, den_k = checkpoint(
                        self.Attn.chunk_stats, uk,
                        preserve_rng_state=True, use_reentrant=False
                    )
                else:
                    num_k, den_k = self.Attn.chunk_stats(uk)
                global_num = num_k if global_num is None else (global_num + num_k)
                global_den = den_k if global_den is None else (global_den + den_k)
            # Return unnormalized accumulators so the caller can aggregate across batches
            batch_slice_token = global_num
            slice_norm = global_den
            slice_token = global_num / (global_den[..., None] + eps)
        else:
            slice_token = physical_state
            batch_slice_token = None
            slice_norm = None

        out_slice = self.Attn.slice_attend(slice_token)

        out_list = []
        for fxk in fx_list:
            def chunk_compute(f_k, o_slice):
                uk = self.ln_1(f_k)
                a_out = self.Attn.chunk_deslice_to_out(uk, o_slice)
                res_fx = a_out + f_k
                mlp_out = self.mlp(self.ln_2(res_fx)) + res_fx
                if self.last_layer:
                    mlp_out = self.mlp2(self.ln_3(mlp_out))
                return mlp_out
            if use_checkpoint:
                fxk2 = checkpoint(chunk_compute, fxk, out_slice, use_reentrant=False)
            else:
                fxk2 = chunk_compute(fxk, out_slice)
            out_list.append(fxk2)

        return out_list, batch_slice_token, slice_norm


class _DecodingBlock(Transolver_block):
    """
    Transolver block used during full mesh decoding (Stage 2).

    Always uses physical states from the precomputed physical state cache.
    Never recomputes global attention statistics.
    """
    def forward_chunks(self, fx_list, physical_state, eps=1e-5, use_checkpoint=True):
        out_slice = self.Attn.slice_attend(physical_state)

        out_list = []
        for fxk in fx_list:
            def chunk_compute(f_k, o_slice):
                uk = self.ln_1(f_k)
                a_out = self.Attn.chunk_deslice_to_out(uk, o_slice)
                res_fx = a_out + f_k
                mlp_out = self.mlp(self.ln_2(res_fx)) + res_fx
                if self.last_layer:
                    mlp_out = self.mlp2(self.ln_3(mlp_out))
                return mlp_out
            if use_checkpoint:
                fxk2 = checkpoint(chunk_compute, fxk, out_slice, use_reentrant=False)
            else:
                fxk2 = chunk_compute(fxk, out_slice)
            out_list.append(fxk2)

        return out_list


class PhysicalStateCachingModel(nn.Module):
    """
    Stage 1 of the decoupled inference framework: physical state caching.

    Iterates the full mesh in memory-compatible chunks, one layer at a time,
    to build the physical state cache s_cache = {s_out'(l)}_{l=1..L}.
    The caller accumulates per-chunk unnormalized statistics across batches,
    then normalizes to obtain each layer's physical state before proceeding
    to the next layer.
    """
    def __init__(self,
                 space_dim=1,
                 n_layers=5,
                 n_hidden=256,
                 dropout=0,
                 n_head=8,
                 act='gelu',
                 mlp_ratio=1,
                 fun_dim=1,
                 out_dim=1,
                 slice_num=32,
                 ref=8,
                 unified_pos=False):
        super().__init__()
        self.__name__ = 'UniPDE_3D'
        self._n_hidden = n_hidden
        self.ref = ref
        self.unified_pos = unified_pos

        if unified_pos:
            self.preprocess = MLP(fun_dim + ref**3, n_hidden * 2, n_hidden,
                                  n_layers=0, res=False, act=act)
        else:
            self.preprocess = MLP(fun_dim + space_dim, n_hidden * 2, n_hidden,
                                  n_layers=0, res=False, act=act)

        self.blocks = nn.ModuleList([
            _CachingBlock(num_heads=n_head, hidden_dim=n_hidden, dropout=dropout,
                          act=act, mlp_ratio=mlp_ratio, out_dim=out_dim,
                          slice_num=slice_num, last_layer=(i == n_layers - 1))
            for i in range(n_layers)
        ])

        self.apply(self._init_weights)
        self.placeholder = nn.Parameter(
            (1 / n_hidden) * torch.rand(n_hidden, dtype=torch.float)
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, data, state_cache, layer, use_checkpoint=True):
        """
        Process one chunk batch up to `layer` and return unnormalized
        physical-state accumulators at that layer.

        state_cache: list of normalized physical states already computed
            for layers 0..(layer-1).
        layer: index of the layer at which to extract new statistics.

        Returns: (fx_list, slice_token_wo_norm, slice_norm)
            slice_token_wo_norm and slice_norm are unnormalized accumulators
            to be summed across chunk batches before normalization.
        """
        fx_list = []
        for chunk in data:
            xk = _as_feature_tensor(chunk)
            fxk = self.preprocess(xk)
            fxk = fxk + self.placeholder[None, None, :]
            fx_list.append(fxk)

        for i, block in enumerate(self.blocks):
            if i < layer:
                fx_list, _, _ = block.forward_chunks(
                    fx_list, state_cache[i],
                    use_checkpoint=use_checkpoint
                )
            elif i == layer:
                fx_list, slice_token_wo_norm, slice_norm = block.forward_chunks(
                    fx_list, None, use_checkpoint=use_checkpoint
                )
                break

        return fx_list, slice_token_wo_norm, slice_norm


class FullMeshDecodingModel(nn.Module):
    """
    Stage 2 of the decoupled inference framework: full mesh decoding.

    Runs inference at arbitrary mesh coordinates using the precomputed
    physical state cache from Stage 1. The same trained checkpoint as
    Transolver_chunk_opt_matrix_mul.Model can be loaded into this class.
    """
    def __init__(self,
                 space_dim=1,
                 n_layers=5,
                 n_hidden=256,
                 dropout=0,
                 n_head=8,
                 act='gelu',
                 mlp_ratio=1,
                 fun_dim=1,
                 out_dim=1,
                 slice_num=32,
                 ref=8,
                 unified_pos=False):
        super().__init__()
        self.__name__ = 'UniPDE_3D'
        self._n_hidden = n_hidden
        self.ref = ref
        self.unified_pos = unified_pos

        if unified_pos:
            self.preprocess = MLP(fun_dim + ref**3, n_hidden * 2, n_hidden,
                                  n_layers=0, res=False, act=act)
        else:
            self.preprocess = MLP(fun_dim + space_dim, n_hidden * 2, n_hidden,
                                  n_layers=0, res=False, act=act)

        self.blocks = nn.ModuleList([
            _DecodingBlock(num_heads=n_head, hidden_dim=n_hidden, dropout=dropout,
                           act=act, mlp_ratio=mlp_ratio, out_dim=out_dim,
                           slice_num=slice_num, last_layer=(i == n_layers - 1))
            for i in range(n_layers)
        ])

        self.apply(self._init_weights)
        self.placeholder = nn.Parameter(
            (1 / n_hidden) * torch.rand(n_hidden, dtype=torch.float)
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, data, state_cache, use_checkpoint=True):
        """
        Run full inference using the precomputed physical state cache.

        state_cache: list of normalized physical states, one per layer,
            built by PhysicalStateCachingModel.

        Returns: list of per-chunk output tensors.
        """
        fx_list = []
        for chunk in data:
            xk = _as_feature_tensor(chunk)
            fxk = self.preprocess(xk)
            fxk = fxk + self.placeholder[None, None, :]
            fx_list.append(fxk)

        for i, block in enumerate(self.blocks):
            fx_list = block.forward_chunks(
                fx_list, state_cache[i], use_checkpoint=use_checkpoint
            )

        return fx_list
