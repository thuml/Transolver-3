import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange

ACTIVATION = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
}
trunc_normal_ = nn.init.trunc_normal_


def _as_feature_tensor(chunk):
    if isinstance(chunk, (tuple, list)):
        return chunk[0]
    return chunk


# -----------------------
# Physics Attention
# -----------------------
class Physics_Attention_Irregular_Mesh(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64):
        super().__init__()

        self.heads = heads
        self.dim_head = dim_head
        self.slice_num = slice_num

        self.in_project = nn.Linear(dim, 2 * heads * dim_head)  # produces fx and x
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        torch.nn.init.orthogonal_(self.in_project_slice.weight)
        self.to_out_linear = nn.Linear(heads * dim_head, dim)

        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)

    def forward(self, x):
        B, N, C = x.shape
        H, D, G = self.heads, self.dim_head, self.slice_num

        # --- Stage 1: Slice - generate weights and aggregate into slice tokens ---
        f_w, f_b = self._get_fused_weight_slice()
        slice_weights = self.chunk_weights(x, f_w, f_b)  # (B, H, N, G)

        slice_norm = slice_weights.sum(dim=2, keepdim=True) + 1e-5  # (B, H, 1, G)

        # Aggregate raw features into slice domain: (B, N, C) x (B, H, N, G) -> (B, H, G, C)
        raw_states = torch.einsum("bnc, bhng -> bhgc", x, slice_weights)
        raw_states = raw_states / slice_norm.transpose(-1, -2)

        # Project in slice domain O(G*C*D) instead of O(N*C*D)
        w_fx = self.in_project.weight[:H*D].view(H, D, C)
        b_fx = self.in_project.bias[:H*D].view(H, D)
        slice_token = torch.einsum("bhgc, hdc -> bhgd", raw_states, w_fx) + b_fx.view(1, H, 1, D)

        # --- Stage 2: Attention over slice tokens (complexity depends on G, not N) ---
        out_slice_token = self.slice_attend(slice_token)  # (B, H, G, D)

        # --- Stage 3: Deslice back to point space ---
        return self.chunk_deslice_to_out(x, out_slice_token, slice_weights)

    def chunk_stats(self, x: torch.Tensor):
        B, N, C = x.shape
        H, D, G = self.heads, self.dim_head, self.slice_num

        # 1. Get fused weights for slice assignment
        f_w, f_b = self._get_fused_weight_slice()
        slice_weights = self.chunk_weights(x, f_w, f_b)  # (B, H, N, G)

        # 2. Compute density in slice domain: O(NG) -> O(G)
        den = slice_weights.sum(dim=2)  # (B, H, G)

        # 3. Aggregate raw features into slice domain first
        # x: (B, N, C), slice_weights: (B, H, N, G) -> x_agg: (B, H, G, C)
        x_agg = torch.einsum("bnc, bhng -> bhgc", x, slice_weights)

        # 4. Project in slice domain (G tokens instead of N)
        w_fx = self.in_project.weight[:H*D].view(H, D, C)
        b_fx = self.in_project.bias[:H*D].view(H, D)

        # (B, H, G, C) @ (H, C, D) -> (B, H, G, D)
        num = torch.einsum("bhgc, hdc -> bhgd", x_agg, w_fx)

        # Account for the bias term accumulated over all N points
        num = num + den.unsqueeze(-1) * b_fx.unsqueeze(1)

        return num, den

    def slice_attend(self, slice_token: torch.Tensor):
        """slice_token: (B,H,G,D) -> out_slice_token: (B,H,G,D)"""
        q = self.to_q(slice_token)
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)
        out_slice_token = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False
        )
        return out_slice_token

    def _get_fused_weight_slice(self):
        """Fuse in_project (x_mid part) with in_project_slice to project C -> G in one matmul."""
        # Extract the x_mid portion of in_project: (H*D, C) -> (H, D, C)
        w_in = self.in_project.weight[self.heads * self.dim_head:]
        w_in = w_in.view(self.heads, self.dim_head, -1)

        w_slice = self.in_project_slice.weight  # (G, D)

        # Fuse: (G, D) @ (H, D, C) -> (H, G, C)
        fused_w = torch.matmul(w_slice, w_in)

        # Fuse bias
        b_in = self.in_project.bias[self.heads * self.dim_head:].view(self.heads, self.dim_head)
        fused_b = torch.matmul(w_slice, b_in.unsqueeze(-1)).squeeze(-1)  # (H, G)
        fused_b = fused_b + self.in_project_slice.bias

        return fused_w, fused_b

    def chunk_weights(self, x: torch.Tensor, fused_w=None, fused_b=None):
        """Compute per-point slice assignment weights via a single C -> G matmul."""
        if fused_w is None:
            fused_w, fused_b = self._get_fused_weight_slice()

        # x: (B, N, C), fused_w: (H, G, C) -> logits: (B, H, N, G)
        logits = torch.einsum("bnc, hgc -> bhng", x, fused_w) + fused_b.view(1, self.heads, 1, self.slice_num)
        return F.softmax(logits / self.temperature, dim=-1)

    def chunk_deslice_to_out(self, x: torch.Tensor, out_slice_token: torch.Tensor, slice_weights=None):
        """
        Deslice back to point space using associativity of linear operators:
        instead of deslice then project (O(N*HD*C)),
        project in slice domain first then deslice (O(G*HD*C) + O(N*G*C)).
        """
        B, N, C = x.shape
        H, G, D = self.heads, self.slice_num, self.dim_head

        if slice_weights is None:
            slice_weights = self.chunk_weights(x)  # (B, H, N, G)

        w_out = self.to_out_linear.weight.view(-1, H, D).permute(1, 2, 0)  # (H, D, C)

        # Project in slice domain: (B, H, G, D) @ (H, D, C) -> (B, H, G, C)
        projected_slices = torch.einsum("bhgd, hdc -> bhgc", out_slice_token, w_out)

        # Aggregate back to point space, summing over heads:
        # (B, H, N, G) @ (B, H, G, C) -> (B, N, C)
        out_x = torch.einsum("bhng, bhgc -> bnc", slice_weights, projected_slices)

        return self.dropout(out_x + self.to_out_linear.bias)


# -----------------------
# MLP
# -----------------------
class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super().__init__()
        if act in ACTIVATION:
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            x = self.linears[i](x) + x if self.res else self.linears[i](x)
        x = self.linear_post(x)
        return x


# -----------------------
# Transolver block
# -----------------------
class Transolver_block(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act='gelu',
        mlp_ratio=4,
        last_layer=False,
        out_dim=1,
        slice_num=32,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Physics_Attention_Irregular_Mesh(
            hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
            dropout=dropout, slice_num=slice_num
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx

    def forward_chunks(self, fx_list, eps=1e-5, use_checkpoint=True):
        """
        fx_list: list of tensors [(B,n1,C), (B,n2,C), ...]
        Returns list with same chunking, updated for the next layer.
        """
        # Pass 1: accumulate global num/den from all chunks
        global_num, global_den = None, None
        for fxk in fx_list:
            uk = self.ln_1(fxk)
            if use_checkpoint:
                num_k, den_k = checkpoint(self.Attn.chunk_stats, uk, preserve_rng_state=True, use_reentrant=False)
            else:
                num_k, den_k = self.Attn.chunk_stats(uk)

            global_num = num_k if global_num is None else (global_num + num_k)
            global_den = den_k if global_den is None else (global_den + den_k)

        slice_token = global_num / (global_den[..., None] + eps)  # (B,H,G,D)

        # Global slice attention (once for all chunks)
        out_slice = self.Attn.slice_attend(slice_token)  # (B,H,G,D)

        # Pass 2: per-chunk deslice + residual + MLP
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


# -----------------------
# Model
# -----------------------
class Model(nn.Module):
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
        self.ref = ref
        self.unified_pos = unified_pos

        if self.unified_pos:
            self.preprocess = MLP(fun_dim + self.ref**3, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)
        else:
            self.preprocess = MLP(fun_dim + space_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)

        self.blocks = nn.ModuleList([
            Transolver_block(
                num_heads=n_head, hidden_dim=n_hidden, dropout=dropout,
                act=act, mlp_ratio=mlp_ratio, out_dim=out_dim,
                slice_num=slice_num, last_layer=(i == n_layers - 1)
            ) for i in range(n_layers)
        ])

        self.initialize_weights()
        self.placeholder = nn.Parameter((1 / n_hidden) * torch.rand(n_hidden, dtype=torch.float))

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, data, use_checkpoint=True, input_list=True):
        """
        data: list of feature tensors when input_list=True, or a single feature tensor otherwise.
        Returns a list of per-chunk outputs when input_list=True.
        """
        eps = 1e-5
        if input_list:
            fx_list = []
            for chunk in data:
                xk = _as_feature_tensor(chunk)
                fxk = self.preprocess(xk)
                fxk = fxk + self.placeholder[None, None, :]
                fx_list.append(fxk)

            for block in self.blocks:
                fx_list = block.forward_chunks(fx_list, eps=eps, use_checkpoint=use_checkpoint)
            return fx_list
        else:
            x = _as_feature_tensor(data)
            fx = self.preprocess(x)
            fx = fx + self.placeholder[None, None, :]
            for block in self.blocks:
                fx = block(fx)
            return fx

