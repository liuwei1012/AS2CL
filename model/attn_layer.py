import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt
import os


class Attention(nn.Module):
    def __init__(self, window_size, mask_flag=False, scale=None, dropout=0.0):
        super(Attention, self).__init__()
        self.window_size = window_size
        self.mask_flag = mask_flag
        self.scale = scale
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, queries, keys, values, attn_mask=None):
        '''
        queries : N x L x Head x d
        keys : N x L(s) x Head x d
        values : N x L x Head x d
        '''
        N, L, Head, C = queries.shape

        scale = self.scale if self.scale is not None else 1. / sqrt(C)

        attn_scores = torch.einsum('nlhd,nshd->nhls', queries, keys)    # N x Head x L x L
        attn_weights = self.dropout(torch.softmax(scale * attn_scores, dim=-1))

        updated_values = torch.einsum('nhls,nshd->nlhd', attn_weights, values)  # N x L x Head x d

        return updated_values.contiguous()
    

class AttentionLayer(nn.Module):
    def __init__(self, window_size, d_model, n_heads, d_keys=None, d_values=None, mask_flag=False, 
                 scale=None, dropout=0.0):
        super(AttentionLayer, self).__init__()

        self.d_keys = d_keys if d_keys is not None else (d_model // n_heads)
        self.d_values = d_values if d_values is not None else (d_model // n_heads)
        self.n_heads = n_heads
        self.d_model = d_model  # d_model = C

        # Linear projections to Q, K, V
        self.W_Q = nn.Linear(self.d_model, self.n_heads * self.d_keys)
        self.W_K = nn.Linear(self.d_model, self.n_heads * self.d_keys)
        self.W_V = nn.Linear(self.d_model, self.n_heads * self.d_values)

        self.out_proj = nn.Linear(self.n_heads * self.d_values, self.d_model)

        self.attn = Attention(window_size=window_size, mask_flag=mask_flag, scale=scale, dropout=dropout)

    def forward(self, input):
        '''
        input : N x L x C(=d_model)
        '''
        N, L, _ = input.shape

        Q = self.W_Q(input).contiguous().view(N, L, self.n_heads, -1)
        K = self.W_K(input).contiguous().view(N, L, self.n_heads, -1)
        V = self.W_V(input).contiguous().view(N, L, self.n_heads, -1)

        updated_V = self.attn(Q, K, V)  # N x L x Head x d_values
        out = updated_V.view(N, L, -1)

        return self.out_proj(out)   # N x L x C(=d_model)


class DependencyPatternModule(nn.Module):
    """
    动态变量依赖模式矩阵 (论文 4.2.4 节, 公式 4.16 / 4.17)

    对每个时间步 t, 利用变量嵌入 E_t ∈ R^{D × d_emb} 通过共享的 Q/K 投影
    计算 D×D 的注意力矩阵 A_t, 描述变量间的瞬时依赖强度。

    输入:  E  [B, T, D, d_emb]   变量独立解耦嵌入 (来自 VariableIndependentEmbedding)
    输出:  A  [B, T, D, D]       动态依赖模式序列
               A[b, t, i, j] 表示时刻 t 变量 j 对变量 i 的依赖强度
    """

    def __init__(self, d_emb: int, dropout: float = 0.0):
        """
        Args:
            d_emb:   变量嵌入维度 (与 VariableIndependentEmbedding 的 d_emb 一致)
            dropout: 注意力 Dropout 概率
        """
        super(DependencyPatternModule, self).__init__()
        self.d_emb = d_emb
        self.scale = 1.0 / math.sqrt(d_emb)

        # 共享的可学习投影矩阵 W^Q, W^K (公式 4.16)
        # 在所有时间步 t 上共享
        self.W_Q = nn.Linear(d_emb, d_emb, bias=False)
        self.W_K = nn.Linear(d_emb, d_emb, bias=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, E: torch.Tensor) -> torch.Tensor:
        """
        E: [B, T, D, d_emb]
        返回 A: [B, T, D, D]
        """
        # ── 公式 4.16: Q_t = E_t W^Q,  K_t = E_t W^K ──────────────────────
        # W_Q / W_K 作用在最后一维 d_emb 上, 对 B 和 T 维度自动广播
        Q = self.W_Q(E)   # [B, T, D, d_emb]
        K = self.W_K(E)   # [B, T, D, d_emb]

        # ── 公式 4.17: A_t = Softmax(Q_t K_t^T / sqrt(d_emb)) ─────────────
        # Q @ K^T: [B, T, D, d_emb] × [B, T, d_emb, D] → [B, T, D, D]
        scores = torch.matmul(Q, K.transpose(-1, -2)) * self.scale  # [B, T, D, D]

        # Softmax 沿最后一维 (key 维度), 使每行和为 1
        A = self.dropout(torch.softmax(scores, dim=-1))              # [B, T, D, D]
        return A


if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from model.embedding import VariableIndependentEmbedding

    B, T, D, d_emb = 4, 100, 12, 64
    x = torch.randn(B, T, D)

    # Step 1: 变量独立嵌入
    emb_model = VariableIndependentEmbedding(n_vars=D, d_emb=d_emb)
    E = emb_model(x)                          # [4, 100, 12, 64]
    print(f"[Embedding]  E shape : {E.shape}")

    # Step 2: 动态依赖模式矩阵
    dep_model = DependencyPatternModule(d_emb=d_emb)
    A = dep_model(E)                          # [4, 100, 12, 12]
    print(f"[DepPattern] A shape : {A.shape}")

    # 验证: A 在最后一维 (key 维) 上的和为 1 (Softmax 性质)
    row_sums = A.sum(dim=-1)                  # [4, 100, 12]
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), \
        f"Softmax 行和不为 1! max_err={( row_sums - 1).abs().max().item():.2e}"
    print(f"[DepPattern] Softmax row-sum check passed: all sums ~= 1.0")

