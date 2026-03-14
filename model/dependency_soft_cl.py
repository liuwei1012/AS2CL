"""
层次化依赖模式一致性软对比损失 (论文 4.2.4 - 4.2.7 节)

核心创新:
  - 基于 JS 散度的依赖模式一致性权重 (公式 4.18-4.20)
  - 层次化时间粒度的动态依赖图 (公式 4.21-4.23)
  - 综合时间邻近性与依赖一致性的软权重 (公式 4.24-4.26)
  - 软对比损失 (公式 4.27-4.29)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    KL(p || q) = Σ p * log(p / q)

    Args:
        p, q: [..., D]  概率分布 (最后一维为分布维度)
        eps:  数值稳定常数

    Returns:
        kl: [...]  KL 散度
    """
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    return (p * (p / q).log()).sum(dim=-1)


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    JS(p || q) = 0.5 * KL(p || M) + 0.5 * KL(q || M),  M = 0.5 * (p + q)

    公式 4.18 / 4.19 的核心计算

    Args:
        p, q: [..., D]  概率分布

    Returns:
        js: [...]  JS 散度 ∈ [0, log2]
    """
    M = 0.5 * (p + q)                                    # 公式 4.18: 混合分布
    return 0.5 * kl_divergence(p, M, eps) + 0.5 * kl_divergence(q, M, eps)


def dep_consistency_weight(
    A: torch.Tensor,
    sigma: float = 1.0,
    tau_w: float = 1.0,
    eps: float = 1e-8,
    use_gaussian: bool = True,
) -> torch.Tensor:
    """
    计算时间窗口内所有时间步对之间的依赖一致性权重矩阵 W_DC

    公式 4.19: D_dep(t,t') = (1/D) Σ_i JS(A_t^{i,:} || A_{t'}^{i,:})
    公式 4.20: W_DC(t,t') = exp(-D_dep^2 / (2σ^2))          [use_gaussian=True]
    公式 4.23: W_DC^(l)(u,v) = exp(-JS(A_u^(l) || A_v^(l)) / τ_w) [use_gaussian=False]

    Args:
        A:            [B, T, D, D]  依赖模式矩阵序列 (每行已经过 Softmax)
        sigma:        高斯核带宽 σ (公式 4.20 使用)
        tau_w:        温度系数 τ_w (公式 4.23 使用)
        eps:          数值稳定常数
        use_gaussian: True → 公式 4.20 (高斯核); False → 公式 4.23 (指数衰减)

    Returns:
        W_DC: [B, T, T]  依赖一致性权重矩阵
    """
    B, T, D, _ = A.shape

    # A_i: [B, T, D, D] → 展开为 [B, T, D*D] 方便广播
    # 对每对 (t, t') 计算 JS 散度
    # 利用广播: A[:, :, None, :, :] vs A[:, None, :, :, :]
    A_t  = A.unsqueeze(2)   # [B, T, 1, D, D]
    A_tp = A.unsqueeze(1)   # [B, 1, T, D, D]

    # JS 散度作用在最后一维 (key 维度, 即每行的概率分布)
    # js_per_var: [B, T, T, D]
    js_per_var = js_divergence(A_t, A_tp, eps)   # [..., D, D] → [..., D]

    # D_dep(t,t') = (1/D) Σ_i JS(A_t^{i,:} || A_{t'}^{i,:})  (公式 4.19)
    # js_per_var: [B, T, T, D]  → 对变量维度取均值
    D_dep = js_per_var.mean(dim=-1)              # [B, T, T]

    if use_gaussian:
        # 公式 4.20: W_DC = exp(-D_dep^2 / (2σ^2))
        W_DC = torch.exp(-D_dep.pow(2) / (2.0 * sigma ** 2 + eps))
    else:
        # 公式 4.23: W_DC^(l) = exp(-JS / τ_w)
        W_DC = torch.exp(-D_dep / (tau_w + eps))

    return W_DC   # [B, T, T]


def maxpool_var_emb(E: torch.Tensor) -> torch.Tensor:
    """
    对变量嵌入序列进行时间维度的最大池化 (公式 4.21)

    Z_{t,i}^{(l+1)} = MaxPool(Z_{2t,i}^{(l)}, Z_{2t+1,i}^{(l)})

    Args:
        E: [B, T, D, d_emb]

    Returns:
        E_pooled: [B, T//2, D, d_emb]  (若 T 为奇数则截断最后一步)
    """
    B, T, D, d = E.shape
    T_even = T - (T % 2)
    # 将 D 和 d 合并后做 max_pool1d, 再拆回
    # E[:, :T_even]: [B, T_even, D, d] → reshape [B*D, d, T_even]
    E_t = E[:, :T_even].permute(0, 2, 3, 1)          # [B, D, d, T_even]
    E_t = E_t.reshape(B * D, d, T_even)
    E_pooled = F.max_pool1d(E_t, kernel_size=2)       # [B*D, d, T_even//2]
    E_pooled = E_pooled.reshape(B, D, d, T_even // 2).permute(0, 3, 1, 2)
    return E_pooled.contiguous()                       # [B, T//2, D, d_emb]


def hierarchical_soft_weights(
    E_list: list,
    dep_module,
    tau_T_base: float = 2.0,
    sigma: float = 1.0,
    pool_factor: int = 2,
    eps: float = 1e-8,
) -> list:
    """
    层次化软权重构建 (公式 4.21-4.26)

    对每一层 l:
      1. 计算依赖矩阵 A^(l) (公式 4.22)
      2. 计算依赖一致性权重 W_DC^(l) (公式 4.23)
      3. 计算时间邻近性权重 (Sigmoid 衰减)
      4. 综合得到 W_T^(l) = 2·σ(-τ_T^(l)|t-t'|) · W_DC^(l) (公式 4.26)

    Args:
        E_list:      [E^(0), E^(1), ...]  各层变量嵌入, E^(l): [B, T_l, D, d_emb]
        dep_module:  DependencyPatternModule 实例 (计算 A^(l))
        tau_T_base:  基础温度系数 τ̃_T (公式 4.25)
        sigma:       高斯核带宽 σ (公式 4.20)
        pool_factor: 池化核尺寸 m (公式 4.25)
        eps:         数值稳定常数

    Returns:
        W_T_list: [W_T^(0), W_T^(1), ...]  各层软权重, W_T^(l): [B, T_l, T_l]
    """
    W_T_list = []

    for l, E_l in enumerate(E_list):
        B, T_l, D, d_emb = E_l.shape

        # ── 公式 4.22: A_t^(l) = Softmax(Q_t K_t^T / √d) ──────────────────
        A_l = dep_module(E_l)   # [B, T_l, D, D]

        # ── 公式 4.23: W_DC^(l) = exp(-JS(A_u^(l) || A_v^(l)) / τ_w) ──────
        W_DC_l = dep_consistency_weight(
            A_l, sigma=sigma, tau_w=1.0, eps=eps, use_gaussian=True
        )   # [B, T_l, T_l]

        # ── 公式 4.25: τ_T^(l) = m^l · τ̃_T ────────────────────────────────
        tau_T_l = (pool_factor ** l) * tau_T_base

        # ── 时间邻近性权重: σ(-τ_T^(l) |t-t'|) ────────────────────────────
        # 构造时间差矩阵 |t-t'|: [T_l, T_l]
        t_idx = torch.arange(T_l, device=E_l.device).float()
        t_diff = (t_idx.unsqueeze(0) - t_idx.unsqueeze(1)).abs()  # [T_l, T_l]
        time_weight = torch.sigmoid(-tau_T_l * t_diff)             # [T_l, T_l]

        # ── 公式 4.26: W_T^(l) = 2 · σ(-τ_T^(l)|t-t'|) · W_DC^(l) ─────────
        W_T_l = 2.0 * time_weight.unsqueeze(0) * W_DC_l            # [B, T_l, T_l]

        W_T_list.append(W_T_l)

    return W_T_list


def temporal_soft_cl_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    W_T: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    单层时间软对比损失 (公式 4.27-4.28)

    z1, z2 是同一时间窗口的两个视图的嵌入序列。
    将 [z1; z2] 拼接为长度 2T 的序列, 对每个时间步 t 计算:

      p(i,(t,t')) = exp(z_{i,t} · z_{i,t'}) / Σ_{u≠t} exp(z_{i,t} · z_{i,u})

      L^{(i,t)} = -log p(i,(t, t+T))
                  - Σ_{u≠{t,t+T}} w_T(t, u mod T) · log p(i,(t,u))

    Args:
        z1:  [B, T, C]  视图 1 的嵌入 (已 L2 归一化)
        z2:  [B, T, C]  视图 2 的嵌入 (已 L2 归一化)
        W_T: [B, T, T]  软权重矩阵 (公式 4.26)

    Returns:
        loss: 标量
    """
    B, T, C = z1.shape

    # 拼接两个视图: [B, 2T, C]
    z = torch.cat([z1, z2], dim=1)

    # 相似度矩阵: [B, 2T, 2T]
    sim = torch.bmm(z, z.transpose(1, 2))   # 点积相似度

    # 排除自身 (对角线置 -inf)
    mask_self = torch.eye(2 * T, device=z.device).bool().unsqueeze(0)
    sim = sim.masked_fill(mask_self, float('-inf'))

    # log-softmax 概率: [B, 2T, 2T]
    log_prob = F.log_softmax(sim, dim=-1)

    # ── 构造软权重矩阵 W_full: [B, 2T, 2T] ─────────────────────────────────
    # 对于 t ∈ [0,T): 正样本为 t+T; 软负样本权重来自 W_T[t, u mod T]
    # 对于 t ∈ [T,2T): 正样本为 t-T; 软负样本权重来自 W_T[t-T, u mod T]
    W_full = torch.zeros(B, 2 * T, 2 * T, device=z.device)

    # 填充软权重: W_full[b, t, u] = W_T[b, t mod T, u mod T]
    # 利用广播: W_T [B, T, T] → 扩展到 [B, 2T, 2T]
    W_full = W_T.repeat(1, 2, 2)   # [B, 2T, 2T]  (平铺)

    # 正样本位置 (t → t+T 和 t+T → t) 不参与软权重加权, 置 0
    pos_idx = torch.arange(T, device=z.device)
    W_full[:, pos_idx, pos_idx + T] = 0.0
    W_full[:, pos_idx + T, pos_idx] = 0.0

    # 自身位置也置 0
    W_full = W_full.masked_fill(mask_self, 0.0)

    # ── 公式 4.28: L^{(i,t)} ────────────────────────────────────────────────
    # 硬正样本项: -log p(i,(t, t+T))
    loss_pos_fwd = -log_prob[:, pos_idx, pos_idx + T]   # [B, T]
    loss_pos_bwd = -log_prob[:, pos_idx + T, pos_idx]   # [B, T]

    # 软负样本项: -Σ_{u≠{t,t+T}} w_T · log p(i,(t,u))
    # 注意: log_prob 对角线为 -inf (自身 mask), 0 * (-inf) = nan
    # 用 nan_to_num 将 -inf 替换为 0, 因为对应 W_full 权重也为 0
    log_prob_safe = log_prob.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    loss_soft_fwd = -(W_full[:, :T, :] * log_prob_safe[:, :T, :]).sum(dim=-1)   # [B, T]
    loss_soft_bwd = -(W_full[:, T:, :] * log_prob_safe[:, T:, :]).sum(dim=-1)   # [B, T]

    # ── 公式 4.29: L_CL = (1/4BT) Σ L^{(i,t)} ──────────────────────────────
    loss = (loss_pos_fwd + loss_soft_fwd + loss_pos_bwd + loss_soft_bwd).sum()
    loss = loss / (4.0 * B * T)
    return loss


def hierarchical_dependency_soft_cl_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    E: torch.Tensor,
    dep_module,
    tau_T_base: float = 2.0,
    sigma: float = 1.0,
    tau_w: float = 1.0,
    pool_factor: int = 2,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    层次化依赖模式一致性软对比损失 (公式 4.21-4.29)

    沿时间轴反复做 MaxPool(kernel=2), 直到 T=1 为止, 在每一层独立计算对比损失。
    层数由窗口长度 T 自动决定: L = floor(log2(T)) + 1 层
    例如 T=100: 100→50→25→12→6→3→1, 共 7 层。
    最顶层 T=1 包含整个实例的全局信息。

    Args:
        z1:  [B, T, C]  视图 1 的编码器输出 (投影后)
        z2:  [B, T, C]  视图 2 的编码器输出
        E:   [B, T, D, d_emb]  原始视图的变量独立嵌入 (用于依赖矩阵)
        dep_module:  DependencyPatternModule 实例
        tau_T_base:  基础温度系数 τ̃_T (公式 4.25)
        sigma:       高斯核带宽 σ (公式 4.20, 基础层使用)
        tau_w:       温度系数 τ_w (公式 4.23, 层次化使用)
        pool_factor: 池化核尺寸 m, 默认 2 (公式 4.25: Φ(m,k)=m^k)
        eps:         数值稳定常数

    Returns:
        L_CL: 标量, 所有层损失的均值
    """
    loss_total = 0.0
    n_layers   = 0

    z1_cur = z1   # [B, T_l, C]
    z2_cur = z2   # [B, T_l, C]
    E_cur  = E    # [B, T_l, D, d_emb]

    while True:
        T_l = z1_cur.shape[1]

        # ── 公式 4.22: 当前层依赖矩阵 A^(l) ─────────────────────────────────
        A_l = dep_module(E_cur)   # [B, T_l, D, D]

        # ── 公式 4.23 / 4.26: 软权重 W_T^(l) ────────────────────────────────
        # 依赖一致性权重 W_DC^(l): [B, T_l, T_l]
        # 公式 4.23: W_DC^(l) = exp(-JS / τ_w)  (指数衰减, 非高斯核)
        W_DC_l = dep_consistency_weight(A_l, tau_w=tau_w, eps=eps, use_gaussian=False)

        # 公式 4.25: τ_T^(l) = m^l · τ̃_T  (l = n_layers, 从 0 开始)
        tau_T_l = (pool_factor ** n_layers) * tau_T_base

        # 时间邻近性权重: 2·σ(-τ_T^(l)|t-t'|)
        t_idx  = torch.arange(T_l, device=z1_cur.device).float()
        t_diff = (t_idx.unsqueeze(0) - t_idx.unsqueeze(1)).abs()   # [T_l, T_l]
        time_w = torch.sigmoid(-tau_T_l * t_diff)                   # [T_l, T_l]

        # 公式 4.26: W_T^(l) = 2·σ(·) · W_DC^(l)
        W_T_l = 2.0 * time_w.unsqueeze(0) * W_DC_l                 # [B, T_l, T_l]

        # ── 公式 4.27-4.28: 当前层对比损失 ───────────────────────────────────
        z1_n = F.normalize(z1_cur, p=2, dim=-1)
        z2_n = F.normalize(z2_cur, p=2, dim=-1)
        loss_l = temporal_soft_cl_loss(z1_n, z2_n, W_T_l, eps)
        loss_total += loss_l
        n_layers   += 1

        # T=1 时已到顶层 (整个实例的全局表示), 停止
        if T_l == 1:
            break

        # ── 公式 4.21: MaxPool 降采样到下一层 ────────────────────────────────
        # z: [B, T_l, C] → max_pool1d 需要 [B, C, T_l] → [B, C, T_l//2]
        z1_cur = F.max_pool1d(
            z1_cur.transpose(1, 2), kernel_size=2
        ).transpose(1, 2)                                           # [B, T_l//2, C]
        z2_cur = F.max_pool1d(
            z2_cur.transpose(1, 2), kernel_size=2
        ).transpose(1, 2)

        # E: [B, T_l, D, d_emb] → maxpool_var_emb → [B, T_l//2, D, d_emb]
        E_cur = maxpool_var_emb(E_cur)

    # 公式 4.29: L_CL = 各层损失均值
    return loss_total / n_layers


def total_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    E: torch.Tensor,
    dep_module,
    M_freq: torch.Tensor,
    tau_T_base: float = 2.0,
    sigma: float = 1.0,
    tau_w: float = 1.0,
    pool_factor: int = 2,
    lambda_reg: float = 0.5,
    eps: float = 1e-8,
) -> tuple:
    """
    AS2CL-AD 总损失 (公式 4.31): L_total = L_CL + λ·L_reg

    Args:
        z1:  [B, T, C]        视图 1 投影嵌入
        z2:  [B, T, C]        视图 2 投影嵌入
        E:   [B, T, D, d_emb] 原始视图变量独立嵌入
        dep_module:  DependencyPatternModule
        M_freq:      [F]  ASSA 软频谱掩码
        tau_T_base, sigma, tau_w, pool_factor, lambda_reg, eps: 同上

    Returns:
        (L_total, L_CL, L_reg)
    """
    L_CL = hierarchical_dependency_soft_cl_loss(
        z1, z2, E, dep_module, tau_T_base, sigma, tau_w, pool_factor, eps
    )
    L_reg   = M_freq.abs().mean()                  # 公式 4.30
    L_total = L_CL + lambda_reg * L_reg            # 公式 4.31
    return L_total, L_CL, L_reg


if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from model.embedding import VariableIndependentEmbedding
    from model.attn_layer import DependencyPatternModule

    B, T, D, d_emb, C = 4, 100, 12, 64, 128

    z1 = torch.randn(B, T, C, requires_grad=True)
    z2 = torch.randn(B, T, C, requires_grad=True)
    x  = torch.randn(B, T, D)

    emb_model = VariableIndependentEmbedding(n_vars=D, d_emb=d_emb)
    dep_model = DependencyPatternModule(d_emb=d_emb)
    E = emb_model(x)   # [B, T, D, d_emb]

    M_freq = torch.sigmoid(torch.randn(T // 2 + 1))

    # 层数由 T 自动决定: T=100 → 100,50,25,12,6,3,1 共 7 层
    L_total, L_CL, L_reg = total_loss(z1, z2, E, dep_model, M_freq)

    import math
    expected_layers = math.floor(math.log2(T)) + 1
    print(f"[Hierarchical CL]  T={T}, expected layers={expected_layers}")
    print(f"  L_CL   : {L_CL.item():.4f}")
    print(f"  L_reg  : {L_reg.item():.4f}")
    print(f"  L_total: {L_total.item():.4f}")

    L_total.backward()
    print(f"  backward() OK, dep W_Q grad max: "
          f"{dep_model.W_Q.weight.grad.abs().max().item():.2e}")



