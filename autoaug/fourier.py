import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class FreRA(Module):
    """
    ASSA: Adaptive Spectral Saliency Augmentation
    自适应频谱显著性增强模块 (论文 4.2.1 节)

    输入: x  [B, L, D]  (Batch, 序列长度, 变量维度)
    输出: (x_out [B, L, D], M_freq [F])
        x_out  — 训练时为增强视图, 推理时为原始输入
        M_freq — 软频谱掩码, 供稀疏性正则化损失 L_reg 使用 (公式 4.7)
    """

    def __init__(self, len_sw, device=None, dtype=None,
                 alpha_limit=0.2, noise_std=0.5) -> None:
        super(FreRA, self).__init__()
        print('Initializing ASSA (Adaptive Spectral Saliency Augmentation)')
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.len_sw = len_sw
        # F = L//2 + 1  (rfft 输出的频率分量数)
        self.n_freq = len_sw // 2 + 1

        # 全局共享可学习参数向量 S ∈ R^F  (公式 4.2)
        # 在 D 个变量维度上共享, 强制模型关注所有传感器共有的周期性模式
        self.saliency_weight = Parameter(
            torch.empty(self.n_freq, **factory_kwargs)
        )
        self.reset_parameters()

        # alpha ~ U(-alpha_limit, alpha_limit) 的范围
        self.alpha_limit = alpha_limit
        # epsilon ~ N(0, noise_std^2) 的标准差
        self.noise_std = noise_std

    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.saliency_weight, mean=0.0, std=0.1)

    def forward(self, x: torch.Tensor):
        """
        x: [B, L, D]  输入标准化多维时间序列窗口
        返回:
            x_out:  [B, L, D]  增强视图 (train) / 原始输入 (eval)
            M_freq: [F]        软频谱掩码, 用于正则化损失 (公式 4.7)
        """
        # ── 公式 4.1: DFT, 沿时间维度 (dim=1) ──────────────────────────────
        # x_freq: [B, F, D],  F = L//2 + 1
        x_freq = torch.fft.rfft(x, dim=1)

        # ── 公式 4.2: 软频谱掩码 M_freq ∈ [0,1]^F ──────────────────────────
        # M_freq: [F]  (在 D 个变量维度上共享, 保留多变量通道间相关性)
        M_freq = torch.sigmoid(self.saliency_weight)

        if self.training:
            # ── 公式 4.3: 幅相解耦 ──────────────────────────────────────────
            A   = torch.abs(x_freq)    # 幅度谱: [B, F, D]
            phi = torch.angle(x_freq)  # 相位谱: [B, F, D]  (冻结, 不参与增强)

            # ── 公式 4.4: 自适应幅度增强 ────────────────────────────────────
            # alpha ~ U(-alpha_limit, alpha_limit), 标量, 模拟正常信号的微小幅值波动
            alpha = (torch.rand(1, device=x.device) * 2 - 1) * self.alpha_limit

            # epsilon ~ N(0, noise_std^2): [B, F, D]  实数高斯噪声 (作用于幅度域)
            epsilon = torch.randn_like(A) * self.noise_std

            # M_expanded: [1, F, 1]  广播到 [B, F, D]
            M_expanded = M_freq.view(1, -1, 1)

            # A_aug = ReLU( A ⊙ (1 + α·M) + ε ⊙ (1 - M) )
            # 显著频段 (M→1): 保留并微小波动; 背景频段 (M→0): 注入随机噪声
            A_aug = torch.relu(
                A * (1 + alpha * M_expanded) + epsilon * (1 - M_expanded)
            )

            # ── 公式 4.5: 重组增强频谱 (冻结相位) ──────────────────────────
            # x_freq_aug: [B, F, D]
            x_freq_aug = torch.polar(A_aug, phi)  # A_aug * exp(j * phi)

            # ── 公式 4.6: IDFT 映射回时域 ───────────────────────────────────
            # x_out: [B, L, D]
            x_out = torch.fft.irfft(x_freq_aug, n=self.len_sw, dim=1)
        else:
            # 推理阶段不做增强, 直接返回原始输入
            x_out = x

        # 同时返回掩码 M_freq, 供外部计算稀疏性正则化损失 L_reg (公式 4.7)
        return x_out, M_freq


if __name__ == '__main__':
    B, L, D = 4, 100, 12
    x = torch.randn(B, L, D)

    model = FreRA(len_sw=L, alpha_limit=0.2, noise_std=0.5)

    # 训练模式: 返回增强视图
    model.train()
    x_aug, M = model(x)
    print(f"[ASSA train] input shape : {x.shape}")      # [4, 100, 12]
    print(f"[ASSA train] x_aug shape : {x_aug.shape}")  # [4, 100, 12]
    print(f"[ASSA train] M_freq shape: {M.shape}")       # [51]  (F = 100//2+1)

    # 推理模式: 返回原始输入
    model.eval()
    with torch.no_grad():
        x_out, M_eval = model(x)
    print(f"[ASSA eval]  x_out shape : {x_out.shape}")   # [4, 100, 12]
    print(f"[ASSA eval]  M_freq shape: {M_eval.shape}")   # [51]
