"""
AS2CL-AD Solver — 训练与测试主控模块
参考论文第 4 章算法 1 (训练) 与算法 2 (推理)
"""

import os
import time
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
)
from tqdm import tqdm

from data_factory.data_loader import get_loader_segment

# ── AS2CL-AD 核心模块 ────────────────────────────────────────────────────────
from autoaug.fourier import FreRA                          # ASSA 频域增强
from model.Transformer import AS2CLAD_Encoder             # Transformer + 投影头
from model.embedding import VariableIndependentEmbedding  # 变量独立嵌入
from model.attn_layer import DependencyPatternModule      # 依赖模式矩阵
from model.dependency_soft_cl import (                    # 层次化软对比损失
    total_loss,
    hierarchical_dependency_soft_cl_loss,
    maxpool_var_emb,
    kl_divergence,
)

# ── 评价指标 ─────────────────────────────────────────────────────────────────
from metrics.affiliation.generics import convert_vector_to_events
from metrics.affiliation.metrics import pr_from_events
from metrics.AUC import point_wise_AUC


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def minmax_norm(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Min-Max 归一化到 [0, 1]"""
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + eps)


def adjust_learning_rate(optimizer, epoch: int, base_lr: float):
    """每 epoch 指数衰减学习率"""
    lr = base_lr * (0.5 ** ((epoch - 1) // 1))
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    print(f'  LR adjusted to {lr:.2e}')


class EarlyStopping:
    def __init__(self, patience: int = 10, dataset_name: str = ''):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.dataset = dataset_name

    def __call__(self, val_loss: float, models: dict, path: str):
        score = -val_loss
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self._save(models, path)
            self.counter = 0
        else:
            self.counter += 1
            print(f'  EarlyStopping: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def _save(self, models: dict, path: str):
        for name, m in models.items():
            ckpt = os.path.join(path, f'{self.dataset}_{name}.pth')
            torch.save(m.state_dict(), ckpt)
        print(f'  Checkpoint saved.')


# ─────────────────────────────────────────────────────────────────────────────
# Solver
# ─────────────────────────────────────────────────────────────────────────────

class Solver:
    """AS2CL-AD 训练 / 测试主控"""

    def __init__(self, config: dict):
        self.__dict__.update(config)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ── 数据加载 ──────────────────────────────────────────────────────────
        self.train_loader, self.vali_loader, _ = get_loader_segment(
            self.data_path, batch_size=self.batch_size,
            win_size=self.win_size, mode='train', dataset=self.dataset
        )
        self.test_loader, _ = get_loader_segment(
            self.data_path, batch_size=self.batch_size,
            win_size=self.win_size, mode='test', dataset=self.dataset
        )

        # ── 构建模型 ──────────────────────────────────────────────────────────
        self._build_model()

        # ── 日志 ──────────────────────────────────────────────────────────────
        self.logger = logging.getLogger('AS2CL-AD')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
            self.logger.addHandler(h)

    # ─────────────────────────────────────────────────────────────────────────
    def _build_model(self):
        """初始化所有可训练模块"""
        D = self.input_c

        # 1. ASSA 频域增强模块 (含可学习掩码 S ∈ R^F)
        self.assa = FreRA(
            len_sw=self.win_size,
            alpha_limit=getattr(self, 'alpha_limit', 0.2),
            noise_std=getattr(self, 'noise_std', 0.5),
        ).to(self.device)

        # 2. 变量独立解耦嵌入 (用于依赖模式计算)
        self.var_emb = VariableIndependentEmbedding(
            n_vars=D,
            d_emb=getattr(self, 'd_emb', 64),
            max_len=self.win_size + 10,
        ).to(self.device)

        # 3. 依赖模式模块 (共享 W^Q, W^K)
        self.dep_module = DependencyPatternModule(
            d_emb=getattr(self, 'd_emb', 64),
        ).to(self.device)

        # 4. Transformer 编码器 + 投影头 (连体, 两视图共享权重)
        self.encoder = AS2CLAD_Encoder(
            win_size=self.win_size,
            enc_in=D,
            d_model=self.d_model,
            n_heads=getattr(self, 'n_heads', 8),
            e_layers=getattr(self, 'e_layers', 3),
            d_ff=getattr(self, 'd_ff', 512),
            d_proj=getattr(self, 'd_proj', 128),
            dropout=getattr(self, 'dropout', 0.0),
            device=self.device,
        ).to(self.device)

        # ── 优化器: 覆盖所有可训练参数 ───────────────────────────────────────
        params = (
            list(self.assa.parameters())
            + list(self.var_emb.parameters())
            + list(self.dep_module.parameters())
            + list(self.encoder.parameters())
        )
        self.optimizer = torch.optim.Adam(params, lr=self.lr)

    # ─────────────────────────────────────────────────────────────────────────
    def _load_checkpoint(self):
        """加载所有模块的 checkpoint"""
        for name, m in self._named_modules().items():
            ckpt = os.path.join(self.model_save_path, f'{self.dataset}_{name}.pth')
            m.load_state_dict(torch.load(ckpt, map_location=self.device))
            print(f'  Loaded {ckpt}')

    def _named_modules(self) -> dict:
        return {
            'assa': self.assa,
            'var_emb': self.var_emb,
            'dep_module': self.dep_module,
            'encoder': self.encoder,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 算法 1: 训练
    # ─────────────────────────────────────────────────────────────────────────
    def train(self):
        print('=' * 55)
        print('  TRAIN MODE  —  AS2CL-AD  (Algorithm 1)')
        print('=' * 55)

        os.makedirs(self.model_save_path, exist_ok=True)
        early_stop = EarlyStopping(patience=10, dataset_name=self.dataset)
        lambda_reg = getattr(self, 'lambda_reg', 0.5)   # λ for L_reg (公式 4.31)
        n_hier     = getattr(self, 'n_hier', 3)          # 层次化池化层数

        for epoch in range(1, self.num_epochs + 1):
            self.assa.train()
            self.var_emb.train()
            self.dep_module.train()
            self.encoder.train()

            loss_epoch, t0 = [], time.time()

            for batch_x, _ in tqdm(self.train_loader,
                                   desc=f'Epoch {epoch}/{self.num_epochs}',
                                   leave=False):
                # ── 数据准备 ─────────────────────────────────────────────────
                # x1: 原始视图  x2: ASSA 增强视图
                # x1, x2: [B, L, D]
                x1 = batch_x.float().to(self.device)

                # 算法 1 步骤 05: x2 ← ASSA_augment(x1)
                # 训练模式下 assa 返回增强视图; M_freq 用于 L_reg
                x2, M_freq = self.assa(x1)   # x2: [B,L,D], M_freq: [F]

                # ── 变量独立嵌入 → 层次化依赖模式 ────────────────────────────
                # 算法 1 步骤 06-08: 计算 E_dec 和 W_DC
                # E: [B, L, D, d_emb]
                E = self.var_emb(x1)

                # 构造层次化嵌入列表 (公式 4.21 MaxPool)
                E_list = [E]
                for _ in range(n_hier - 1):
                    E = maxpool_var_emb(E)
                    E_list.append(E)

                # ── Transformer 编码 + 投影头 ─────────────────────────────────
                # 算法 1 步骤 09-17: 两视图共享编码器权重
                # Z: [B, L, d_proj]  H: [B, L, d_model]
                Z1, _ = self.encoder(x1)
                Z2, _ = self.encoder(x2)

                # 构造各层 z 嵌入列表 (对应 E_list 的时间分辨率)
                # 粗粒度层通过 max_pool1d 降采样 Z
                z1_list, z2_list = [Z1], [Z2]
                z1_cur, z2_cur = Z1, Z2
                for _ in range(n_hier - 1):
                    # max_pool1d 作用在时间维度: [B, L, C] → [B, L//2, C]
                    z1_cur = F.max_pool1d(
                        z1_cur.transpose(1, 2), kernel_size=2
                    ).transpose(1, 2)
                    z2_cur = F.max_pool1d(
                        z2_cur.transpose(1, 2), kernel_size=2
                    ).transpose(1, 2)
                    z1_list.append(z1_cur)
                    z2_list.append(z2_cur)

                # ── 损失计算 ──────────────────────────────────────────────────
                # 算法 1 步骤 18-19: L_total = L_CL + λ·L_reg (公式 4.31)
                L_total, L_CL, L_reg = total_loss(
                    z1_list, z2_list, E_list,
                    dep_module=self.dep_module,
                    M_freq=M_freq,
                    tau_T_base=getattr(self, 'tau_T_base', 2.0),
                    sigma=getattr(self, 'sigma', 1.0),
                    pool_factor=2,
                    lambda_reg=lambda_reg,
                )

                # ── 反向传播 ──────────────────────────────────────────────────
                self.optimizer.zero_grad()
                L_total.backward()
                self.optimizer.step()

                loss_epoch.append(L_total.item())

            # ── Epoch 结束: 验证 + 早停 + 保存 ──────────────────────────────
            avg_train = np.mean(loss_epoch)
            avg_vali  = self._vali()

            print(f'Epoch {epoch:3d} | '
                  f'Train L={avg_train:.4f} | Vali L={avg_vali:.4f} | '
                  f'Time={time.time()-t0:.1f}s')

            early_stop(avg_vali, self._named_modules(), self.model_save_path)
            if early_stop.early_stop:
                print('Early stopping triggered.')
                break

    # ─────────────────────────────────────────────────────────────────────────
    def _vali(self) -> float:
        """验证集上计算平均对比损失 (不做增强, 仅用于早停监控)"""
        self.assa.eval(); self.var_emb.eval()
        self.dep_module.eval(); self.encoder.eval()

        losses = []
        n_hier = getattr(self, 'n_hier', 3)
        with torch.no_grad():
            for batch_x, _ in self.vali_loader:
                x1 = batch_x.float().to(self.device)
                x2, M_freq = self.assa(x1)

                E = self.var_emb(x1)
                E_list = [E]
                for _ in range(n_hier - 1):
                    E = maxpool_var_emb(E)
                    E_list.append(E)

                Z1, _ = self.encoder(x1)
                Z2, _ = self.encoder(x2)
                z1_list, z2_list = [Z1], [Z2]
                z1_c, z2_c = Z1, Z2
                for _ in range(n_hier - 1):
                    z1_c = F.max_pool1d(z1_c.transpose(1,2), 2).transpose(1,2)
                    z2_c = F.max_pool1d(z2_c.transpose(1,2), 2).transpose(1,2)
                    z1_list.append(z1_c); z2_list.append(z2_c)

                L, _, _ = total_loss(
                    z1_list, z2_list, E_list, self.dep_module, M_freq,
                    lambda_reg=getattr(self, 'lambda_reg', 0.5),
                )
                losses.append(L.item())
        return float(np.mean(losses))

    # ─────────────────────────────────────────────────────────────────────────
    # 算法 2: 推理与异常打分
    # ─────────────────────────────────────────────────────────────────────────
    def test(self):
        print('=' * 55)
        print('  TEST MODE  —  AS2CL-AD  (Algorithm 2)')
        print('=' * 55)

        # ── 加载 checkpoint ───────────────────────────────────────────────────
        self._load_checkpoint()

        # ── 冻结所有参数 (含 ASSA 掩码 S) ────────────────────────────────────
        self.assa.eval(); self.var_emb.eval()
        self.dep_module.eval(); self.encoder.eval()

        beta = getattr(self, 'beta', 0.5)   # 公式 4.35 融合系数
        eps  = 1e-8

        # 收集每个窗口的分数和标签
        # 每个窗口产生 1 个 S_con (窗口级) 和 1 个 S_dep (窗口级)
        s_con_all, s_dep_all, labels_all = [], [], []

        with torch.no_grad():
            for batch_x, batch_y in tqdm(self.test_loader, desc='Scoring'):
                # x1: 原始视图  x2: ASSA 增强视图 (eval 模式不做增强, 手动调 train)
                x1 = batch_x.float().to(self.device)   # [B, L, D]

                # 推理阶段: 临时切换 assa 到 train 以生成增强视图
                self.assa.train()
                with torch.no_grad():
                    x2, _ = self.assa(x1)
                self.assa.eval()

                B, L, D = x1.shape

                # ── 步骤 03-05: 编码两视图 ────────────────────────────────────
                # Z1, Z2: [B, L, d_proj]
                Z1, _ = self.encoder(x1)
                Z2, _ = self.encoder(x2)

                # ── 步骤 07: 语义对齐分数 S_con (公式 4.32) ──────────────────
                # 逐时间步计算余弦距离, 保留时间维度
                # Z1, Z2: [B, L, d_proj]
                z1_n = F.normalize(Z1, p=2, dim=-1)
                z2_n = F.normalize(Z2, p=2, dim=-1)
                # s_con_win: [B, L]  每个时间步的语义对齐分数
                s_con_win = (1.0 - (z1_n * z2_n).sum(dim=-1))   # [B, L]

                # ── 步骤 09-14: 依赖分布一致性分数 S_dep (公式 4.33-4.34) ────
                E1 = self.var_emb(x1)   # [B, L, D, d_emb]
                E2 = self.var_emb(x2)   # [B, L, D, d_emb]

                # 逐时间步依赖矩阵: [B, L, D, D]
                A1 = self.dep_module(E1)
                A2 = self.dep_module(E2)

                # 公式 4.33: 全局依赖分布 P, Q: [B, D, D]
                P = A1.mean(dim=1)
                Q = A2.mean(dim=1)

                # 公式 4.34: 对称 KL 散度, 对变量维度求和 → 窗口级标量 [B]
                # 再广播到时间维度, 使每个时间步共享同一窗口的 S_dep
                kl_pq = kl_divergence(P.clamp(eps), Q.clamp(eps), eps)  # [B, D]
                kl_qp = kl_divergence(Q.clamp(eps), P.clamp(eps), eps)  # [B, D]
                s_dep_scalar = (kl_pq + kl_qp).sum(dim=-1)              # [B]
                # 扩展到 [B, L], 窗口内每个时间步共享同一 S_dep 值
                s_dep_win = s_dep_scalar.unsqueeze(1).expand(-1, L)      # [B, L]

                # ── 展平为时间点序列 [B*L] ────────────────────────────────────
                # s_con_win, s_dep_win: [B, L] → reshape [B*L]
                s_con_all.append(s_con_win.cpu().numpy().reshape(-1))    # [B*L]
                s_dep_all.append(s_dep_win.cpu().numpy().reshape(-1))    # [B*L]

                # ── 收集标签: 保持逐时间步 ────────────────────────────────────
                # batch_y: [B, L] (数据集返回每个时间步的标签)
                y = batch_y.numpy()
                if y.ndim == 1:
                    # 若 DataLoader 只返回窗口级标签 [B], 广播到 [B*L]
                    y = np.repeat(y, L)
                else:
                    # [B, L] → [B*L]
                    y = y.reshape(-1)
                labels_all.append(y)

        # ── 拼接为一维 numpy 数组 (时间点级) ────────────────────────────────
        # 每个窗口贡献 B*L 个时间点, 最终长度 ≈ N_test_timesteps
        s_con   = np.concatenate(s_con_all,  axis=0)   # [N_timesteps]
        s_dep   = np.concatenate(s_dep_all,  axis=0)   # [N_timesteps]
        gt      = np.concatenate(labels_all, axis=0).astype(int)  # [N_timesteps]

        # ── 步骤 15: 公式 4.35 融合 ───────────────────────────────────────────
        s_con_n = minmax_norm(s_con)
        s_dep_n = minmax_norm(s_dep)
        score   = beta * s_con_n + (1.0 - beta) * s_dep_n   # [N_windows]

        # ── 阈值: 按异常比例百分位 ────────────────────────────────────────────
        thresh = np.percentile(score, 100 - self.anormly_ratio)
        pred   = (score >= thresh).astype(int)

        print(f'  Threshold = {thresh:.4f}  '
              f'(anormly_ratio={self.anormly_ratio}%)')
        print(f'  pred shape: {pred.shape},  gt shape: {gt.shape}')

        # ── Point-Adjust (PA) 后处理 ──────────────────────────────────────────
        pred_pa = pred.copy()
        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred_pa[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    pred_pa[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    pred_pa[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred_pa[i] = 1

        # ── 评价指标计算 ──────────────────────────────────────────────────────
        self._evaluate(score, pred_pa, gt)

    # ─────────────────────────────────────────────────────────────────────────
    def _evaluate(self, score: np.ndarray, pred: np.ndarray, gt: np.ndarray):
        """计算并打印所有评价指标 (对应论文表 4.2 / 4.3)"""
        eps = 1e-8

        # 1. 基础指标
        acc = accuracy_score(gt, pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            gt, pred, average='binary', zero_division=0
        )

        # 2. AUC-ROC / AUC-PR (基于连续分数)
        try:
            auc_roc = roc_auc_score(gt, score)
        except ValueError:
            auc_roc = float('nan')
        try:
            auc_pr = average_precision_score(gt, score)
        except ValueError:
            auc_pr = float('nan')

        # 3. Affiliation F-score (Aff-F)
        try:
            events_pred = convert_vector_to_events(pred)
            events_gt   = convert_vector_to_events(gt)
            Trange      = (0, len(gt))
            aff         = pr_from_events(events_pred, events_gt, Trange)
            aff_p  = aff['precision']
            aff_r  = aff['recall']
            aff_f  = 2 * aff_p * aff_r / (aff_p + aff_r + eps)
        except Exception as e:
            aff_p = aff_r = aff_f = float('nan')
            print(f'  [WARN] Affiliation metric failed: {e}')

        # ── 表格输出 ──────────────────────────────────────────────────────────
        sep = '-' * 55
        print(sep)
        print(f'  Dataset : {self.dataset}')
        print(sep)
        print(f'  {"Metric":<22} {"Value":>10}')
        print(sep)
        print(f'  {"Accuracy":<22} {acc:>10.4f}')
        print(f'  {"Precision":<22} {prec:>10.4f}')
        print(f'  {"Recall":<22} {rec:>10.4f}')
        print(f'  {"F1":<22} {f1:>10.4f}')
        print(f'  {"Aff-Precision":<22} {aff_p:>10.4f}')
        print(f'  {"Aff-Recall":<22} {aff_r:>10.4f}')
        print(f'  {"Aff-F (Aff-F)":<22} {aff_f:>10.4f}')
        print(f'  {"AUC-ROC (A-ROC)":<22} {auc_roc:>10.4f}')
        print(f'  {"AUC-PR  (A-PR)":<22} {auc_pr:>10.4f}')
        print(sep)
        print(f'  [Summary] F1={f1:.4f} | Aff-F={aff_f:.4f} | '
              f'A-ROC={auc_roc:.4f} | A-PR={auc_pr:.4f}')
        print(sep)

        self.logger.info(
            f'{self.dataset} | F1={f1:.4f} Aff-F={aff_f:.4f} '
            f'A-ROC={auc_roc:.4f} A-PR={auc_pr:.4f}'
        )
        return {'f1': f1, 'aff_f': aff_f, 'auc_roc': auc_roc, 'auc_pr': auc_pr}



