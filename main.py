"""
AS2CL-AD 入口脚本
用法:
  python main.py --mode train --dataset SMD --data_path ./data/SMD/SMD/
  python main.py --mode test  --dataset SMD --data_path ./data/SMD/SMD/
"""

import os
import argparse
import torch
from torch.backends import cudnn
from solver import Solver


def str2bool(v):
    return v.lower() in ('true', '1', 'yes')


def main(config):
    cudnn.benchmark = True
    os.makedirs(config.model_save_path, exist_ok=True)

    solver = Solver(vars(config))

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()
    else:
        raise ValueError(f'Unknown mode: {config.mode}')

    return solver


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AS2CL-AD')

    # ── 基础配置 ──────────────────────────────────────────────────────────────
    parser.add_argument('--mode',      type=str,   default='train',
                        choices=['train', 'test'])
    parser.add_argument('--dataset',   type=str,   default='SMD')
    parser.add_argument('--data_path', type=str,   default='./data/SMD/SMD/')
    parser.add_argument('--model_save_path', type=str, default='checkpoints')

    # ── 数据 ──────────────────────────────────────────────────────────────────
    parser.add_argument('--win_size',   type=int,   default=100)
    parser.add_argument('--input_c',    type=int,   default=38,
                        help='输入变量维度 D')
    parser.add_argument('--batch_size', type=int,   default=32)

    # ── 训练 ──────────────────────────────────────────────────────────────────
    parser.add_argument('--num_epochs', type=int,   default=10)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--lambda_reg', type=float, default=0.5,
                        help='ASSA 稀疏正则化系数 λ (公式 4.31)')

    # ── 模型结构 ──────────────────────────────────────────────────────────────
    parser.add_argument('--d_model',  type=int,   default=512)
    parser.add_argument('--d_emb',    type=int,   default=64,
                        help='变量独立嵌入维度 d_emb')
    parser.add_argument('--d_proj',   type=int,   default=128,
                        help='投影头输出维度 d_proj')
    parser.add_argument('--n_heads',  type=int,   default=8)
    parser.add_argument('--e_layers', type=int,   default=3)
    parser.add_argument('--d_ff',     type=int,   default=512)
    parser.add_argument('--dropout',  type=float, default=0.0)
    parser.add_argument('--n_hier',   type=int,   default=3,
                        help='层次化池化层数')

    # ── ASSA 超参数 ───────────────────────────────────────────────────────────
    parser.add_argument('--alpha_limit', type=float, default=0.2,
                        help='ASSA 幅度扰动范围 δ')
    parser.add_argument('--noise_std',   type=float, default=0.5,
                        help='ASSA 高斯噪声标准差')

    # ── 软对比超参数 ──────────────────────────────────────────────────────────
    parser.add_argument('--tau_T_base', type=float, default=2.0,
                        help='时间邻近性基础温度 τ̃_T (公式 4.25)')
    parser.add_argument('--sigma',      type=float, default=1.0,
                        help='依赖一致性高斯核带宽 σ (公式 4.20)')

    # ── 测试 / 评分 ───────────────────────────────────────────────────────────
    parser.add_argument('--anormly_ratio', type=float, default=1.0,
                        help='异常比例 (%) 用于确定阈值百分位')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='S_con 与 S_dep 融合系数 β (公式 4.35)')

    config = parser.parse_args()

    print('─' * 50)
    print('  AS2CL-AD Configuration')
    print('─' * 50)
    for k, v in sorted(vars(config).items()):
        print(f'  {k:<20}: {v}')
    print('─' * 50)

    main(config)
