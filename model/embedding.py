import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        self.pe = torch.zeros((max_len, d_model), dtype=torch.float)
        self.pe.requires_grad = False

        pos = torch.arange(0, max_len).float().unsqueeze(1)
        _2i = torch.arange(0, d_model, step=2).float()

        self.pe[:, ::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.pe[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

        self.pe = self.pe.unsqueeze(0)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
    

class TokenEmbedding(nn.Module):
    """
    逐时间步线性投影 (论文公式 4.8)

    z_t^0 = x_t W_emb + b_emb,  W_emb ∈ R^{D × d_model}

    使用 Conv1d(kernel_size=1) 实现, 等价于对每个时间步独立做线性变换,
    不混合相邻时间步的信息。
    """
    def __init__(self, in_dim, d_model):
        super(TokenEmbedding, self).__init__()
        # kernel_size=1: 逐点卷积, 等价于 nn.Linear(in_dim, d_model)
        # 每个时间步独立映射, 符合公式 4.8 的逐时间步线性投影
        self.conv = nn.Conv1d(in_channels=in_dim, out_channels=d_model,
                              kernel_size=1, bias=True)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        # x: [B, L, D] → permute → [B, D, L] → conv1d → [B, d_model, L] → [B, L, d_model]
        return self.conv(x.permute(0, 2, 1)).transpose(1, 2)


class InputEmbedding(nn.Module):
    """
    输入嵌入层 (论文公式 4.8-4.9)

    公式 4.8: z_t^0 = x_t W_emb + b_emb
    公式 4.9: E_t = z_t^0 + PE(t)

    对每个时间步 t 的多维输入 x_t ∈ R^D 进行线性投影到 d_model 维,
    然后叠加正弦位置编码。

    输入:  x [B, L, D]
    输出:  E [B, L, d_model]
    """
    def __init__(self, in_dim, d_model, device, dropout=0.0):
        super(InputEmbedding, self).__init__()
        self.device = device
        self.token_embedding = TokenEmbedding(in_dim=in_dim, d_model=d_model)
        self.pos_embedding = PositionalEmbedding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):

        pe = self.pos_embedding(x).to(x.device)
        return self.dropout(self.token_embedding(x) + pe)


class VariableIndependentEmbedding(nn.Module):
    """
    变量独立解耦嵌入 (论文 4.2.4 节, 公式 4.15)

    对每个变量 i 独立进行线性映射 Linear_i, 并叠加位置编码 PE(t)。
    不将多变量混合投影为单一隐向量, 显式保留变量维度 D。

        E_{t,i} = Linear_i(x_{t,i}) + PE(t)

    输入:  x [B, T, D]
    输出:  E [B, T, D, d_emb]
    """

    def __init__(self, n_vars: int, d_emb: int,
                 max_len: int = 5000, dropout: float = 0.0):
        """
        Args:
            n_vars:  变量数 D
            d_emb:   每个变量的嵌入维度
            max_len: 位置编码支持的最大序列长度
            dropout: Dropout 概率
        """
        super(VariableIndependentEmbedding, self).__init__()
        self.n_vars = n_vars  # D
        self.d_emb  = d_emb

        # 每个变量 i 的独立线性映射: Linear_i: R^1 → R^{d_emb}
        # 向量化实现: weight [D, d_emb], bias [D, d_emb]
        # 等价于 D 个独立的 nn.Linear(1, d_emb), 但无需 Python 循环
        self.weight = nn.Parameter(torch.empty(n_vars, d_emb))   # [D, d_emb]
        self.bias   = nn.Parameter(torch.zeros(n_vars, d_emb))   # [D, d_emb]
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # 正弦位置编码 PE(t): register_buffer 不参与梯度更新
        pe  = torch.zeros(max_len, d_emb)                        # [max_len, d_emb]
        pos = torch.arange(0, max_len).float().unsqueeze(1)      # [max_len, 1]
        div = torch.exp(
            torch.arange(0, d_emb, 2).float() * (-math.log(10000.0) / d_emb)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))              # [1, max_len, d_emb]

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]  输入多维时间序列窗口
        返回 E: [B, T, D, d_emb]
        """
        B, T, D = x.shape
        assert D == self.n_vars, f"变量数不匹配: 期望 {self.n_vars}, 实际 {D}"

        # 变量独立线性映射 (向量化, 无 Python 循环)
        # x.unsqueeze(-1):              [B, T, D, 1]
        # weight.unsqueeze(0).unsqueeze(0): [1, 1, D, d_emb]
        # E_token:                      [B, T, D, d_emb]
        E_token = (x.unsqueeze(-1) * self.weight.unsqueeze(0).unsqueeze(0)
                   + self.bias.unsqueeze(0).unsqueeze(0))

        # 位置编码 PE(t): [1, T, d_emb] → [1, T, 1, d_emb]  广播到 [B, T, D, d_emb]
        pe = self.pe[:, :T, :].unsqueeze(2)   # [1, T, 1, d_emb]

        E = E_token + pe                       # [B, T, D, d_emb]
        return self.dropout(E)


if __name__ == '__main__':
    B, T, D, d_emb = 4, 100, 12, 64

    x = torch.randn(B, T, D)
    model = VariableIndependentEmbedding(n_vars=D, d_emb=d_emb)

    E = model(x)
    print(f"[VariableIndependentEmbedding]")
    print(f"  input  shape: {x.shape}")   # [4, 100, 12]
    print(f"  output shape: {E.shape}")   # [4, 100, 12, 64]