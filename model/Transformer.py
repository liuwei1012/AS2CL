import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn_layer import AttentionLayer
from .embedding import TokenEmbedding, InputEmbedding

# ours
from .ours_memory_module import MemoryModule
# memae
# from .memae_memory_module import MemoryModule
# mnad
# from .mnad_memory_module import MemoryModule

class EncoderLayer(nn.Module):
    def __init__(self, attn, d_model, d_ff=None, dropout=0.1, activation='relu'):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff if d_ff is not None else 4 * d_model
        self.attn_layer = attn
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x):
        '''
        x : N x L x C(=d_model)
        '''
        out = self.attn_layer(x)
        x = x + self.dropout(out)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y)    # N x L x C(=d_model)
    

# Transformer Encoder
class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x):
        '''
        x : N x L x C(=d_model)
        '''
        for attn_layer in self.attn_layers:
            x = attn_layer(x)

        if self.norm is not None:
            x = self.norm(x)

        return x
    
class Decoder(nn.Module):
    def __init__(self, d_model, c_out, d_ff=None, activation='relu', dropout=0.1):
        super(Decoder, self).__init__()
        # self.decoder_layer = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=2,
        #                              batch_first=True, bidirectional=True)
        self.out_linear = nn.Linear(d_model, c_out)
        d_ff = d_ff if d_ff is not None else 4 * d_model
        self.decoder_layer1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)

        self.decoder_layer_add = nn.Conv1d(in_channels=d_ff, out_channels=d_ff, kernel_size=1)

        self.decoder_layer2 = nn.Conv1d(in_channels=d_ff, out_channels=c_out, kernel_size=1)
        self.activation = F.relu if activation == 'relu' else F.gelu
        self.dropout = nn.Dropout(p=dropout)
        self.batchnorm = nn.BatchNorm1d(d_ff)

    def forward(self, x):
        '''
        x : N x L x C(=d_model)
        '''

        # out = self.decoder_layer1(x.transpose(-1, 1))
        # out = self.dropout(self.activation(self.batchnorm(out)))

        # decoder ablation
        # for _ in range(10):
        #     out = self.dropout(self.activation(self.decoder_layer_add(out)))
        #
        # out = self.decoder_layer2(out).transpose(-1, 1)
        '''
        out : reconstructed output
        '''
        out = self.out_linear(x)
        return out      # N x L x c_out


class ProjectionHead(nn.Module):
    """
    投影头 (论文 4.2.2 节, 图 4.4, 公式 4.13/4.14)

    将编码器输出 H ∈ R^{L × d_model} 映射到对比学习的潜在空间 Z。
    结构: Linear → ReLU → Linear
    两个视图共享同一投影头权重 (连体网络)。

    输入:  H [B, L, d_model]
    输出:  Z [B, L, d_proj]
    """
    def __init__(self, d_model: int, d_proj: int, dropout: float = 0.0):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),   # 第一个线性层
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model, d_proj),    # 第二个线性层
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, L, d_model]  →  Z: [B, L, d_proj]
        return self.net(h)


class AS2CLAD_Encoder(nn.Module):
    """
    AS2CL-AD 特征提取主干 (论文 4.2.2 节)

    流程: InputEmbedding → Transformer Encoder (N 层) → ProjectionHead
    两个增强视图共享全部权重 (连体结构)。

    输入:  x  [B, L, D]
    输出:  Z  [B, L, d_proj]   对比学习潜在特征
           H  [B, L, d_model]  编码器原始输出 (供依赖模式模块使用)
    """
    def __init__(self, win_size: int, enc_in: int, d_model: int = 512,
                 n_heads: int = 8, e_layers: int = 3, d_ff: int = 512,
                 d_proj: int = 128, dropout: float = 0.0,
                 activation: str = 'gelu', device=None):
        super(AS2CLAD_Encoder, self).__init__()

        self.embedding = InputEmbedding(
            in_dim=enc_in, d_model=d_model, dropout=dropout, device=device
        )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(win_size, d_model, n_heads, dropout=dropout),
                    d_model, d_ff, dropout=dropout, activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )

        # 共享投影头 (公式 4.13 / 4.14)
        self.proj_head = ProjectionHead(d_model=d_model, d_proj=d_proj, dropout=dropout)

    def forward(self, x: torch.Tensor):
        """
        x: [B, L, D]
        返回:
            Z: [B, L, d_proj]   投影后的潜在特征, 用于对比损失
            H: [B, L, d_model]  编码器输出, 用于依赖模式矩阵计算
        """
        # 实例归一化 (Non-stationary Transformer 风格)
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        E = self.embedding(x)          # [B, L, d_model]
        H = self.encoder(E)            # [B, L, d_model]
        Z = self.proj_head(H)          # [B, L, d_proj]
        return Z, H


class TransformerVar(nn.Module):
    # ours: shrink_thres=0.0025
    def __init__(self, win_size, enc_in, c_out, n_memory, shrink_thres=0, \
                 d_model=512, n_heads=8, e_layers=3, d_ff=512, dropout=0.0, activation='gelu', \
                 device=None, memory_init_embedding=None, memory_initial=False, phase_type=None, dataset_name=None):
        super(TransformerVar, self).__init__()

        self.memory_initial = memory_initial

        # Encoding
        self.embedding = InputEmbedding(in_dim=enc_in, d_model=d_model, dropout=dropout, device=device)   # N x L x C(=d_model)
        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        win_size, d_model, n_heads, dropout=dropout
                    ), d_model, d_ff, dropout=dropout, activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer = nn.LayerNorm(d_model)
        )

        self.mem_module = MemoryModule(n_memory=n_memory, fea_dim=d_model, shrink_thres=shrink_thres, device=device, memory_init_embedding=memory_init_embedding, phase_type=phase_type, dataset_name=dataset_name)
        

        # ours
        self.weak_decoder = Decoder(2 * d_model, c_out, d_ff=d_ff, activation='gelu', dropout=0.1)

        # baselines
        # self.weak_decoder = Decoder(d_model, c_out, d_ff=d_ff, activation='gelu', dropout=0.1)


    def forward(self, x):
        '''
        x (input time window) : N x L x enc_in，N为批量大小，L为时间窗口的长度，enc_in为输入数据的特征维度
        '''
        # 1. 引入标准化 (Normalization)
        use_norm = True
        if use_norm:
            # Normalization from Non-stationary Transformer
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev

        x = self.embedding(x)   # embeddin : N x L x C(=d_model)

        queries = out = self.encoder(x)   # encoder out : N x L x C(=d_model)
        
        outputs = self.mem_module(out)
        out, attn, memory_item_embedding = outputs['output'], outputs['attn'], outputs['memory_init_embedding']

        mem = self.mem_module.mem
        
        if self.memory_initial:
            return {"out":out, "memory_item_embedding":None, "queries":queries, "mem":mem}

        else:
            
            out = self.weak_decoder(out)
            
            '''
            out (reconstructed input time window) : N x L x enc_in
            enc_in == c_out
            '''
            return {"out":out, "memory_item_embedding":memory_item_embedding, "queries":queries, "mem":mem, "attn":attn}


if __name__ == '__main__':
    B, L, D = 4, 100, 12
    d_model, d_proj = 512, 128
    x = torch.randn(B, L, D)

    model = AS2CLAD_Encoder(
        win_size=L, enc_in=D, d_model=d_model, n_heads=8,
        e_layers=3, d_ff=512, d_proj=d_proj, dropout=0.0
    )
    model.eval()
    with torch.no_grad():
        Z, H = model(x)

    print(f"[AS2CLAD_Encoder]")
    print(f"  input  x : {x.shape}")    # [4, 100, 12]
    print(f"  encoder H: {H.shape}")    # [4, 100, 512]
    print(f"  proj    Z: {Z.shape}")    # [4, 100, 128]
