import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
class FreRA(Module):
    # FreRA: Adaptive Spectral Saliency Augmentation 自适应频域增强模块
    def __init__(self, len_sw, device=None, dtype=None, alpha_limit=0.2, noise_std=0.5) -> None:
        super(FreRA,self).__init__()
        print('Initializing ASSA (Adaptive Spectral Saliency Augmentation)')
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.len_sw = len_sw
        # Frequency components count for rfft
        self.n_freq = len_sw // 2 + 1
        
        # Global learnable parameter vector W for saliency
        self.saliency_weight = Parameter(torch.empty(self.n_freq, **factory_kwargs))
        self.reset_parameters()
        
        self.alpha_limit = alpha_limit
        self.noise_std = noise_std
        self.mask = None

    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.saliency_weight, mean=0.0, std=0.1)

    def forward(self, x, temperature=None):
        # x: [Batch, Length, Dim]
        
        # 1. FFT
        x_ft = torch.fft.rfft(x, dim=-2)
        
        # 2. Generate Saliency Mask M
        # M = Sigmoid(W)
        M = torch.sigmoid(self.saliency_weight)
        self.mask = M # Save for regularization
        
        # Expand M for broadcasting: [Freq] -> [1, Freq, 1]
        M_expanded = M.view(1, -1, 1)
        
        if self.training:
            # 3. Augmentation
            
            # 3.1 Saliency Scaling
            # alpha ~ U(-alpha_limit, alpha_limit)
            alpha = (torch.rand(1, device=x.device) * 2 - 1) * self.alpha_limit
            
            # 3.2 Background Perturbation
            # Complex Gaussian Noise epsilon
            # Real and Imag parts ~ N(0, 1) * noise_std
            noise_real = torch.randn_like(x_ft.real) * self.noise_std
            noise_imag = torch.randn_like(x_ft.imag) * self.noise_std
            epsilon = torch.complex(noise_real, noise_imag)
            
            # 3.3 Apply Formula
            # F' = F * (1 + alpha * M) + epsilon * (1 - M)
            x_ft_aug = x_ft * (1 + alpha * M_expanded) + epsilon * (1 - M_expanded)
            
            # 4. iFFT
            x_aug = torch.fft.irfft(x_ft_aug, n=self.len_sw, dim=-2)
            return x_aug
        else:
            return x