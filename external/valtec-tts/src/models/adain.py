"""
Adaptive Instance Normalization (AdaIN) for style transfer in TTS.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaIN1d(nn.Module):
    """
    Adaptive Instance Normalization for 1D sequences.
    Applies style-dependent affine transformation after instance normalization.
    """
    
    def __init__(self, style_dim, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

        self.norm = nn.InstanceNorm1d(num_features, eps=eps, affine=False)

        self.gamma_fc = nn.Linear(style_dim, num_features)
        self.beta_fc = nn.Linear(style_dim, num_features)
        
    def forward(self, x, style):
        """
        Apply AdaIN.
        
        Args:
            x: Input features [batch, num_features, time]
            style: Style embedding [batch, style_dim]
            
        Returns:
            Normalized and style-transformed features [batch, num_features, time]
        """
        if style.dim() == 3:
            style = style.squeeze(-1)

        x_normalized = self.norm(x)

        gamma = self.gamma_fc(style).unsqueeze(-1)
        beta = self.beta_fc(style).unsqueeze(-1)

        return (1 + gamma) * x_normalized + beta
