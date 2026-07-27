"""
Zero-shot TTS Synthesizer with AdaIN (Adaptive Instance Normalization).
Supports voice cloning from reference audio.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm

from src.nn.commons import init_weights, get_padding
from src import alignment as monotonic_align
from src.nn import commons
from src.models.adain import AdaIN1d
from src.models.synthesizer import (
    DurationDiscriminator,
    TransformerCouplingBlock,
    StochasticDurationPredictor,
    DurationPredictor,
    TextEncoder,
    ResidualCouplingBlock,
    PosteriorEncoder,
)

LRELU_SLOPE = 0.1


class FiLM(nn.Module):
    """Feature-wise Linear Modulation for F0/energy conditioning."""
    
    def __init__(self, cond_dim, channels):
        super().__init__()
        self.conv_gamma = nn.Conv1d(cond_dim, channels, 1)
        self.conv_beta = nn.Conv1d(cond_dim, channels, 1)
        self.conv_gamma.apply(init_weights)
        self.conv_beta.apply(init_weights)

    def forward(self, x, cond):
        if cond.shape[-1] != x.shape[-1]:
            cond = F.interpolate(cond, size=x.shape[-1], mode='linear')
        
        gamma = self.conv_gamma(cond)
        beta = self.conv_beta(cond)
        
        return x * (gamma + 1) + beta


class ResBlock1AdaIN(nn.Module):
    """Residual block with AdaIN for prosody conditioning."""
    
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), prosody_dim=128):
        super().__init__()
        self.channels = channels
        self.prosody_dim = prosody_dim
        
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1,
                              dilation=dilation[i],
                              padding=get_padding(kernel_size, dilation[i])))
            for i in range(len(dilation))
        ])
        self.convs1.apply(init_weights)
        
        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1,
                              dilation=1,
                              padding=get_padding(kernel_size, 1)))
            for _ in range(len(dilation))
        ])
        self.convs2.apply(init_weights)
        
        self.adains = nn.ModuleList([
            AdaIN1d(prosody_dim, channels)
            for _ in range(len(dilation))
        ])
        
        self.films = nn.ModuleList([
            FiLM(2, channels)
            for _ in range(len(dilation))
        ])
    
    def forward(self, x, x_mask=None, prosody=None, f0=None, energy=None):
        cond = None
        if f0 is not None and energy is not None:
            if f0.dim() == 2: 
                f0 = f0.unsqueeze(1)
            if energy.dim() == 2: 
                energy = energy.unsqueeze(1)
            cond = torch.cat([f0, energy], dim=1)
        
        for i, (c1, c2) in enumerate(zip(self.convs1, self.convs2)):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)
            
            if prosody is not None:
                xt = self.adains[i](xt, prosody)
            
            if cond is not None:
                xt = self.films[i](xt, cond)
            
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c2(xt)
            x = xt + x
        
        if x_mask is not None:
            x = x * x_mask
        return x
    
    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class GeneratorAdaIN(nn.Module):
    """Generator with AdaIN for zero-shot voice cloning."""
    
    def __init__(
        self,
        initial_channel,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels=0,
        prosody_dim=128,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.prosody_dim = prosody_dim
        self.gin_channels = gin_channels
        
        self.conv_pre = Conv1d(
            initial_channel + gin_channels, upsample_initial_channel, 7, 1, padding=3
        )
        
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k, u,
                        padding=(k - u) // 2,
                    )
                )
            )
        
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(
                    ResBlock1AdaIN(ch, k, d, prosody_dim=prosody_dim)
                )
        
        ch = upsample_initial_channel // (2 ** len(self.ups))
        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)
        
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)
    
    def forward(self, x, g=None, prosody=None, f0=None, energy=None):
        # Handle F0
        if f0 is not None:
            if f0.dim() == 2: 
                f0 = f0.unsqueeze(1)
            if f0.shape[-1] != x.shape[-1]:
                f0 = F.interpolate(f0, size=x.shape[-1], mode='linear')
        else:
            f0 = torch.zeros(x.shape[0], 1, x.shape[-1], device=x.device, dtype=x.dtype)
        
        # Handle energy
        if energy is not None:
            if energy.dim() == 2: 
                energy = energy.unsqueeze(1)
            if energy.shape[-1] != x.shape[-1]:
                energy = F.interpolate(energy, size=x.shape[-1], mode='linear')
        else:
            energy = torch.zeros(x.shape[0], 1, x.shape[-1], device=x.device, dtype=x.dtype)
        
        # Handle speaker embedding
        if g is not None:
            g_expanded = g.expand(-1, -1, x.shape[-1])
        else:
            g_expanded = torch.zeros(x.shape[0], self.gin_channels, x.shape[-1], 
                                    device=x.device, dtype=x.dtype)
        
        x = torch.cat([x, g_expanded], dim=1)
        x = self.conv_pre(x)
        
        if g is not None:
            x = x + self.cond(g)
        
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](
                        x, prosody=prosody, f0=f0, energy=energy
                    )
                else:
                    xs += self.resblocks[i * self.num_kernels + j](
                        x, prosody=prosody, f0=f0, energy=energy
                    )
            x = xs / self.num_kernels
        
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        
        return x
    
    def remove_weight_norm(self):
        print("Removing weight norm...")
        for layer in self.ups:
            remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()


class SynthesizerZeroShot(nn.Module):
    """
    Zero-shot TTS Synthesizer.
    Supports voice cloning from reference audio via speaker and prosody embeddings.
    """
    
    def __init__(
        self,
        n_vocab,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        n_speakers=0,
        gin_channels=512,
        prosody_dim=128,
        use_sdp=True,
        num_languages=None,
        num_tones=None,
        use_transformer_flow=True,
        transformer_flow_heads=2,
        flow_share_parameter=False,
        **kwargs
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.prosody_dim = prosody_dim
        self.use_sdp = use_sdp
        self.use_transformer_flow = use_transformer_flow
        
        self.enc_p = TextEncoder(
            n_vocab,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            gin_channels=gin_channels,
            num_languages=num_languages,
            num_tones=num_tones,
        )
        
        self.dec = GeneratorAdaIN(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
            prosody_dim=prosody_dim,
        )
        
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5, 1, 16,
            gin_channels=gin_channels,
        )
        
        if use_transformer_flow:
            self.flow = TransformerCouplingBlock(
                inter_channels,
                hidden_channels,
                filter_channels,
                transformer_flow_heads,
                3, 5, p_dropout,
                n_flows=4,
                gin_channels=gin_channels,
                share_parameter=flow_share_parameter
            )
        else:
            self.flow = ResidualCouplingBlock(
                inter_channels,
                hidden_channels,
                5, 1, 4,
                n_flows=4,
                gin_channels=gin_channels,
            )
        
        if use_sdp:
            self.dp = StochasticDurationPredictor(
                hidden_channels, 192, 3, 0.5, 4,
                gin_channels=gin_channels
            )
        else:
            self.dp = DurationPredictor(
                hidden_channels, 256, 3, 0.5,
                gin_channels=gin_channels
            )
        
        if n_speakers > 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)
        
        self.cond_f0_energy = nn.Conv1d(2, gin_channels, 1)
        nn.init.normal_(self.cond_f0_energy.weight, 0.0, 1.0)
        nn.init.zeros_(self.cond_f0_energy.bias)
    
    def forward(
        self, x, x_lengths, y, y_lengths,
        sid, tone, language, bert, ja_bert,
        g=None, prosody=None, f0=None, energy=None
    ):
        if self.n_speakers > 0 and sid is not None:
            g = self.emb_g(sid).unsqueeze(-1)
        
        x, m_p, logs_p, x_mask = self.enc_p(
            x, x_lengths, tone, language, bert, ja_bert, g=g
        )
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        
        g_flow = g
        if f0 is not None and energy is not None:
            if f0.dim() == 2: 
                f0 = f0.unsqueeze(1)
            if energy.dim() == 2: 
                energy = energy.unsqueeze(1)
            
            if f0.shape[-1] != z.shape[-1]:
                f0 = F.interpolate(f0, size=z.shape[-1], mode='linear')
            if energy.shape[-1] != z.shape[-1]:
                energy = F.interpolate(energy, size=z.shape[-1], mode='linear')
            
            cond = torch.cat([f0, energy], dim=1)
            g_flow = g + self.cond_f0_energy(cond)
        
        z_p = self.flow(z, y_mask, g=g_flow)
        
        with torch.no_grad():
            s_p_sq_r = torch.exp(-2 * logs_p)
            neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True)
            neg_cent2 = torch.matmul(-0.5 * (z_p**2).transpose(1, 2), s_p_sq_r)
            neg_cent3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r))
            neg_cent4 = torch.sum(-0.5 * (m_p**2) * s_p_sq_r, [1], keepdim=True)
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4
            
            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()
        
        w = attn.sum(2)
        if self.use_sdp:
            l_length = self.dp(x, x_mask, w, g=g)
            l_length = l_length / torch.sum(x_mask)
            logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=1.0)
            logw_ = torch.log(w + 1e-6) * x_mask
        else:
            logw_ = torch.log(w + 1e-6) * x_mask
            logw = self.dp(x, x_mask, g=g)
            l_length = torch.sum((logw - logw_) ** 2, [1, 2]) / torch.sum(x_mask)
        
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)
        
        z_slice, ids_slice = commons.rand_slice_segments(z, y_lengths, self.segment_size)
        
        f0_slice = None
        if f0 is not None:
            if f0.dim() == 2: 
                f0 = f0.unsqueeze(1)
            f0_slice = commons.slice_segments(f0, ids_slice, self.segment_size).squeeze(1)
        
        energy_slice = None
        if energy is not None:
            if energy.dim() == 2: 
                energy = energy.unsqueeze(1)
            energy_slice = commons.slice_segments(energy, ids_slice, self.segment_size).squeeze(1)
        
        o = self.dec(z_slice, g=g, prosody=prosody, f0=f0_slice, energy=energy_slice)
        
        return o, l_length, attn, ids_slice, x_mask, y_mask, \
               (z, z_p, m_p, logs_p, m_q, logs_q), \
               (x, logw, logw_)
    
    def infer(
        self, x, x_lengths, sid, tone, language, bert, ja_bert,
        g=None, prosody=None, f0=None, energy=None, prosody_predictor=None,
        noise_scale=0.667, length_scale=1, noise_scale_w=0.8,
        max_len=None, sdp_ratio=0.0
    ):
        if self.n_speakers > 0 and sid is not None:
            g = self.emb_g(sid).unsqueeze(-1)
        
        x, m_p, logs_p, x_mask = self.enc_p(
            x, x_lengths, tone, language, bert, ja_bert, g=g
        )
        
        if self.use_sdp:
            logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
        else:
            logw = self.dp(x, x_mask, g=g)
        
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)
        
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)
        
        if f0 is None and prosody_predictor is not None and prosody is not None:
            f0, energy = prosody_predictor(m_p, prosody)
        
        g_flow = g
        if f0 is not None and energy is not None:
            if f0.dim() == 2: 
                f0 = f0.unsqueeze(1)
            if energy.dim() == 2: 
                energy = energy.unsqueeze(1)
            
            if f0.shape[-1] != m_p.shape[-1]:
                f0 = F.interpolate(f0, size=m_p.shape[-1], mode='linear')
            if energy.shape[-1] != m_p.shape[-1]:
                energy = F.interpolate(energy, size=m_p.shape[-1], mode='linear')
            
            cond = torch.cat([f0, energy], dim=1)
            g_flow = g + self.cond_f0_energy(cond)
        
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g_flow, reverse=True)
        
        o = self.dec((z * y_mask)[:, :, :max_len], g=g, prosody=prosody, f0=f0, energy=energy)
        
        return o, attn, y_mask, (z, z_p, m_p, logs_p)
