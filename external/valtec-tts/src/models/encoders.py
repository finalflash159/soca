"""
Encoder modules for zero-shot voice cloning.
Includes H/ASP Speaker Encoder, Style Encoder (AdaIN), and Prosody Predictor.
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.nn.utils import weight_norm

# ============================================================================
# Style Encoder (originally from adain.py)
# ============================================================================

class LearnedDownSample(nn.Module):
    def __init__(self, layer_type, dim_in):
        super().__init__()
        self.layer_type = layer_type

        if self.layer_type == 'none':
            self.conv = nn.Identity()
        elif self.layer_type == 'timepreserve':
            self.conv = nn.Conv2d(dim_in, dim_in, kernel_size=(3, 1), stride=(2, 1), groups=dim_in, padding=(1, 0))
        elif self.layer_type == 'half':
            self.conv = nn.Conv2d(dim_in, dim_in, kernel_size=(3, 3), stride=(2, 2), groups=dim_in, padding=1)
        else:
            raise RuntimeError('Got unexpected donwsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)

    def forward(self, x):
        return self.conv(x)

class DownSample(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        elif self.layer_type == 'timepreserve':
            return F.avg_pool2d(x, (2, 1))
        elif self.layer_type == 'half':
            if x.shape[-1] % 2 != 0:
                x = torch.cat([x, x[..., -1].unsqueeze(-1)], dim=-1)
            return F.avg_pool2d(x, 2)
        else:
            raise RuntimeError('Got unexpected donwsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)

class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample='none'):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = DownSample(downsample)
        self.downsample_res = LearnedDownSample(downsample, dim_in)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = self.downsample(x)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        x = self.downsample_res(x)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)

class StyleEncoder(nn.Module):
    def __init__(self, n_mel_channels=80, dim_in=48, style_dim=128, max_conv_dim=384):
        super().__init__()

        self.dim_in = dim_in
        blocks = []
        blocks += [nn.Conv2d(1, dim_in, 3, 1, 1)]

        repeat_num = 4
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample='half')]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 5, 1, 0)]
        blocks += [nn.AdaptiveAvgPool2d(1)]
        self.shared = nn.Sequential(*blocks)

        self.unshared = nn.Linear(dim_out, style_dim)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        s = self.unshared(h)
        return s


# ============================================================================
# Speaker Encoder (H/ASP)
# ============================================================================

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = x.view(b, c, -1).mean(dim=2)
        y = self.fc(y)
        y = y.view(b, c, 1, 1)
        return x * y

class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=8):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.se = SEBlock(planes, reduction)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNetSE34V2(nn.Module):
    def __init__(self, nOut=512, encoder_type='ASP', n_mels=64, **kwargs):
        super(ResNetSE34V2, self).__init__()
        self.inplanes = 32
        self.encoder_type = encoder_type
        self.n_mels = n_mels

        block = SEBasicBlock
        layers = [3, 4, 6, 3]
        num_filters = [32, 64, 128, 256]

        self.conv1 = nn.Conv2d(1, num_filters[0], kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, num_filters[0], layers[0])
        self.layer2 = self._make_layer(block, num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(block, num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self._make_layer(block, num_filters[3], layers[3], stride=(2, 2))

        outmap_size = int(self.n_mels / 8)
        self.outmap_size = outmap_size

        attention_dim = num_filters[3] * outmap_size
        self.attention = nn.Sequential(
            nn.Conv1d(attention_dim, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, attention_dim, kernel_size=1),
            nn.Softmax(dim=2),
        )

        if self.encoder_type == "SAP":
            out_dim = attention_dim
        elif self.encoder_type == "ASP":
            out_dim = attention_dim * 2
        else:
            raise ValueError(f'Unknown encoder type: {encoder_type}')

        self.fc = nn.Linear(out_dim, nOut)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.reshape(x.size(0), -1, x.size(-1))
        w = self.attention(x)
        if self.encoder_type == "SAP":
            x = torch.sum(x * w, dim=2)
        elif self.encoder_type == "ASP":
            mu = torch.sum(x * w, dim=2)
            sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-5))
            x = torch.cat((mu, sg), dim=1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class SpeakerEncoder(nn.Module):
    def __init__(self, device='cpu', embed_dim=512):
        super(SpeakerEncoder, self).__init__()
        self.device = device
        self.target_embed_dim = embed_dim
        self.native_embed_dim = 512
        self.sample_rate = 16000
        self.n_mels = 64

        self.model = ResNetSE34V2(nOut=self.native_embed_dim, encoder_type='ASP', n_mels=self.n_mels)

        if self.target_embed_dim != self.native_embed_dim:
            self.projection = nn.Linear(self.native_embed_dim, self.target_embed_dim)
            nn.init.xavier_uniform_(self.projection.weight)
            nn.init.zeros_(self.projection.bias)
        else:
            self.projection = None

        self._load_pretrained()

        self.model = self.model.to(device)
        if self.projection is not None:
            self.projection = self.projection.to(device)

        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=512,
            win_length=400,
            hop_length=160,
            n_mels=self.n_mels,
            power=2.0,
        ).to(device)

        self.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        if self.projection is not None:
            for param in self.projection.parameters():
                param.requires_grad = False

    def _load_pretrained(self):
        # Local cache directory within the project
        project_root = os.getcwd()
        cache_dir = os.path.join(project_root, "pretrained", "hasp")
        os.makedirs(cache_dir, exist_ok=True)
        model_path = os.path.join(cache_dir, "pytorch_model.bin")

        if not os.path.exists(model_path):
            print(f"[SpeakerEncoder] Downloading pretrained model to {cache_dir}...")
            try:
                from huggingface_hub import hf_hub_download
                downloaded_path = hf_hub_download(
                    repo_id="Edresson/Speaker_Encoder_H_ASP",
                    filename="pytorch_model.bin",
                    cache_dir=cache_dir,
                    local_dir=cache_dir,
                )
                model_path = downloaded_path
            except Exception as e:
                print(f"[SpeakerEncoder] Warning: Could not download: {e}")
                print("[SpeakerEncoder] Using random weights (NOT RECOMMENDED)")
                return

        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict, strict=False)
            print("[SpeakerEncoder] ✓ Loaded pretrained weights")
        except Exception as e:
            print(f"[SpeakerEncoder] Warning: Could not load weights: {e}")

    def compute_mel(self, audio, sr=None):
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        if audio.dim() == 3:
            audio = audio.squeeze(1)

        if sr is not None and sr != self.sample_rate:
            key = (int(sr), str(audio.device))
            if not hasattr(self, '_resamplers'):
                self._resamplers = {}
            if key not in self._resamplers:
                self._resamplers[key] = T.Resample(sr, self.sample_rate).to(audio.device)
            audio = self._resamplers[key](audio)

        mel = self.mel_transform(audio)
        mel = torch.log(mel.clamp(min=1e-5))
        mel = mel - mel.mean(dim=-1, keepdim=True)
        return mel

    def forward(self, audio, sr=None):
        mel = self.compute_mel(audio, sr)
        with torch.no_grad():
            emb = self.model(mel)
            if self.projection is not None:
                emb = self.projection(emb)
            emb = F.normalize(emb, p=2, dim=-1)
        return emb


# ============================================================================
# Prosody Predictor
# ============================================================================

class UpSample1d(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        else:
            return F.interpolate(x, scale_factor=2, mode='nearest')

class AdaIN1dPred(nn.Module):
    def __init__(self, style_dim, num_features, eps=1e-6):
        super().__init__()
        self.num_features = num_features
        self.norm = nn.InstanceNorm1d(num_features, eps=eps, affine=False)
        self.gamma_fc = nn.Linear(style_dim, num_features)
        self.beta_fc = nn.Linear(style_dim, num_features)

    def forward(self, x, s):
        if s.dim() == 3:
            s = s.squeeze(-1)
        x = self.norm(x)
        gamma = self.gamma_fc(s).unsqueeze(-1)
        beta = self.beta_fc(s).unsqueeze(-1)
        return (1 + gamma) * x + beta

class AdainResBlk1d(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, actv=nn.LeakyReLU(0.2),
                 upsample='none', dropout_p=0.0):
        super().__init__()
        self.actv = actv
        self.upsample_type = upsample
        self.upsample = UpSample1d(upsample)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)
        self.dropout = nn.Dropout(dropout_p)

        if upsample == 'none':
            self.pool = nn.Identity()
        else:
            self.pool = weight_norm(nn.ConvTranspose1d(
                dim_in, dim_in, kernel_size=3, stride=2, 
                groups=dim_in, padding=1, output_padding=1))

    def _build_weights(self, dim_in, dim_out, style_dim):
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_out, dim_out, 3, 1, 1))
        self.norm1 = AdaIN1dPred(style_dim, dim_in)
        self.norm2 = AdaIN1dPred(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False))

    def _shortcut(self, x):
        x = self.upsample(x)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.pool(x)
        x = self.conv1(self.dropout(x))
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(self.dropout(x))
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        out = (out + self._shortcut(x)) / math.sqrt(2)
        return out

class ProsodyPredictor(nn.Module):
    def __init__(self, style_dim=128, d_hid=256, text_dim=192, dropout=0.1):
        super().__init__()

        self.text_proj = nn.Conv1d(text_dim, d_hid, 1)

        self.shared = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, 
                              batch_first=True, bidirectional=True)

        self.F0 = nn.ModuleList()
        self.F0.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout))
        self.F0.append(AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample='half', dropout_p=dropout))
        self.F0.append(AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout))

        self.N = nn.ModuleList()
        self.N.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout))
        self.N.append(AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample='half', dropout_p=dropout))
        self.N.append(AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout))

        self.F0_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)
        self.N_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)

    def forward(self, x, s, input_lengths=None):
        if x.dim() == 2:
             # If just [B, D], unsqueeze dim 2? No, error said [B, C, T]
             raise ValueError("ProsodyPredictor expects text features [B, C, T], got [B, D].")

        # Project text features
        x = self.text_proj(x)
        x = x.transpose(1, 2)  # [B, T, d_hid]

        # Expand style
        s_expanded = s.unsqueeze(1).expand(-1, x.shape[1], -1)  # [B, T, style_dim]
        
        # Concat text + style
        x_s = torch.cat([x, s_expanded], dim=-1)

        self.shared.flatten_parameters()
        x_shared, _ = self.shared(x_s)  # [B, T, d_hid]

        F0 = x_shared.transpose(1, 2)  # [B, d_hid, T]
        N = x_shared.transpose(1, 2)

        for block in self.F0:
            F0 = block(F0, s)
        F0_out = self.F0_proj(F0)

        for block in self.N:
            N = block(N, s)
        N_out = self.N_proj(N)

        return F0_out.squeeze(1), N_out.squeeze(1)
