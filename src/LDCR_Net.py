# -*- coding: utf-8 -*-
"""LDCR-Net: Local Dual-window Coherence-Residual representation network.

Designed for low-dimensional 4-D relative-PDW jamming sequences.  The model
uses only local operations: multi-scale CNN, one robust median downsampling,
coherence/residual channel branches, and local change statistics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MedianPool1d(nn.Module):
    def __init__(self, kernel_size=5, stride=2, padding=2):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)

    def forward(self, x):
        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding), mode="reflect")
        windows = x.unfold(2, self.kernel_size, self.stride)
        return windows.median(dim=-1).values


class ConvNormAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, groups=1, dropout=0.0):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, groups=groups, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MultiScaleLocalBlock(nn.Module):
    def __init__(self, input_features=4, hidden_dim=128, dropout=0.0):
        super().__init__()
        self.input_proj = ConvNormAct(input_features, hidden_dim, kernel_size=1, dropout=dropout)
        self.branch1 = ConvNormAct(hidden_dim, hidden_dim, kernel_size=3, dilation=1, dropout=dropout)
        self.branch2 = ConvNormAct(hidden_dim, hidden_dim, kernel_size=3, dilation=2, dropout=dropout)
        self.branch3 = ConvNormAct(hidden_dim, hidden_dim, kernel_size=3, dilation=4, dropout=dropout)
        self.fuse = nn.Sequential(
            nn.Conv1d(hidden_dim * 3, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        base = self.input_proj(x)
        mixed = self.fuse(torch.cat([self.branch1(base), self.branch2(base), self.branch3(base)], dim=1))
        return base + mixed


class SingleScaleLocalBlock(nn.Module):
    """Ablation block: keep parameter interface but remove multi-scale dilation."""

    def __init__(self, input_features=4, hidden_dim=128, dropout=0.0):
        super().__init__()
        self.input_proj = ConvNormAct(input_features, hidden_dim, kernel_size=1, dropout=dropout)
        self.conv = ConvNormAct(hidden_dim, hidden_dim, kernel_size=3, dilation=1, dropout=dropout)

    def forward(self, x):
        base = self.input_proj(x)
        return base + self.conv(base)


class CoherenceResidualBlock(nn.Module):
    """Local joint-channel branch + explicit depthwise residual branch."""

    def __init__(self, input_features=4, hidden_dim=128, dropout=0.0):
        super().__init__()
        self.coherence = nn.Sequential(
            ConvNormAct(hidden_dim, hidden_dim, kernel_size=5, dilation=1, dropout=dropout),
            ConvNormAct(hidden_dim, hidden_dim, kernel_size=3, dilation=2, dropout=dropout),
        )
        self.residual = nn.Sequential(
            # Depthwise local filters on the original four physical variables.
            ConvNormAct(input_features, input_features, kernel_size=5, groups=input_features, dropout=dropout),
            ConvNormAct(input_features, input_features, kernel_size=3, dilation=2, groups=input_features, dropout=dropout),
            nn.Conv1d(input_features, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, h, x_down):
        hc = self.coherence(h)
        hr = self.residual(x_down)
        return h + self.fuse(torch.cat([hc, hr], dim=1))


class LDCRNet(nn.Module):
    def __init__(
        self,
        input_features=4,
        hidden_dim=128,
        z_dim=64,
        tv_dim=32,
        dropout=0.0,
        use_median_downsample=True,
        median_kernel=5,
        median_stride=2,
        use_reconstruction_head=True,
        use_multiscale_local=True,
        use_tv_branch=True,
    ):
        super().__init__()
        self.input_features = input_features
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.tv_dim = tv_dim
        self.use_median_downsample = use_median_downsample
        self.median_kernel = int(median_kernel)
        self.median_stride = int(median_stride) if use_median_downsample else 1
        self.output_stride = self.median_stride
        self.output_offset = 0.0
        self.use_reconstruction_head = use_reconstruction_head
        self.use_multiscale_local = use_multiscale_local
        self.use_tv_branch = use_tv_branch

        local_block = MultiScaleLocalBlock if use_multiscale_local else SingleScaleLocalBlock
        self.multiscale = local_block(input_features, hidden_dim, dropout=dropout)
        if use_median_downsample:
            self.h_down = MedianPool1d(kernel_size=median_kernel, stride=median_stride, padding=median_kernel // 2)
            self.x_down = MedianPool1d(kernel_size=median_kernel, stride=median_stride, padding=median_kernel // 2)
        else:
            self.h_down = nn.Identity()
            self.x_down = nn.Identity()
        self.cr_block = CoherenceResidualBlock(input_features, hidden_dim, dropout=dropout)
        self.ti_head = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, z_dim, kernel_size=1),
        )
        if use_tv_branch:
            self.tv_head = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Conv1d(hidden_dim, tv_dim, kernel_size=1),
            )
        if use_reconstruction_head:
            self.reconstruction_head = nn.Sequential(
                nn.Conv1d(z_dim + tv_dim, hidden_dim, kernel_size=1),
                nn.GELU(),
                nn.Conv1d(hidden_dim, input_features, kernel_size=1),
            )

    @property
    def variant_name(self):
        return "ldcr_unsupervised"

    def output_lengths(self, lengths):
        if not self.use_median_downsample:
            return lengths
        # MedianPool with symmetric padding k//2 and stride s gives ceil-like length.
        return torch.div(lengths + self.median_stride - 1, self.median_stride, rounding_mode="floor")

    def latent_to_input_index(self, index):
        return int(round(index * self.output_stride + self.output_offset))

    def input_to_latent_index(self, index, latent_len):
        pos = int(round((index - self.output_offset) / max(1, self.output_stride)))
        return int(max(0, min(latent_len, pos)))

    def encode_parts(self, x):
        h = self.multiscale(x)
        h_down = self.h_down(h)
        x_down = self.x_down(x)
        h = self.cr_block(h_down, x_down)
        z_ti_raw = self.ti_head(h).transpose(1, 2)
        if self.use_tv_branch:
            z_tv = self.tv_head(h).transpose(1, 2)
        else:
            z_tv = z_ti_raw.new_zeros(z_ti_raw.size(0), z_ti_raw.size(1), self.tv_dim)
        return z_ti_raw, z_tv

    def encode(self, x):
        z_ti_raw, _ = self.encode_parts(x)
        return F.normalize(z_ti_raw, dim=-1)

    def forward(self, x, feature=False):
        z_ti_raw, z_tv = self.encode_parts(x)
        z = F.normalize(z_ti_raw, dim=-1)
        if feature:
            return z
        out = {
            "z": z,
            "z_ti": z,
            "z_ti_raw": z_ti_raw,
            "z_tv": z_tv,
        }
        if self.use_reconstruction_head:
            out["reconstruction"] = self.reconstruct(z_ti_raw, z_tv)
        return out

    def reconstruct(self, z_ti, z_tv):
        z = torch.cat([z_ti, z_tv], dim=-1)
        return self.reconstruction_head(z.transpose(1, 2))

    def dual_window_score(self, z, window=8):
        single = z.dim() == 2
        if z.dim() == 2:
            z = z.unsqueeze(0)
        seq_len = z.size(1)
        if seq_len < 2:
            result = z.new_zeros(z.size(0), seq_len)
            return result[0] if single else result
        if window <= 1 or seq_len < 2 * window + 1:
            score = 1.0 - F.cosine_similarity(z[:, :-1], z[:, 1:], dim=-1)
            score = torch.cat([score[:, :1], score], dim=1)
            return score[0] if single else score
        zt = z.transpose(1, 2)
        left = F.avg_pool1d(F.pad(zt, (window, 0), mode="replicate"), kernel_size=window, stride=1)
        right = F.avg_pool1d(F.pad(zt, (0, window), mode="replicate"), kernel_size=window, stride=1)
        left = left[:, :, :seq_len].transpose(1, 2)
        right = right[:, :, 1 : seq_len + 1].transpose(1, 2)
        score = 1.0 - F.cosine_similarity(left, right, dim=-1)
        return score[0] if single else score

    def recurve_score(self, z, delta=3):
        """Local trajectory turning-angle score inspired by RECURVE."""
        single = z.dim() == 2
        if z.dim() == 2:
            z = z.unsqueeze(0)
        seq_len = z.size(1)
        if seq_len < 2 * delta + 1:
            result = z.new_zeros(z.size(0), seq_len)
            return result[0] if single else result
        before = z[:, delta:-delta] - z[:, :-2 * delta]
        after = z[:, 2 * delta:] - z[:, delta:-delta]
        curvature = 1.0 - F.cosine_similarity(before, after, dim=-1)
        score = F.pad(curvature, (delta, delta), mode="replicate")
        return score[0] if single else score

    def change_score(self, z, window=8, curvature_delta=3, curvature_weight=0.2):
        d = self.dual_window_score(z, window=window)
        if curvature_weight <= 0:
            return d
        c = self.recurve_score(z, delta=curvature_delta)
        if c.numel() != d.numel():
            return d
        return (1.0 - curvature_weight) * d + curvature_weight * c

    def segment_embeddings(self, z, boundaries, trim=0):
        if z.dim() == 3:
            z = z.squeeze(0)
        values = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            latent_start = self.input_to_latent_index(start, z.size(0))
            latent_end = self.input_to_latent_index(end, z.size(0))
            latent_start = min(z.size(0), max(0, latent_start + trim))
            latent_end = min(z.size(0), max(0, latent_end - trim))
            if latent_end <= latent_start:
                continue
            part = z[latent_start:latent_end]
            values.append(torch.cat([part.mean(dim=0), part.std(dim=0, unbiased=False)], dim=0))
        if not values:
            return z.new_empty(0, self.z_dim * 2)
        return torch.stack(values, dim=0)
