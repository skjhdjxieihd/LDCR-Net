# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Upstream-derived portions: see THIRD_PARTY_NOTICES.md and LICENSES/UnsupSeg-MIT.txt.
"""
Simplified core model for Contrastive Predictive Coding (CPC).
This file contains the main model architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

class LambdaLayer(nn.Module):
    """
    A helper layer to apply a lambda function in an nn.Sequential model.
    """
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

class MedianPool1d(nn.Module):
    """
    Median pooling layer for 1D inputs (N, C, L).
    Robust to impulse noise.
    """
    def __init__(self, kernel_size, stride=None, padding=0):
        super(MedianPool1d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x):
        # Manually pad the input
        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding), mode='reflect')
        
        # Use unfold to create sliding windows
        windows = x.unfold(2, self.kernel_size, self.stride)
        # windows shape: (N, C, L_out, kernel_size)
        
        # Take the median over the kernel_size dimension
        median, _ = torch.median(windows, dim=3)
        return median

class NextFrameClassifier(nn.Module):
    """
    The core Contrastive Predictive Coding (CPC) model.
    It uses a convolutional or RNN encoder to extract latent representations
    and then uses a contrastive loss to predict future latents.
    This version uses Median Pooling for robustness to noise.
    """
    def __init__(self, hp, embedding=False, rnn=False, kernel_size=None):
        super(NextFrameClassifier, self).__init__()
        self.hp = hp
        self.embedding = embedding
        self.rnn = rnn
        
        # Use kernel_size from args or default
        kernel_size = kernel_size if kernel_size is not None else [2, 2, 2, 1, 1]
        # Get strides from hp config, with a default
        strides = getattr(hp, 'strides', [2, 2, 1, 1, 1])
        # Get pooling kernel size from hp config, with a default
        pool_kernels = getattr(hp, 'pool_kernels', [5, 5])
        # Check whether to use median pooling for downsampling
        use_median_pool = getattr(hp, 'use_median_pool', True)


        Z_DIM = hp.z_dim
        LS = hp.latent_dim
        input_features = hp.input_features
        
        if embedding:
            self.emb = nn.Embedding(10, input_features)

        if rnn:
            self.enc = nn.Sequential(
                nn.GRU(input_size=input_features, hidden_size=int(Z_DIM/2), num_layers=1, bidirectional=True, batch_first=True),
                LambdaLayer(lambda x: x[0]),
            )
        else:
            # Dynamically build the encoder based on whether to use median pooling
            layers = []
            in_channels = input_features
            num_conv_layers = len(kernel_size)

            for i in range(num_conv_layers):
                out_channels = LS if i < num_conv_layers - 1 else Z_DIM
                
                # Determine stride for the Conv1d layer
                conv_stride = strides[i] if not use_median_pool and i < len(strides) else 1
                
                layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size[i], stride=conv_stride, padding=0, bias=False))
                
                if i < num_conv_layers - 1:
                    layers.append(nn.BatchNorm1d(out_channels))
                    layers.append(nn.LeakyReLU())

                # Add Median Pooling layer if enabled and required
                if use_median_pool and i < len(strides) and strides[i] > 1:
                    layers.append(MedianPool1d(kernel_size=pool_kernels[i], stride=strides[i]))
                
                in_channels = out_channels

            layers.append(LambdaLayer(lambda x: x.transpose(1,2)))
            self.enc = nn.Sequential(*layers)

        self.enc.add_module(
            "z_proj",
            nn.Sequential(
                nn.Linear(Z_DIM, self.hp.z_proj)
            )
        )
                
        self.pred_steps = [1]

    def score(self, f, b):
        """Calculates similarity score between two tensors."""
        return F.cosine_similarity(f, b, dim=-1)
    
    def forward(self, spect, feature=False):
        """
        Forward pass of the model.
        Encodes the input, and if not in feature extraction mode,
        calculates scores for positive and negative samples.
        """
        if self.embedding:
            spect = self.emb(spect)
            if not self.rnn:
                spect = spect.permute(0, 2, 1)
        
        z = self.enc(spect)
        
        if feature:
            return z

        preds = defaultdict(list)
        for i, t in enumerate(self.pred_steps):
            pos_pred = self.score(z[:, :-t], z[:, t:])
            preds[t].append(pos_pred)

            for _ in range(self.hp.n_negatives):
                if self.training:
                    time_reorder = torch.randperm(pos_pred.shape[1])
                    batch_reorder = torch.arange(pos_pred.shape[0])
                    if self.hp.batch_shuffle:
                        batch_reorder = torch.randperm(pos_pred.shape[0])
                else: # For validation/testing, don't shuffle
                    time_reorder = torch.arange(pos_pred.shape[1])
                    batch_reorder = torch.arange(pos_pred.shape[0])
                    
                neg_pred = self.score(z[:, :-t], z[batch_reorder][:, time_reorder])
                preds[t].append(neg_pred)
        
        return preds

    def loss(self, preds):
        """
        Calculates the InfoNCE loss.
        """
        loss = 0
        for t, t_preds in preds.items():
            out = torch.stack(t_preds, dim=-1)
            out = F.log_softmax(out, dim=-1)
            # We want to maximize the score of the positive sample (index 0)
            out = out[..., 0]
            loss += -out.mean()
        return loss
