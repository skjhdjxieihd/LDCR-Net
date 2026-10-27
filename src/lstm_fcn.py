import numpy as np
import torch
import torch.nn as nn
from baseline_common import normalized_pdw


class LstmFcnWindowClassifier(nn.Module):
    def __init__(self, input_dim=4, lstm_hidden=64, conv_channels=128, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, lstm_hidden, batch_first=True)
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, conv_channels, kernel_size=8, padding=4),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden + conv_channels, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        lstm_feat = h_n[-1]
        conv_feat = self.conv(x.transpose(1, 2)).squeeze(-1)
        return self.head(torch.cat([lstm_feat, conv_feat], dim=1)).squeeze(-1)


def infer_sequence_probabilities(model, sample, device, window_size, inference_stride):
    sequence = normalized_pdw(sample)
    seq_len = len(sequence)
    probabilities = np.zeros(seq_len, dtype=np.float32)
    counts = np.zeros(seq_len, dtype=np.float32)
    half = window_size // 2
    centers = np.arange(0, seq_len, inference_stride, dtype=np.int64)
    if len(centers) == 0 or centers[-1] != seq_len - 1:
        centers = np.append(centers, seq_len - 1)

    model.eval()
    with torch.no_grad():
        windows = []
        valid_centers = []
        for center in centers:
            start = int(center) - half
            end = start + window_size
            window = np.zeros((window_size, sequence.shape[1]), dtype=np.float32)
            src_start = max(0, start)
            src_end = min(seq_len, end)
            dst_start = src_start - start
            dst_end = dst_start + (src_end - src_start)
            if src_end > src_start:
                window[dst_start:dst_end] = sequence[src_start:src_end]
            windows.append(window)
            valid_centers.append(int(center))

        for start in range(0, len(windows), 256):
            batch = torch.from_numpy(np.stack(windows[start:start + 256])).to(device)
            probs = torch.sigmoid(model(batch)).cpu().numpy()
            for center, prob in zip(valid_centers[start:start + 256], probs):
                probabilities[center] += float(prob)
                counts[center] += 1.0

    mask = counts > 0
    probabilities[mask] /= counts[mask]
    if np.any(mask):
        known = np.flatnonzero(mask)
        probabilities = np.interp(np.arange(seq_len), known, probabilities[known]).astype(np.float32)
    return probabilities
