"""Published LDCR model configuration, objective, dataset and checkpoint functions."""
import os,random
from dataclasses import dataclass,asdict
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from LDCR_Net import LDCRNet


@dataclass
class ModelConfig:
    input_features: int = 4
    hidden_dim: int = 128
    z_dim: int = 64
    tv_dim: int = 32
    dropout: float = 0.0
    median_kernel: int = 5
    median_stride: int = 2
    use_median_downsample: bool = True
    use_multiscale_local: bool = True
    use_tv_branch: bool = True
    use_reconstruction_head: bool = True


@dataclass
class LossConfig:
    invariance_weight: float = 25.0
    variance_weight: float = 25.0
    covariance_weight: float = 1.0
    ti_weight: float = 5.0
    stable_consistency_weight: float = 0.5
    stable_consistency_steps: int = 2
    reconstruction_weight: float = 0.2
    separation_weight: float = 1.0
    input_sigma_scale: float = 1.0
    variance_target: float = 1.0
    max_consistency_points: int = 1024
    noise_std: float = 0.01
    feature_drop_prob: float = 0.05


class PdwDataset(Dataset):
    def __init__(self, path):
        with np.load(path, allow_pickle=True) as loaded:
            self.samples = loaded["data"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        return {
            "index": index,
            "pdw": torch.as_tensor(sample["pdw_data"], dtype=torch.float32),
            "change_points": np.asarray(sample.get("change_points", []), dtype=np.int64),
        }


def collate_batch(batch):
    lengths = torch.tensor([len(x["pdw"]) for x in batch], dtype=torch.long)
    values = torch.zeros(len(batch), int(lengths.max()), 4)
    for i, item in enumerate(batch):
        values[i, : lengths[i]] = item["pdw"]
    return {"pdw": values.transpose(1, 2), "lengths": lengths}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def length_mask(lengths, max_len, device):
    return torch.arange(max_len, device=device)[None] < lengths.to(device)[:, None]


def augment(x, mask, config):
    value = x + config.noise_std * torch.randn_like(x)
    if config.feature_drop_prob:
        keep = torch.rand(x.size(0), x.size(1), 1, device=x.device) >= config.feature_drop_prob
        value = value * keep
    return value * mask[:, None]


def sample_valid_rows(z1, z2, valid, max_points):
    rows1, rows2 = z1[valid], z2[valid]
    if len(rows1) > max_points:
        index = torch.randperm(len(rows1), device=z1.device)[:max_points]
        rows1, rows2 = rows1[index], rows2[index]
    return rows1, rows2


def off_diagonal(matrix):
    size = matrix.size(0)
    return matrix.flatten()[:-1].view(size - 1, size + 1)[:, 1:].flatten()


def negative_free_consistency_loss(z1, z2, valid, config):
    """VICReg-style consistency: no negative samples and therefore no false negatives."""
    a, b = sample_valid_rows(z1, z2, valid, config.max_consistency_points)
    if len(a) < 2:
        zero = z1.new_tensor(0.0)
        return zero, zero, zero
    invariance = F.mse_loss(a, b)
    std_a = torch.sqrt(a.var(dim=0, unbiased=False) + 1e-4)
    std_b = torch.sqrt(b.var(dim=0, unbiased=False) + 1e-4)
    variance = 0.5 * (
        F.relu(config.variance_target - std_a).mean()
        + F.relu(config.variance_target - std_b).mean()
    )
    a = a - a.mean(dim=0)
    b = b - b.mean(dim=0)
    cov_a = a.T @ a / max(1, len(a) - 1)
    cov_b = b.T @ b / max(1, len(b) - 1)
    covariance = (
        off_diagonal(cov_a).square().sum()
        + off_diagonal(cov_b).square().sum()
    ) / (2 * a.size(1))
    return invariance, variance, covariance


def ti_loss(z, x_down, valid, sigma_scale):
    """Input-adaptive time-invariance: smooth stable regions, relax at real changes."""
    if z.size(1) < 2:
        return z.new_tensor(0.0)
    dz = F.smooth_l1_loss(z[:, 1:], z[:, :-1], reduction="none").mean(-1)
    active = valid[:, 1:] & valid[:, :-1]
    if not active.any():
        return z.new_tensor(0.0)
    raw_diff = x_down[:, :, 1:] - x_down[:, :, :-1]
    # Normalize each physical variable by its own batch difference scale.
    # Otherwise DTOA/RF can dominate PW-ratio/PA solely because of units.
    channel_scale = []
    for channel in range(raw_diff.size(1)):
        values = raw_diff[:, channel][active].detach().abs()
        channel_scale.append(values.median().clamp_min(1e-4))
    channel_scale = torch.stack(channel_scale)[None, :, None]
    dx = (raw_diff / channel_scale).square().mean(1)
    sigma = dx[active].detach().median().clamp_min(1e-4) * sigma_scale
    weight = torch.exp(-dx / sigma.clamp_min(1e-4))
    return (dz[active] * weight[active]).mean() if active.any() else z.new_tensor(0.0)


def stable_region_consistency_loss(z, x_down, valid, max_steps, sigma_scale):
    """Multi-step smoothing only where every intervening input transition is stable."""
    if z.size(1) < 2 or max_steps < 1:
        return z.new_tensor(0.0)
    adjacent_valid = valid[:, 1:] & valid[:, :-1]
    if not adjacent_valid.any():
        return z.new_tensor(0.0)
    raw_diff = x_down[:, :, 1:] - x_down[:, :, :-1]
    scales = []
    for channel in range(raw_diff.size(1)):
        values = raw_diff[:, channel][adjacent_valid].detach().abs()
        scales.append(values.median().clamp_min(1e-4))
    scales = torch.stack(scales)[None, :, None]
    input_change = (raw_diff / scales).square().mean(1)
    sigma = input_change[adjacent_valid].detach().median().clamp_min(1e-4)
    adjacent_stability = torch.exp(
        -input_change / (sigma * sigma_scale).clamp_min(1e-4)
    )

    z_normalized = F.normalize(z, dim=-1)
    total, used = z.new_tensor(0.0), 0
    for step in range(1, min(max_steps, z.size(1) - 1) + 1):
        pair_valid = valid[:, :-step] & valid[:, step:]
        # A pair is stable only if all transitions between its endpoints are stable.
        interval_stability = torch.stack(
            [
                adjacent_stability[:, offset : offset + z.size(1) - step]
                for offset in range(step)
            ],
            dim=0,
        ).amin(dim=0)
        distance = 1.0 - F.cosine_similarity(
            z_normalized[:, :-step],
            z_normalized[:, step:],
            dim=-1,
        )
        if pair_valid.any():
            weights = interval_stability[pair_valid]
            total = total + (
                distance[pair_valid] * weights
            ).sum() / weights.sum().clamp_min(1e-6)
            used += 1
    return total / max(1, used)


def reconstruction_loss(reconstruction, target, valid):
    error = F.smooth_l1_loss(reconstruction.transpose(1, 2), target.transpose(1, 2), reduction="none").mean(-1)
    return error[valid].mean()


def ti_tv_separation_loss(z_ti, z_tv, valid, max_points):
    """Discourage TI and TV branches from encoding the same factors."""
    ti = z_ti[valid]
    tv = z_tv[valid]
    if len(ti) < 2:
        return z_ti.new_tensor(0.0)
    if len(ti) > max_points:
        index = torch.randperm(len(ti), device=ti.device)[:max_points]
        ti, tv = ti[index], tv[index]
    ti = (ti - ti.mean(0)) / (ti.std(0, unbiased=False) + 1e-4)
    tv = (tv - tv.mean(0)) / (tv.std(0, unbiased=False) + 1e-4)
    cross_correlation = ti.T @ tv / len(ti)
    return cross_correlation.square().mean()


def training_loss(model, x, lengths, config):
    input_valid = length_mask(lengths, x.size(-1), x.device)
    view1 = augment(x, input_valid, config)
    view2 = augment(x, input_valid, config)
    output1 = model(view1)
    output2 = model(view2)
    z = output1["z_ti_raw"]
    latent_lengths = model.output_lengths(lengths)
    valid = length_mask(latent_lengths, z.size(1), x.device)
    x_down = model.x_down(x)
    target = x_down[:, :, : z.size(1)]
    invariance, variance, covariance = negative_free_consistency_loss(
        output1["z_ti_raw"], output2["z_ti_raw"], valid, config
    )
    if "reconstruction" in output1 and "reconstruction" in output2:
        reconstruction = 0.5 * (
            reconstruction_loss(output1["reconstruction"], target, valid)
            + reconstruction_loss(output2["reconstruction"], target, valid)
        )
    else:
        reconstruction = z.new_tensor(0.0)
    terms = {
        "invariance": invariance,
        "variance": variance,
        "covariance": covariance,
        "ti": ti_loss(output1["z_ti_raw"], target, valid, config.input_sigma_scale),
        "stable_consistency": stable_region_consistency_loss(
            output1["z_ti_raw"],
            target,
            valid,
            config.stable_consistency_steps,
            config.input_sigma_scale,
        ),
        "reconstruction": reconstruction,
        "separation": ti_tv_separation_loss(
            output1["z_ti_raw"], output1["z_tv"], valid, config.max_consistency_points
        ),
    }
    loss = (
        config.invariance_weight * terms["invariance"]
        + config.variance_weight * terms["variance"]
        + config.covariance_weight * terms["covariance"]
        + config.ti_weight * terms["ti"]
        + config.stable_consistency_weight * terms["stable_consistency"]
        + config.reconstruction_weight * terms["reconstruction"]
        + config.separation_weight * terms["separation"]
    )
    return loss, {k: float(v.detach()) for k, v in terms.items()}


def make_model(config):
    return LDCRNet(**asdict(config))


def save_checkpoint(path, model, model_config, loss_config, train_indices, test_indices):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "model_config": asdict(model_config),
        "loss_config": asdict(loss_config),
        "train_indices": train_indices,
        "test_indices": test_indices,
    }, path)


def load_checkpoint(path, device):
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    model = make_model(ModelConfig(**state["model_config"])).to(device)
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model, state
