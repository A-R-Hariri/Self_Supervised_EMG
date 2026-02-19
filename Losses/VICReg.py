import torch
import torch.nn.functional as F


# ======== AUGMENTATION (GPU, B,C,L) ========
def augment(x: torch.Tensor) -> torch.Tensor:
    B, C, L = x.shape
    y = x.clone().detach()

    # ---- (1) global amplitude scaling (per sample) ----
    mask = torch.rand(B, device=x.device) < 0.25
    scale = 1.0 + 0.1 * torch.randn(B, device=x.device, dtype=x.dtype)
    y[mask] *= scale[mask].view(-1, 1, 1)

    # ---- (2) per-channel amplitude scaling (per sample) ----
    if torch.rand((), device=x.device) < 0.25:
        ch_scale = 1.0 + 0.1 * torch.randn(
            B, C, device=x.device, dtype=x.dtype)
        y *= ch_scale.unsqueeze(-1)

    # ---- (3) temporal roll (per sample, vectorized) ----
    if torch.rand((), device=x.device) < 0.25:
        shifts = torch.randint(-4, 5, (B,), device=x.device)
        idx = (torch.arange(L, device=x.device)[None, :] - shifts[:, None]) % L
        y = y.gather(2, idx.unsqueeze(1).expand(-1, C, -1))

    # ---- (4) gaussian noise (per sample) ----
    if torch.rand((), device=x.device) < 0.25:
        std = y.std(dim=(1, 2), keepdim=True)
        noise = 0.02 * std * torch.randn(
           y.shape, device=x.device, dtype=x.dtype)
        y += noise

    # ---- (5) channel dropout (per sample, max 1 channel) ----
    if torch.rand((), device=x.device) < 0.1:
        drop_mask = torch.zeros(B, C, device=x.device, dtype=y.dtype)
        ch_idx = torch.randint(0, C, (B,), device=x.device)
        drop_mask.scatter_(1, ch_idx[:, None], 1.0)
        y = y * (1.0 - drop_mask.unsqueeze(-1))

    return y


# ======== VICREG LOSS ========
def var_term(z: torch.Tensor, gamma: float, eps: float = 1e-4) -> torch.Tensor:
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    return torch.mean(F.relu(gamma - std))

def cov_term(z: torch.Tensor) -> torch.Tensor:
    z = z - z.mean(dim=0)
    n = z.size(0)
    cov = (z.T @ z) / max(1, (n - 1))
    off = cov - torch.diag(torch.diag(cov))
    return (off ** 2).mean()

def vicreg_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    lamb: float = 8.0,
    mu: float = 32.0,
    nu: float = 1.0,
    gamma: float = 1.0) -> torch.Tensor:
    sim = torch.mean((z1 - z2) ** 2)
    v = var_term(z1, gamma) + var_term(z2, gamma)
    c = cov_term(z1) + cov_term(z2)
    return lamb * sim + mu * v + nu * c