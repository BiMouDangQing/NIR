"""对抗域适应（DANN，Domain-Adversarial Neural Network）。

用于跨批次预测：训练编码器时加入「批次判别器」对抗，让瓶颈特征在源域
（有标签批次）与目标域（无标签批次）之间不可分，从而学到跨批不变的特征。

- ``GradientReversal``：梯度反转层（前向不变、反向取负）
- ``DomainDiscriminator``：批次判别器（特征 → 批次）
- ``train_dann``：对抗训练（回归 + 域判别）
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model.autoencoder import get_device


class GradientReversal(torch.autograd.Function):
    """梯度反转层：前向恒等，反向乘 -alpha。"""

    @staticmethod
    def forward(ctx, x, alpha):  # noqa: ANN001
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):  # noqa: ANN001
        return -ctx.alpha * grad_output, None


class DomainDiscriminator(nn.Module):
    """批次判别器：特征 z → 2 分类（源域/目标域）。"""

    def __init__(self, feat_dim: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def train_dann(
    encoder: nn.Module,
    head: nn.Module,
    disc: DomainDiscriminator,
    X_src: np.ndarray,
    y_src: np.ndarray,
    X_tgt: np.ndarray,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    lambda_domain: float = 1.0,
    device: torch.device | None = None,
    log=None,
) -> dict:
    """DANN 对抗训练。

    ``X_src``/``y_src`` 为源域（有标签）；``X_tgt`` 为目标域（无标签，只用于
    对抗判别）。返回 {"train_reg": [...], "train_disc": [...]}。
    输入均为 z-score 后的 float32（N, 1024）。
    """
    device = device or get_device()
    encoder.to(device)
    head.to(device)
    disc.to(device)

    Xs = torch.tensor(X_src, dtype=torch.float32, device=device)
    ys = torch.tensor(y_src, dtype=torch.float32, device=device)
    src_loader = DataLoader(TensorDataset(Xs, ys), batch_size=batch_size, shuffle=True)
    Xt = torch.tensor(X_tgt, dtype=torch.float32, device=device)
    tgt_loader = DataLoader(TensorDataset(Xt), batch_size=batch_size, shuffle=True)

    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()) + list(disc.parameters()),
        lr=lr,
    )
    reg_crit = nn.MSELoss()
    disc_crit = nn.CrossEntropyLoss()

    hist: dict = {"train_reg": [], "train_disc": []}
    for epoch in range(1, epochs + 1):
        # DANN 退火：对抗强度从 0 渐增到 1
        p = min(1.0, epoch / max(1, epochs * 0.5))
        alpha = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0  # 经典退火

        encoder.train()
        head.train()
        disc.train()
        tgt_iter = iter(tgt_loader)
        total_reg = 0.0
        total_disc = 0.0
        n_batches = 0
        for (xb, yb) in src_loader:
            # 取一个目标域 batch（循环填充）
            try:
                (tb,) = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                (tb,) = next(tgt_iter)

            # 源域特征
            zs = encoder(xb)
            pred = head(zs).squeeze(-1)
            reg_loss = reg_crit(pred, yb)

            # 目标域特征
            zt = encoder(tb)

            # 判别损失（源=0，目标=1）
            z_rev_s = GradientReversal.apply(zs, alpha)
            z_rev_t = GradientReversal.apply(zt, alpha)
            logits_s = disc(z_rev_s)
            logits_t = disc(z_rev_t)
            disc_loss = disc_crit(
                torch.cat([logits_s, logits_t], dim=0),
                torch.cat([torch.zeros(len(zs), dtype=torch.long, device=device),
                           torch.ones(len(zt), dtype=torch.long, device=device)]),
            )

            loss = reg_loss + lambda_domain * disc_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_reg += reg_loss.item() * len(xb)
            total_disc += disc_loss.item() * len(xb)
            n_batches += 1

        hist["train_reg"].append(total_reg / len(Xs))
        hist["train_disc"].append(total_disc / len(Xs))
        if log and (epoch == 1 or epoch % 25 == 0 or epoch == epochs):
            log(f"epoch {epoch}/{epochs}  reg={hist['train_reg'][-1]:.6f}  disc={hist['train_disc'][-1]:.6f}  alpha={alpha:.3f}")
    return hist


@torch.no_grad()
def dann_predict(encoder: nn.Module, head: nn.Module, X: np.ndarray,
                 device: torch.device | None = None, batch_size: int = 512) -> np.ndarray:
    """用 DANN 训练后的编码器 + 回归头预测糖度。X 为 z-score 后的 (N, 1024)。"""
    device = device or get_device()
    encoder.to(device)
    head.to(device)
    encoder.eval()
    head.eval()
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    out = []
    for i in range(0, len(Xt), batch_size):
        z = encoder(Xt[i:i + batch_size])
        out.append(head(z).squeeze(-1).cpu().numpy())
    return np.concatenate(out)
