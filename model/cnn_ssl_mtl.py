"""双通道自监督多任务学习（SSL-MTL-CNN）。

- 双通道：原始光谱 + 一阶导数光谱（均 1024 维，从 SpectrumData 派生，
  规避标签数据缺 Abs 的问题）
- 编码器：1D CNN（局部卷积 + 池化）
- SSL 预训练：无标签数据重建双通道光谱（自监督）
- 微调：糖度回归（主任务）+ 可选重建辅助损失（多任务正则）

输入统一为 z-score 后的 float32 张量。
"""
from __future__ import annotations

import threading
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model.autoencoder import get_device


def make_dual_channel(X: np.ndarray) -> np.ndarray:
    """把单通道光谱 (N, L) 变成双通道 (N, 2, L)：[原始, 一阶导数]。"""
    deriv = np.gradient(X, axis=1)
    deriv = (deriv - deriv.mean(0)) / (deriv.std(0) + 1e-8)
    return np.stack([X, deriv], axis=1).astype(np.float32)


class CNNDualEncoder(nn.Module):
    """双通道 1D CNN 编码器：输入 (B, 2, L)，输出 (B, feat_dim)。"""

    def __init__(self, feat_dim: int = 64, in_channels: int = 2) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),          # L/2
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),          # L/4
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),         # L/8
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(1),  # 1
        )
        self.fc = nn.Linear(128, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x).squeeze(-1))


class SSLCMTModel(nn.Module):
    """双通道 CNN 编码器 + 全连接解码器（SSL 重建用）。"""

    def __init__(self, feat_dim: int = 64, length: int = 1024, in_channels: int = 2) -> None:
        super().__init__()
        self.encoder = CNNDualEncoder(feat_dim, in_channels)
        self.decoder = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.ReLU(),
            nn.Linear(256, in_channels * length),
        )
        self.length = length
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (瓶颈特征, 重建双通道)。"""
        z = self.encoder(x)
        rec = self.decoder(z).view(x.shape[0], self.in_channels, self.length)
        return z, rec


class CNNRegressionHead(nn.Module):
    """糖度回归头：feat_dim → 32 → 1。"""

    def __init__(self, feat_dim: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


def train_ssl(
    model: SSLCMTModel,
    X: np.ndarray,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    val_X: np.ndarray | None = None,
    device: torch.device | None = None,
    pause_event: threading.Event | None = None,
    on_epoch=None,
    log=None,
) -> dict:
    """SSL 预训练：重建双通道光谱（MSE）。X 形状 (N, 2, L)。"""
    device = device or get_device()
    model.to(device)
    model.train()
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    loader = DataLoader(TensorDataset(Xt), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    train_losses: list[float] = []
    val_losses: list[float] = []
    Xv = torch.tensor(val_X, dtype=torch.float32, device=device) if val_X is not None else None
    for epoch in range(1, epochs + 1):
        if pause_event is not None and pause_event.is_set():
            if log:
                log("训练已暂停，等待继续...")
            while pause_event.is_set():
                time.sleep(0.1)
            if log:
                log("训练继续。")
        model.train()
        total = 0.0
        for (batch,) in loader:
            opt.zero_grad()
            _, rec = model(batch)
            loss = criterion(rec, batch)
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)
        train_losses.append(total / len(Xt))
        val_loss: float | None = None
        if Xv is not None:
            model.eval()
            with torch.no_grad():
                _, rec = model(Xv)
                val_loss = float(criterion(rec, Xv).item())
            val_losses.append(val_loss)
        if on_epoch is not None:
            on_epoch(epoch, epochs, train_losses[-1], val_loss, None)
        if log and (epoch == 1 or epoch % 5 == 0 or epoch == epochs):
            vmsg = f"  val={val_loss:.6f}" if val_loss is not None else ""
            log(f"epoch {epoch}/{epochs}  loss={train_losses[-1]:.6f}{vmsg}")
    return {"train": train_losses, "val": val_losses}


def train_mtl(
    model: SSLCMTModel,
    head: CNNRegressionHead,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 300,
    batch_size: int = 64,
    lr: float = 1e-3,
    freeze: bool = True,
    unfreeze_epochs: int = 0,
    unfreeze_lr: float | None = None,
    recon_weight: float = 0.0,
    val_X: np.ndarray | None = None,
    val_y: np.ndarray | None = None,
    device: torch.device | None = None,
    pause_event: threading.Event | None = None,
    on_epoch=None,
    log=None,
) -> dict:
    """多任务微调：糖度回归（主）+ 重建（辅助，权重 recon_weight）。

    X 形状 (N, 2, L)。freeze=True 时冻结编码器只训回归头；unfreeze_epochs>0
    时后段解冻编码器联合微调。y 需 z-score 标准化。返回历史与 y_mean/y_std。
    """
    device = device or get_device()
    model.to(device)
    head.to(device)
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.float32, device=device)
    ds = TensorDataset(Xt, yt)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    for p in model.parameters():
        p.requires_grad = not freeze
    opt = torch.optim.Adam(
        [{"params": head.parameters(), "lr": lr}]
        + ([{"params": model.parameters(), "lr": lr}] if not freeze else []),
    )
    reg_crit = nn.MSELoss()
    rec_crit = nn.MSELoss()

    history: dict = {"train": [], "val": []}
    total_epochs = epochs
    for epoch in range(1, total_epochs + 1):
        if freeze and unfreeze_epochs > 0 and epoch == total_epochs - unfreeze_epochs + 1:
            for p in model.parameters():
                p.requires_grad = True
            u_lr = unfreeze_lr if unfreeze_lr else lr * 0.1
            opt = torch.optim.Adam(
                [{"params": head.parameters(), "lr": u_lr},
                 {"params": model.parameters(), "lr": u_lr}],
            )
            if log:
                log(f"epoch {epoch}: 解冻编码器，学习率 {u_lr}")
        model.train()
        head.train()
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            z, rec = model(xb)
            pred = head(z)
            reg = reg_crit(pred, yb)
            if recon_weight > 0:
                rec_loss = rec_crit(rec, xb)
                loss = reg + recon_weight * rec_loss
            else:
                loss = reg
            loss.backward()
            opt.step()
            total += reg.item() * len(xb)
        history["train"].append(total / len(Xt))
        val_loss: float | None = None
        if val_X is not None and val_y is not None:
            model.eval()
            head.eval()
            with torch.no_grad():
                Xv = torch.tensor(val_X, dtype=torch.float32, device=device)
                yv = torch.tensor(val_y, dtype=torch.float32, device=device)
                z, _ = model(Xv)
                val_loss = float(reg_crit(head(z), yv).item())
            history["val"].append(val_loss)
        if on_epoch is not None:
            on_epoch(epoch, total_epochs, history["train"][-1], val_loss, None)
        if log and (epoch == 1 or epoch % 25 == 0 or epoch == total_epochs):
            vmsg = f"  val={val_loss:.6f}" if val_loss is not None else ""
            log(f"epoch {epoch}/{total_epochs}  loss={history['train'][-1]:.6f}{vmsg}")
    return history


@torch.no_grad()
def predict(model: SSLCMTModel, head: CNNRegressionHead, X: np.ndarray,
           device: torch.device | None = None, batch_size: int = 512) -> np.ndarray:
    """用回归头预测糖度。X 形状 (N, 2, L)。"""
    device = device or get_device()
    model.to(device)
    head.to(device)
    model.eval()
    head.eval()
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    out = []
    for i in range(0, len(Xt), batch_size):
        z, _ = model(Xt[i:i + batch_size])
        out.append(head(z).cpu().numpy())
    return np.concatenate(out)
