"""自编码器模型：网络定义、自监督训练、特征提取与重构。"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

INPUT_DIM = 1024
BOTTLENECK = 16


def find_default_file() -> str | None:
    """在常见数据目录中自动查找 NIR parquet 文件。"""
    for base in (Path("D:/model/data"), Path("data")):
        if base.exists():
            files = sorted(p for p in base.rglob("*.parquet") if "NIR" in str(p))
            if files:
                return str(files[0])
    return None


def get_device(prefer: str | None = None) -> torch.device:
    """解析训练设备。

    - ``prefer`` 为 "cuda"/"gpu" 时强制 GPU（不可用则报错）
    - 为 "cpu" 时强制 CPU
    - 为 None 时自动检测：有 GPU 用 GPU，否则 CPU
    """
    if prefer is not None:
        name = prefer.strip().lower()
        if name in ("cuda", "gpu"):
            if torch.cuda.is_available():
                return torch.device("cuda")
            raise RuntimeError("已选择 GPU 训练，但未检测到可用的 CUDA 设备")
        if name == "cpu":
            return torch.device("cpu")
        raise ValueError(f"未知设备: {prefer!r}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_spectra(df: pd.DataFrame, column: str = "SpectrumData") -> np.ndarray:
    """把逗号分隔的 1024 维光谱字符串解析为 float32 矩阵。"""
    return np.stack([np.array(s.split(","), dtype=np.float32) for s in df[column]])


def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """逐波长 Z-score 标准化，返回 (标准化矩阵, 均值, 标准差)。"""
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    return (X - mean) / std, mean, std


class Autoencoder(nn.Module):
    """对称全连接自编码器：input_dim → 256 → 64 → bottleneck → 64 → 256 → input_dim。"""

    def __init__(self, input_dim: int = INPUT_DIM, bottleneck: int = BOTTLENECK) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, bottleneck),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def train_val_split(
    X: np.ndarray, ratio: float, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """按比例划分训练/验证集（ratio 为训练占比）。"""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_train = int(len(X) * ratio)
    return X[idx[:n_train]], X[idx[n_train:]]


def train_ae(
    model: Autoencoder,
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
    """自监督训练（重建损失）。返回 {"train": [...], "val": [...]}。

    ``pause_event`` 非空时，每个 epoch 开始前若被 set 则阻塞等待（用于暂停）；
    ``on_epoch`` 每个 epoch 结束时回调 ``on_epoch(epoch, epochs, train_loss, val_loss)``。
    """
    device = device or get_device()
    model.to(device)
    model.train()
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    loader = DataLoader(TensorDataset(Xt), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    train_losses: list[float] = []
    val_losses: list[float] = []
    Xv = (
        torch.tensor(val_X, dtype=torch.float32, device=device)
        if val_X is not None
        else None
    )
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
            loss = criterion(model(batch), batch)
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)
        epoch_loss = total / len(Xt)
        train_losses.append(epoch_loss)

        val_loss: float | None = None
        if Xv is not None:
            model.eval()
            with torch.no_grad():
                val_loss = float(criterion(model(Xv), Xv).item())
            val_losses.append(val_loss)

        if on_epoch is not None:
            on_epoch(epoch, epochs, epoch_loss, val_loss, None)

        if log and (epoch == 1 or epoch % 5 == 0 or epoch == epochs):
            vmsg = f"  val={val_loss:.6f}" if val_loss is not None else ""
            log(f"epoch {epoch}/{epochs}  loss={epoch_loss:.6f}{vmsg}")
    return {"train": train_losses, "val": val_losses}


@torch.no_grad()
def encode_all(
    model: Autoencoder,
    X: np.ndarray,
    device: torch.device | None = None,
    batch_size: int = 512,
) -> np.ndarray:
    """特征学习：提取瓶颈层特征。"""
    device = device or get_device()
    model.to(device)
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    feats = [
        model.encode(Xt[i : i + batch_size]).cpu().numpy()
        for i in range(0, len(Xt), batch_size)
    ]
    return np.concatenate(feats)


@torch.no_grad()
def reconstruction_error(
    model: Autoencoder,
    X: np.ndarray,
    device: torch.device | None = None,
    batch_size: int = 512,
) -> np.ndarray:
    """逐样本重建 MSE，用于异常检测。"""
    device = device or get_device()
    model.to(device)
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    errs = [
        ((model(Xt[i : i + batch_size]) - Xt[i : i + batch_size]) ** 2)
        .mean(dim=1)
        .cpu()
        .numpy()
        for i in range(0, len(Xt), batch_size)
    ]
    return np.concatenate(errs)


def save_model(
    model: Autoencoder, out_dir: str | Path, mean: np.ndarray, std: np.ndarray
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "autoencoder.pt")
    np.savez(out_dir / "standardize.npz", mean=mean, std=std)
    return out_dir


def load_model(model_dir: str | Path, input_dim: int | None = None, bottleneck: int | None = None):
    model_dir = Path(model_dir)
    state = torch.load(model_dir / "autoencoder.pt", map_location="cpu", weights_only=True)
    # 从权重推断真实输入维度与瓶颈维度，兼容 SpectrumData(1024)/Abs(301) 等不同字段
    if input_dim is None:
        input_dim = int(state["encoder.0.weight"].shape[1])
    if bottleneck is None:
        bottleneck = int(state["encoder.4.weight"].shape[0])
    model = Autoencoder(input_dim, bottleneck)
    model.load_state_dict(state)
    d = np.load(model_dir / "standardize.npz")
    return model, d["mean"], d["std"]
