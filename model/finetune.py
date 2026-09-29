"""有监督微调：在自编码器瓶颈特征上训练回归头，用有标签数据预测指标（如糖度）。

迁移学习思路：冻结预训练编码器 → 瓶颈特征（16 维）→ 回归头 → 标签。
标签来自 ``PredictedValue`` 等逗号分隔字段的某个分量（第 1 分量=糖度 Brix）。
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model.autoencoder import Autoencoder, get_device, load_model, parse_spectra, standardize

logger = __import__("logging").getLogger("NIR.model")


class RegressionHead(nn.Module):
    """瓶颈特征 → 单值标签 的回归头 MLP（带 Dropout 防过拟合）。"""

    def __init__(self, in_dim: int, hidden: int = 32, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def parse_label(df: pd.DataFrame, column: str = "PredictedValue", index: int = 0) -> np.ndarray:
    """从逗号分隔的标签字段解析第 ``index`` 个分量，返回 float32 数组（无效值记为 NaN）。"""
    parts = df[column].astype(str).str.split(",", expand=True)
    if parts.shape[1] <= index:
        raise ValueError(
            f"标签字段 {column} 只有 {parts.shape[1]} 个分量，无法取第 {index + 1} 个"
        )
    return pd.to_numeric(parts[index], errors="coerce").to_numpy(dtype=np.float32)


def train_finetune(
    encoder: Autoencoder,
    head: RegressionHead,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    val_X: np.ndarray | None = None,
    val_y: np.ndarray | None = None,
    freeze: bool = True,
    unfreeze_epochs: int = 0,
    unfreeze_lr: float | None = None,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
    pause_event: threading.Event | None = None,
    on_epoch=None,
    log=None,
) -> dict:
    """微调训练。

    - ``freeze=True`` 且 ``unfreeze_epochs=0``：全程冻结编码器，只训回归头。
    - ``freeze=True`` 且 ``unfreeze_epochs>0``：两阶段——先冻结训回归头
      ``epochs-unfreeze_epochs`` 轮，再解冻编码器用 ``unfreeze_lr``（默认 ``lr*0.1``）
      联合微调 ``unfreeze_epochs`` 轮。
    - ``freeze=False``：全程解冻联合微调。

    标签在内部做 z-score 标准化，预测时需用返回的 ``y_mean``/``y_std`` 反变换。
    返回 {"train": [...], "val": [...], "y_mean": float, "y_std": float}（MSE 为标准化尺度）。
    """
    device = device or get_device()
    encoder.to(device)
    head.to(device)

    y_mean = float(np.mean(y))
    y_std = float(np.std(y)) or 1.0
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    yt = torch.tensor((y - y_mean) / y_std, dtype=torch.float32, device=device)
    Xv = torch.tensor(val_X, dtype=torch.float32, device=device) if val_X is not None else None
    yv = (
        torch.tensor((val_y - y_mean) / y_std, dtype=torch.float32, device=device)
        if val_y is not None
        else None
    )

    def _set_frozen(frozen: bool) -> None:
        for p in encoder.parameters():
            p.requires_grad_(not frozen)
        encoder.eval() if frozen else encoder.train()

    two_stage = bool(freeze and unfreeze_epochs > 0)
    stage1_epochs = epochs - unfreeze_epochs if two_stage else epochs
    unfreeze_lr = unfreeze_lr or lr * 0.1
    _set_frozen(freeze)
    head.train()

    criterion = nn.MSELoss()
    train_losses: list[float] = []
    val_losses: list[float] = []

    def _make_opt(lr_val: float) -> torch.optim.Adam:
        params = list(head.parameters())
        if not freeze or two_stage:
            params = params + list(encoder.parameters())
        return torch.optim.Adam(params, lr=lr_val, weight_decay=weight_decay)

    opt = _make_opt(lr)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        if two_stage and epoch == stage1_epochs + 1:
            _set_frozen(False)
            head.train()
            opt = torch.optim.Adam(
                list(head.parameters()) + list(encoder.parameters()),
                lr=unfreeze_lr,
                weight_decay=weight_decay,
            )
            if log:
                log(f"阶段2: 解冻编码器联合微调（lr={unfreeze_lr}）")

        if pause_event is not None and pause_event.is_set():
            if log:
                log("微调已暂停，等待继续...")
            while pause_event.is_set():
                time.sleep(0.1)
            if log:
                log("微调继续。")

        head.train()
        total = 0.0
        for bx, by in loader:
            opt.zero_grad()
            pred = head(encoder.encode(bx))
            loss = criterion(pred, by)
            loss.backward()
            opt.step()
            total += loss.item() * len(bx)
        epoch_loss = total / len(Xt)
        train_losses.append(epoch_loss)

        val_loss: float | None = None
        val_metrics = None
        if Xv is not None:
            head.eval()
            encoder.eval()
            with torch.no_grad():
                val_pred = head(encoder.encode(Xv))
                val_loss = float(criterion(val_pred, yv).item())
                val_pred_real = val_pred.cpu().numpy() * y_std + y_mean
                val_metrics = {
                    "rmse": float(np.sqrt(np.mean((val_pred_real - val_y) ** 2))),
                    "r2": regression_metrics(val_y, val_pred_real)["r2"],
                }
            val_losses.append(val_loss)
            head.train()
            if not freeze or (two_stage and epoch > stage1_epochs):
                encoder.train()

        if on_epoch is not None:
            on_epoch(epoch, epochs, epoch_loss, val_loss, val_metrics)

        if log and (
            epoch == 1
            or epoch % 5 == 0
            or epoch == epochs
            or (two_stage and epoch == stage1_epochs + 1)
        ):
            vmsg = f"  val={val_loss:.6f}" if val_loss is not None else ""
            if val_metrics is not None:
                vmsg += f"  val_RMSE={val_metrics['rmse']:.4f}  val_R²={val_metrics['r2']:.4f}"
            log(f"epoch {epoch}/{epochs}  loss={epoch_loss:.6f}{vmsg}")
    return {"train": train_losses, "val": val_losses, "y_mean": y_mean, "y_std": y_std}


@torch.no_grad()
def predict_finetune(
    encoder: Autoencoder,
    head: RegressionHead,
    X: np.ndarray,
    device: torch.device | None = None,
    batch_size: int = 512,
    y_mean: float | None = None,
    y_std: float | None = None,
) -> np.ndarray:
    """用微调模型预测标签，返回 float32 数组。传入 ``y_mean``/``y_std`` 时反变换回真实尺度。"""
    device = device or get_device()
    encoder.to(device)
    head.to(device)
    encoder.eval()
    head.eval()
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    preds = [
        head(encoder.encode(Xt[i : i + batch_size])).cpu().numpy()
        for i in range(0, len(Xt), batch_size)
    ]
    out = np.concatenate(preds) if preds else np.array([], dtype=np.float32)
    if y_mean is not None and y_std is not None:
        out = out * float(y_std) + float(y_mean)
    return out


def save_finetune(
    head: RegressionHead,
    out_dir: str | Path,
    encoder: Autoencoder,
    mean: np.ndarray,
    std: np.ndarray,
    label_column: str,
    label_index: int,
    freeze: bool,
    head_dim: int,
    y_mean: float = 0.0,
    y_std: float = 1.0,
    dropout: float = 0.2,
) -> Path:
    """保存回归头与编码器权重及配置；解冻微调会修改编码器，故一并保存。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), out_dir / "finetune_head.pt")
    torch.save(encoder.state_dict(), out_dir / "finetune_encoder.pt")
    (out_dir / "finetune_meta.json").write_text(
        json.dumps(
            {
                "label_column": label_column,
                "label_index": int(label_index),
                "freeze": bool(freeze),
                "head_dim": int(head_dim),
                "y_mean": float(y_mean),
                "y_std": float(y_std),
                "dropout": float(dropout),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return out_dir


def load_finetune(model_dir: str | Path):
    """加载微调模型，返回 (encoder, head, meta, mean, std)。

    优先使用微调目录保存的编码器权重（解冻微调会修改），否则用原始预训练模型。
    """
    model_dir = Path(model_dir)
    meta = json.loads((model_dir / "finetune_meta.json").read_text(encoding="utf-8"))
    base_file = model_dir / "base_model.json"
    if base_file.exists():
        base_dir = Path(json.loads(base_file.read_text(encoding="utf-8"))["base_model_dir"])
    else:
        base_dir = model_dir  # 兼容：微调文件与编码器同目录
    encoder, mean, std = load_model(base_dir)
    enc_file = model_dir / "finetune_encoder.pt"
    if enc_file.exists():
        encoder.load_state_dict(
            torch.load(enc_file, map_location="cpu", weights_only=True)
        )
    head = RegressionHead(int(meta["head_dim"]), dropout=float(meta.get("dropout", 0.2)))
    head.load_state_dict(
        torch.load(model_dir / "finetune_head.pt", map_location="cpu", weights_only=True)
    )
    return encoder, head, meta, mean, std


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """计算 RMSE / MAE / R² / 相关系数 / 平均偏差。"""
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    diff = y_pred - y_true
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    corr = 0.0
    if len(y_true) > 1 and float(np.std(y_true)) > 0 and float(np.std(y_pred)) > 0:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    bias = float(np.mean(diff))
    return {"rmse": rmse, "mae": mae, "r2": r2, "corr": corr, "bias": bias}


def extended_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """在基础回归指标上补充 MAPE、误差标准差、误差分位数与偏差区间占比。"""
    m = regression_metrics(y_true, y_pred)
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    err = y_pred - y_true
    mask = np.abs(y_true) > 1e-6
    mape = (
        float(np.mean(np.abs(err[mask] / y_true[mask])) * 100.0)
        if mask.any()
        else 0.0
    )
    m["mape"] = mape
    m["err_std"] = float(np.std(err))
    m["err_p5"] = float(np.percentile(err, 5))
    m["err_p95"] = float(np.percentile(err, 95))
    # 误差落在 ±0.5 度 / ±1 度 内的样本占比（0~1，绝对值，单位=糖度）
    m["within_0p5"] = float(np.mean(np.abs(err) <= 0.5))
    m["within_1p0"] = float(np.mean(np.abs(err) <= 1.0))
    return m
