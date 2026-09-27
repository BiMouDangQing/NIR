"""建模服务：把自监督预训练、特征学习、降维、异常检测编排成高层接口。

供 GUI（后台线程）与命令行调用。
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from model import anomaly, reduce
from model.outlier import cluster_outlier_mask
from model.preprocess import apply_preprocessing
from model.autoencoder import (
    Autoencoder,
    encode_all,
    get_device,
    load_model,
    parse_spectra,
    reconstruction_error,
    save_model,
    standardize,
    train_ae,
    train_val_split,
)

logger = logging.getLogger("NIR.model")


def run_training(
    data_path: str,
    out_dir: str,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    bottleneck: int = 16,
    train_ratio: float = 0.8,
    max_samples: int | None = None,
    device: str | None = None,
    preprocess: dict | None = None,
    remove_outliers: dict | None = None,
    pause_event=None,
    log=None,
) -> dict:
    """自监督预训练 + 特征/异常输出。

    ``device`` 为 None 自动，或 "cpu"/"cuda"；
    ``preprocess`` 为可选预处理开关（见 model.preprocess.apply_preprocessing）；
    ``remove_outliers`` 为可选离群剔除（见 model.outlier.cluster_outlier_mask）。
    """
    def _log(msg: str) -> None:
        logger.info(msg)
        if log:
            log(msg)

    device = get_device(device)
    _log(f"设备: {device}")

    df = pd.read_parquet(data_path)
    if max_samples:
        df = df.head(max_samples)
    X = parse_spectra(df, "SpectrumData")
    _log(f"样本数: {len(X)}，光谱维数: {X.shape[1]}")

    if preprocess:
        X = apply_preprocessing(X, preprocess)
        used = ", ".join(k for k, v in preprocess.items() if v)
        _log(f"已应用预处理: {used or '无'}")

    if remove_outliers:
        keep = cluster_outlier_mask(
            X,
            k=int(remove_outliers.get("k", 5)),
            percentile=float(remove_outliers.get("percentile", 99.0)),
            auto_k=bool(remove_outliers.get("auto", False)),
            log=_log,
        )
        removed = int((~keep).sum())
        X = X[keep]
        if remove_outliers.get("auto"):
            _log(f"离群剔除: 自动选 k，剔除 {removed} 个，保留 {len(X)} 个")
        else:
            _log(f"离群剔除: 聚类 k={remove_outliers.get('k', 5)}，剔除 {removed} 个，保留 {len(X)} 个")

    Xs, mean, std = standardize(X)
    Xtr, Xva = train_val_split(Xs, train_ratio)
    _log(f"训练/验证: {len(Xtr)} / {len(Xva)}（训练占比 {train_ratio:.0%}）")

    model = Autoencoder(X.shape[1], bottleneck)
    _log(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")
    history = train_ae(model, Xtr, epochs, batch_size, lr, Xva, device, pause_event, _log)

    out = save_model(model, out_dir, mean, std)
    err = reconstruction_error(model, Xs, device)

    pd.DataFrame({"reconstruction_error": err}).to_csv(
        out / "reconstruction_error.csv", index=False, encoding="utf-8-sig"
    )
    idx, top_err = anomaly.top_anomalies(err, 10)
    _log("重构误差最大的 10 个样本（潜在异常）：")
    for i, e in zip(idx, top_err):
        _log(f"  序号 {i}: 误差 {e:.4f}")

    summary = {
        "samples": len(X),
        "dim": int(X.shape[1]),
        "device": str(device),
        "train_loss": history["train"][-1],
        "val_loss": history["val"][-1] if history["val"] else None,
        "recon_err_mean": float(err.mean()),
        "recon_err_max": float(err.max()),
        "out_dir": str(out.resolve()),
        "history": history,
    }
    _log(f"训练完成，结果已保存: {out.resolve()}")
    return summary


def run_prediction(
    data_path: str,
    model_dir: str,
    out_path: str,
    max_samples: int | None = None,
    log=None,
) -> dict:
    """预测/异常检测：加载模型 → 重建 → 输出重构误差。"""
    def _log(msg: str) -> None:
        logger.info(msg)
        if log:
            log(msg)

    device = get_device()
    _log(f"设备: {device}")
    model, mean, std = load_model(model_dir)
    model.to(device)
    model.eval()

    df = pd.read_parquet(data_path)
    if max_samples:
        df = df.head(max_samples)
    X = parse_spectra(df, "SpectrumData")
    _log(f"样本数: {len(X)}，光谱维数: {X.shape[1]}")

    Xs = (X - mean) / (std + 1e-8)
    err = reconstruction_error(model, Xs, device)

    out_df = (
        df[["SnNumber"]].copy()
        if "SnNumber" in df.columns
        else pd.DataFrame(index=df.index)
    )
    out_df["reconstruction_error"] = err
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    idx, top_err = anomaly.top_anomalies(err, 10)
    _log("重构误差最大的 10 个样本：")
    for i, e in zip(idx, top_err):
        _log(f"  序号 {i}: 误差 {e:.4f}")

    summary = {
        "samples": len(X),
        "device": str(device),
        "recon_err_mean": float(err.mean()),
        "recon_err_max": float(err.max()),
        "out_path": str(out_path.resolve()),
    }
    _log(f"预测完成，结果已保存: {out_path.resolve()}")
    return summary


def run_visualize(
    data_path: str,
    model_dir: str,
    max_samples: int | None = None,
    percentile: float = 99.0,
    log=None,
) -> dict:
    """降维可视化：提取瓶颈特征 → PCA 到 2D，返回坐标与异常标记。"""
    def _log(msg: str) -> None:
        logger.info(msg)
        if log:
            log(msg)

    device = get_device()
    _log(f"设备: {device}")
    model, mean, std = load_model(model_dir)
    model.to(device)
    model.eval()

    df = pd.read_parquet(data_path)
    if max_samples:
        df = df.head(max_samples)
    X = parse_spectra(df, "SpectrumData")
    _log(f"样本数: {len(X)}，光谱维数: {X.shape[1]}")

    Xs = (X - mean) / (std + 1e-8)
    feats = encode_all(model, Xs, device)
    coords, explained = reduce.pca(feats, 2)
    err = reconstruction_error(model, Xs, device)
    flags, t = anomaly.detect(err, percentile)

    _log(f"PCA 前 2 主成分解释方差: {explained[:2].sum():.1%}")
    _log(f"异常阈值（{percentile:.0f} 分位）: {t:.4f}，异常样本数: {int(flags.sum())}")

    return {
        "coords": coords,
        "is_anomaly": flags,
        "threshold": t,
        "explained": float(explained[:2].sum()),
        "samples": len(X),
    }
