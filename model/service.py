"""建模服务：把自监督预训练、特征学习、降维、异常检测编排成高层接口。

供 GUI（后台线程）与命令行调用。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model import anomaly, finetune, reduce
from model.outlier import cluster_outlier_mask
from model.preprocess import apply_preprocessing, compute_msc_ref
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

# 预测结果默认保存目录（项目根/predict）
PREDICT_DIR = Path(__file__).resolve().parents[1] / "predict"
# 微调模型默认保存目录（项目根/finetune_output，按日期时间建子目录）
FINETUNE_DIR = Path(__file__).resolve().parents[1] / "finetune_output"


def _load_spectrum_column(model_dir: str | Path) -> str:
    """从模型目录读取训练时使用的光谱字段，缺省 SpectrumData。"""
    f = Path(model_dir) / "spectrum_column.json"
    if f.exists():
        try:
            return str(json.loads(f.read_text(encoding="utf-8")).get("spectrum_column", "SpectrumData"))
        except (json.JSONDecodeError, OSError):
            pass
    return "SpectrumData"


def _error_histogram(err: np.ndarray, bins: int = 25) -> tuple[list[float], list[int]]:
    """对数分桶的重构误差直方图，返回 (桶中心值, 各桶样本数)。

    重构误差是重尾分布（正常样本集中、异常样本稀疏），用对数分桶才能同时看到两者。
    """
    pos = err[err > 0]
    if len(pos) == 0:
        return [], []
    lo = float(np.log10(pos.min()))
    hi = float(np.log10(pos.max()))
    edges = np.linspace(lo, hi, bins + 1)
    counts, _ = np.histogram(pos, bins=np.power(10.0, edges))
    centers = np.sqrt(np.power(10.0, edges[:-1]) * np.power(10.0, edges[1:]))
    return centers.tolist(), counts.tolist()


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
    on_epoch=None,
    spectrum_column: str = "SpectrumData",
    recon_top_n: int = 3,
    log=None,
) -> dict:
    """自监督预训练 + 特征/异常输出。

    ``device`` 为 None 自动，或 "cpu"/"cuda"；
    ``preprocess`` 为可选预处理开关（见 model.preprocess.apply_preprocessing）；
    ``remove_outliers`` 为可选离群剔除（见 model.outlier.cluster_outlier_mask）；
    ``recon_top_n`` 为重建对比图中展示的误差最大样本数量。
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
    X = parse_spectra(df, spectrum_column)
    _log(f"样本数: {len(X)}，光谱维数: {X.shape[1]}（字段: {spectrum_column}）")

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
    history = train_ae(model, Xtr, epochs, batch_size, lr, Xva, device, pause_event, on_epoch, _log)

    # 按日期+时间在指定目录下创建子文件夹存储本次训练结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(out_dir) / timestamp
    _log(f"输出目录: {out_dir}")
    out = save_model(model, out_dir, mean, std)
    # 保存预处理配置与光谱字段，供预测/可视化复用同一套设置
    (out / "spectrum_column.json").write_text(
        json.dumps({"spectrum_column": spectrum_column}, ensure_ascii=False), encoding="utf-8"
    )
    if preprocess:
        (out / "preprocess.json").write_text(
            json.dumps(preprocess, ensure_ascii=False), encoding="utf-8"
        )
    err = reconstruction_error(model, Xs, device)

    pd.DataFrame({"reconstruction_error": err}).to_csv(
        out / "reconstruction_error.csv", index=False, encoding="utf-8-sig"
    )
    idx, top_err = anomaly.top_anomalies(err, 10)
    _log("重构误差最大的 10 个样本（潜在异常）：")
    for i, e in zip(idx, top_err):
        _log(f"  序号 {i}: 误差 {e:.4f}")

    # 重建光谱对比样本：误差最大的 top N 个 + 误差最小 1 个，反标准化回原始尺度
    model.eval()
    with torch.no_grad():
        rec = model(torch.tensor(Xs, dtype=torch.float32, device=device)).cpu().numpy()
    X_orig = Xs * std + mean
    rec_orig = rec * std + mean
    top_n = max(1, int(recon_top_n))
    worst_order = np.argsort(err)[::-1][:top_n]
    worst_samples = []
    for rank, i in enumerate(worst_order, 1):
        worst_samples.append(
            {
                "label": f"误差第 {rank} 大（{err[i]:.4f}）",
                "original": X_orig[i].tolist(),
                "recon": rec_orig[i].tolist(),
            }
        )
    best_i = int(np.argmin(err))
    recon_samples = {
        "worst": worst_samples,
        "best": {
            "label": f"重建误差最小（{err[best_i]:.4f}）",
            "original": X_orig[best_i].tolist(),
            "recon": rec_orig[best_i].tolist(),
        },
    }

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
        "recon_samples": recon_samples,
    }
    _log(f"训练完成，结果已保存: {out.resolve()}")
    return summary


def run_prediction(
    data_path: str,
    model_dir: str,
    out_path: str | None = None,
    max_samples: int | None = None,
    log=None,
) -> dict:
    """预测/异常检测：加载模型 → 重建 → 输出重构误差。

    ``out_path`` 为 None 时自动保存到 ``predict/YYYYMMDD_HHMMSS.csv``。
    """
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
    X = parse_spectra(df, _load_spectrum_column(model_dir))
    _log(f"样本数: {len(X)}，光谱维数: {X.shape[1]}")

    # 复用训练时的预处理（若模型目录保存了 preprocess.json）
    pp_file = Path(model_dir) / "preprocess.json"
    if pp_file.exists():
        preprocess = json.loads(pp_file.read_text(encoding="utf-8"))
        X = apply_preprocessing(X, preprocess)
        used = ", ".join(k for k, v in preprocess.items() if v)
        _log(f"已应用预处理: {used or '无'}")

    Xs = (X - mean) / (std + 1e-8)
    err = reconstruction_error(model, Xs, device)

    out_df = (
        df[["SnNumber"]].copy()
        if "SnNumber" in df.columns
        else pd.DataFrame(index=df.index)
    )
    out_df["reconstruction_error"] = err
    if out_path:
        out_path = Path(out_path)
        out_dir = out_path.parent
    else:
        out_dir = PREDICT_DIR / f"{datetime.now():%Y%m%d_%H%M%S}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "reconstruction_error.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    idx, top_err = anomaly.top_anomalies(err, 10)
    _log("重构误差最大的 10 个样本：")
    for i, e in zip(idx, top_err):
        _log(f"  序号 {i}: 误差 {e:.4f}")

    flags, _ = anomaly.detect(err, 99.0)
    hist_centers, hist_counts = _error_histogram(err)
    summary = {
        "samples": len(X),
        "device": str(device),
        "recon_err_mean": float(err.mean()),
        "recon_err_median": float(np.median(err)),
        "recon_err_p95": float(np.percentile(err, 95)),
        "recon_err_p99": float(np.percentile(err, 99)),
        "recon_err_max": float(err.max()),
        "anomaly_count": int(flags.sum()),
        "hist_centers": hist_centers,
        "hist_counts": hist_counts,
        "out_path": str(out_path.resolve()),
        "out_dir": str(out_dir.resolve()),
    }
    _log(f"预测完成，结果已保存: {out_path.resolve()}")
    return summary


def run_visualize(
    data_path: str,
    model_dir: str,
    max_samples: int | None = None,
    percentile: float = 99.0,
    n_components: int = 2,
    log=None,
) -> dict:
    """降维可视化：提取瓶颈特征 → PCA 降维，返回坐标与异常标记。

    ``n_components`` 为 2 或 3，决定二维/三维散点图。
    """
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
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=0)
    X = parse_spectra(df, _load_spectrum_column(model_dir))
    _log(f"样本数: {len(X)}，光谱维数: {X.shape[1]}")

    # 复用训练时的预处理（若模型目录保存了 preprocess.json）
    pp_file = Path(model_dir) / "preprocess.json"
    if pp_file.exists():
        preprocess = json.loads(pp_file.read_text(encoding="utf-8"))
        X = apply_preprocessing(X, preprocess)
        used = ", ".join(k for k, v in preprocess.items() if v)
        _log(f"已应用预处理: {used or '无'}")

    Xs = (X - mean) / (std + 1e-8)
    feats = encode_all(model, Xs, device)
    coords, explained = reduce.pca(feats, n_components)
    err = reconstruction_error(model, Xs, device)
    flags, t = anomaly.detect(err, percentile)

    _log(f"PCA 前 {n_components} 主成分解释方差: {explained[:n_components].sum():.1%}")
    _log(f"异常阈值（{percentile:.2f} 分位）: {t:.4f}，异常样本数: {int(flags.sum())}")

    return {
        "coords": coords,
        "is_anomaly": flags,
        "threshold": t,
        "explained": float(explained[:n_components].sum()),
        "samples": len(X),
        "n_components": n_components,
    }


def _cross_validate(model, Xs, y, k_folds, bottleneck, epochs, batch_size, lr, freeze, unfreeze_epochs, unfreeze_lr, weight_decay, dropout, device, log):
    """K 折交叉验证评估：每折训练一个回归头，返回各折指标与均值/标准差。"""
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(Xs))
    folds = np.array_split(perm, k_folds)
    original_state = {k: v.clone() for k, v in model.state_dict().items()}
    fold_metrics = []
    for i, val_idx in enumerate(folds, 1):
        train_idx = np.concatenate([f for j, f in enumerate(folds) if j != i - 1])
        model.load_state_dict(original_state)  # 每折从原始权重开始
        head = finetune.RegressionHead(bottleneck, dropout=dropout)
        hist = finetune.train_finetune(
            model, head, Xs[train_idx], y[train_idx], epochs, batch_size, lr,
            None, None, freeze, unfreeze_epochs, unfreeze_lr, weight_decay,
            device, None, None, None,
        )
        preds = finetune.predict_finetune(
            model, head, Xs[val_idx], device,
            y_mean=hist["y_mean"], y_std=hist["y_std"],
        )
        m = finetune.regression_metrics(y[val_idx], preds)
        fold_metrics.append(m)
        if log:
            log(f"  折 {i}/{k_folds}: RMSE={m['rmse']:.4f}, MAE={m['mae']:.4f}, R²={m['r2']:.4f}, corr={m['corr']:.4f}")
    keys = ["rmse", "mae", "r2", "corr", "bias"]
    mean = {k: float(np.mean([m[k] for m in fold_metrics])) for k in keys}
    std = {k: float(np.std([m[k] for m in fold_metrics])) for k in keys}
    return {"k": k_folds, "folds": fold_metrics, "mean": mean, "std": std}


def run_finetune(
    data_path: str,
    model_dir: str,
    out_dir: str | None = None,
    label_column: str = "RealValue",
    label_index: int = 0,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    freeze: bool = True,
    train_ratio: float = 0.8,
    max_samples: int | None = None,
    k_folds: int = 0,
    unfreeze_epochs: int = 0,
    unfreeze_lr: float | None = None,
    weight_decay: float = 0.0,
    dropout: float = 0.2,
    device: str | None = None,
    pause_event=None,
    on_epoch=None,
    log=None,
) -> dict:
    """用有标签数据微调自编码器：冻结编码器 → 回归头 → 标签（糖度等）。

    ``model_dir`` 为预训练自编码器目录；``label_column``/``label_index`` 指定标签
    字段及其分量（如 ``PredictedValue`` 第 1 分量 = 糖度）。``freeze=True`` 时只
    训练回归头（适合少量数据）。
    """
    def _log(msg: str) -> None:
        logger.info(msg)
        if log:
            log(msg)

    device = get_device(device)
    _log(f"设备: {device}")

    spectrum_column = _load_spectrum_column(model_dir)
    df = pd.read_parquet(data_path)
    if label_column not in df.columns:
        avail = "、".join(str(c) for c in df.columns)
        raise ValueError(
            f"标签字段 {label_column!r} 不在数据中。\n可用列: {avail}\n"
            f"请在微调页签把「标签字段」改为实际存在的列（如 RealValue）。"
        )
    X = parse_spectra(df, spectrum_column)
    y = finetune.parse_label(df, label_column, label_index)
    _log(f"样本数: {len(X)}，光谱维数: {X.shape[1]}（字段: {spectrum_column}）")

    valid = ~np.isnan(y)
    if not valid.all():
        _log(f"标签无效 {int((~valid).sum())} 个，已剔除，保留 {int(valid.sum())} 个")
        X = X[valid]
        y = y[valid]

    if max_samples and len(X) > max_samples:
        idx = np.random.default_rng(0).choice(len(X), max_samples, replace=False)
        X = X[idx]
        y = y[idx]
        _log(f"随机选取 {max_samples} 个有标签样本用于微调")

    pp_file = Path(model_dir) / "preprocess.json"
    msc_ref = None
    if pp_file.exists():
        preprocess = json.loads(pp_file.read_text(encoding="utf-8"))
        if preprocess.get("msc"):
            msc_ref = compute_msc_ref(X, preprocess)
        X = apply_preprocessing(X, preprocess, msc_ref=msc_ref)
        used = ", ".join(k for k, v in preprocess.items() if v)
        _log(f"已应用预处理: {used or '无'}")

    model, _, _ = load_model(model_dir)
    # 用标签数据自身的 mean/std 标准化，抵消批次间整体强度偏移
    # （若复用预训练 mean/std，标签数据会被推到分布外，如 -4σ）
    Xs, ft_mean, ft_std = standardize(X)

    bottleneck = int(model.encoder[4].out_features)

    # 可选：K 折交叉验证评估
    cv_result = None
    if k_folds > 1:
        original_state = {k: v.clone() for k, v in model.state_dict().items()}
        _log(f"开始 {k_folds} 折交叉验证...")
        cv_result = _cross_validate(
            model, Xs, y, k_folds, bottleneck, epochs, batch_size, lr, freeze,
            unfreeze_epochs, unfreeze_lr, weight_decay, dropout, device, _log
        )
        model.load_state_dict(original_state)
        _log(
            f"交叉验证汇总: RMSE={cv_result['mean']['rmse']:.4f}±{cv_result['std']['rmse']:.4f}, "
            f"MAE={cv_result['mean']['mae']:.4f}±{cv_result['std']['mae']:.4f}, "
            f"R²={cv_result['mean']['r2']:.4f}±{cv_result['std']['r2']:.4f}, "
            f"corr={cv_result['mean']['corr']:.4f}±{cv_result['std']['corr']:.4f}"
        )

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(Xs))
    n_train = int(len(Xs) * train_ratio)
    Xtr, Xva = Xs[perm[:n_train]], Xs[perm[n_train:]]
    ytr, yva = y[perm[:n_train]], y[perm[n_train:]]
    _log(f"训练/验证: {len(Xtr)} / {len(Xva)}（训练占比 {train_ratio:.0%}）")

    head = finetune.RegressionHead(bottleneck, dropout=dropout)
    mode = "冻结编码器" if freeze else "全模型微调"
    if freeze and unfreeze_epochs > 0:
        mode = f"两阶段（先冻结 {epochs - unfreeze_epochs} 轮，再解冻 {unfreeze_epochs} 轮）"
    _log(f"回归头: {bottleneck} → 32 → 1（{mode}）")

    history = finetune.train_finetune(
        model, head, Xtr, ytr, epochs, batch_size, lr, Xva, yva, freeze,
        unfreeze_epochs, unfreeze_lr, weight_decay,
        device, pause_event, on_epoch, _log,
    )
    y_mean = history["y_mean"]
    y_std = history["y_std"]

    preds_val = finetune.predict_finetune(model, head, Xva, device, y_mean=y_mean, y_std=y_std)
    metrics = finetune.regression_metrics(yva, preds_val)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (Path(out_dir) if out_dir else FINETUNE_DIR) / timestamp
    finetune.save_finetune(head, out_dir, model, ft_mean, ft_std, label_column, label_index, freeze, bottleneck, y_mean, y_std, dropout)
    (out_dir / "base_model.json").write_text(
        json.dumps({"base_model_dir": str(Path(model_dir).resolve())}, ensure_ascii=False),
        encoding="utf-8",
    )
    # 预测时复用原始预训练模型的字段与预处理
    (out_dir / "spectrum_column.json").write_text(
        json.dumps({"spectrum_column": spectrum_column}, ensure_ascii=False), encoding="utf-8"
    )
    if pp_file.exists():
        (out_dir / "preprocess.json").write_text(
            json.dumps(preprocess, ensure_ascii=False), encoding="utf-8"
        )
    # 保存 MSC 参考光谱，预测时复用（跨批一致性）
    if msc_ref is not None:
        np.save(out_dir / "msc_ref.npy", msc_ref)

    # 保存评估指标，便于后续分析
    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "config": {
                    "label_column": label_column,
                    "label_index": int(label_index),
                    "freeze": bool(freeze),
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "train_ratio": train_ratio,
                    "k_folds": k_folds,
                    "unfreeze_epochs": unfreeze_epochs,
                    "unfreeze_lr": unfreeze_lr,
                    "weight_decay": weight_decay,
                    "dropout": dropout,
                    "bottleneck": bottleneck,
                    "device": str(device),
                    "base_model_dir": str(Path(model_dir).resolve()),
                    "spectrum_column": spectrum_column,
                    "preprocess": preprocess if pp_file.exists() else None,
                },
                "label_stats": {
                    "n": len(X),
                    "mean": float(y.mean()),
                    "std": float(y.std()),
                    "min": float(y.min()),
                    "max": float(y.max()),
                },
                "final_val_metrics": {
                    "n_train": len(Xtr),
                    "n_val": len(Xva),
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "r2": metrics["r2"],
                    "corr": metrics["corr"],
                    "bias": metrics["bias"],
                    "train_loss": history["train"][-1],
                    "val_loss": history["val"][-1] if history["val"] else None,
                },
                "cv": cv_result,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = {
        "samples": len(X),
        "train_samples": len(Xtr),
        "val_samples": len(Xva),
        "device": str(device),
        "freeze": freeze,
        "unfreeze_epochs": unfreeze_epochs,
        "label_column": label_column,
        "label_index": label_index,
        "train_loss": history["train"][-1],
        "val_loss": history["val"][-1] if history["val"] else None,
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "r2": metrics["r2"],
        "corr": metrics["corr"],
        "bias": metrics["bias"],
        "cv": cv_result,
        "out_dir": str(out_dir.resolve()),
        "history": history,
    }
    _log(f"微调完成：验证 RMSE={metrics['rmse']:.4f}，MAE={metrics['mae']:.4f}，R²={metrics['r2']:.4f}")
    _log(f"结果已保存: {out_dir.resolve()}")
    return summary


def run_finetune_predict(
    data_path: str,
    model_dir: str,
    out_path: str | None = None,
    max_samples: int | None = None,
    label_column: str | None = None,
    label_index: int = 0,
    log=None,
) -> dict:
    """用微调后的模型预测标签，保存到 ``predict/`` 或 ``out_path``。

    若提供 ``label_column``，同时读取真实值计算评估指标（RMSE/MAE/R²/相关系数/
    MAPE/偏差/误差分位数），并在 CSV 中输出真实值与误差列，便于评估模型效果。
    """
    def _log(msg: str) -> None:
        logger.info(msg)
        if log:
            log(msg)

    device = get_device()
    _log(f"设备: {device}")
    encoder, head, meta, _, _ = finetune.load_finetune(model_dir)
    encoder.to(device)
    head.to(device)

    # 未显式指定标签字段时，自动使用训练时的标签字段（若数据中存在该列则一并评估）
    if label_column is None:
        label_column = meta.get("label_column")

    spectrum_column = _load_spectrum_column(model_dir)
    df = pd.read_parquet(data_path)
    if max_samples and len(df) > max_samples:
        df = df.head(max_samples)
    X = parse_spectra(df, spectrum_column)
    _log(f"样本数: {len(X)}，光谱维数: {X.shape[1]}（字段: {spectrum_column}）")

    pp_file = Path(model_dir) / "preprocess.json"
    if pp_file.exists():
        preprocess = json.loads(pp_file.read_text(encoding="utf-8"))
        msc_ref = None
        if preprocess.get("msc"):
            ref_file = Path(model_dir) / "msc_ref.npy"
            if ref_file.exists():
                msc_ref = np.load(ref_file)
        X = apply_preprocessing(X, preprocess, msc_ref=msc_ref)
        used = ", ".join(k for k, v in preprocess.items() if v)
        _log(f"已应用预处理: {used or '无'}")

    Xs, _, _ = standardize(X)
    preds = finetune.predict_finetune(
        encoder, head, Xs, device,
        y_mean=meta.get("y_mean"), y_std=meta.get("y_std"),
    )

    # 可选：读取真实标签做评估
    y_true = None
    metrics = None
    scatters = None
    if label_column and label_column in df.columns:
        y_true = finetune.parse_label(df, label_column, label_index)
        valid = ~np.isnan(y_true)
        yt = y_true[valid]
        pt = preds[valid]
        metrics = finetune.extended_metrics(yt, pt)
        scatters = {"y_true": yt.tolist(), "y_pred": pt.tolist()}
        _log(
            f"评估: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, "
            f"R²={metrics['r2']:.4f}, corr={metrics['corr']:.4f}, MAPE={metrics['mape']:.2f}%, "
            f"±0.5度内 {metrics['within_0p5']:.1%}, ±1度内 {metrics['within_1p0']:.1%}"
        )

    out_df = (
        df[["SnNumber"]].copy()
        if "SnNumber" in df.columns
        else pd.DataFrame(index=df.index)
    )
    label_name = f"pred_{meta['label_column']}_{int(meta['label_index']) + 1}"
    out_df[label_name] = preds
    if y_true is not None:
        out_df["true_value"] = y_true
        out_df["error"] = preds - y_true
    if out_path:
        out_path = Path(out_path)
        out_dir = out_path.parent
    else:
        out_dir = PREDICT_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_finetune"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "predictions.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    # 若有真实标签评估，把汇总指标（含 ±0.5 度 / ±1 度内准确度）一并保存到预测目录
    if metrics:
        (out_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    _log(f"预测完成，结果已保存: {out_path.resolve()}")
    return {
        "samples": len(X),
        "mean": float(preds.mean()),
        "std": float(preds.std()),
        "min": float(preds.min()),
        "max": float(preds.max()),
        "out_path": str(out_path.resolve()),
        "out_dir": str(out_dir.resolve()),
        "label_name": label_name,
        "metrics": metrics,
        "scatters": scatters,
    }
