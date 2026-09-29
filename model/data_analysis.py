"""多批数据综合分析：分布统计、离群（异常）检测、两两相似性。

针对预训练数据、微调数据、预测数据（可任意多批），对每批做：
  1. 分布统计：样本数、光谱全局均值、逐波长标准差均值、标签分布（如有 RealValue）
  2. 离群检测：SNV → PCA → KMeans 聚类，按簇内距离分位判定离群样本
  3. 两两相似性：PCA 质心归一化距离（越小越相似）+ 领域判别器 AUC（≈0.5 越相似）

复用 model.domain_shift / model.outlier / model.preprocess，无额外依赖。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from model.domain_shift import (
    logistic_regression,
    parse_label,
    parse_spectra,
    pca_fit,
    roc_auc,
    summarize,
)
from model.outlier import cluster_outlier_mask
from model.preprocess import apply_preprocessing

logger = logging.getLogger("NIR.model")

# 离群检测时的抽样上限（避免超大预训练集拖慢聚类）
_OUTLIER_MAX_SAMPLES = 20000
# 相似性判别器的负类抽样上限
_AUC_MAX_SAMPLES = 8000


def _pair_similarity(A: np.ndarray, B: np.ndarray, n_comp: int = 20) -> dict:
    """两批数据的相似性：PCA 质心归一化距离 + 判别器 AUC。"""
    n_comp = min(n_comp, A.shape[1], len(A) + len(B))
    Xall = np.vstack([A, B])
    scores, _, _, _, _ = pca_fit(Xall, n_comp)
    pc_a, pc_b = scores[: len(A)], scores[len(A):]
    centroid = float(np.linalg.norm(pc_a.mean(0) - pc_b.mean(0)))
    spread_a = float(np.mean(np.linalg.norm(pc_a - pc_a.mean(0), axis=1)))
    spread_b = float(np.mean(np.linalg.norm(pc_b - pc_b.mean(0), axis=1)))
    norm = float(centroid / ((spread_a + spread_b) / 2 + 1e-12))

    # 领域判别器：逻辑回归区分来源，留出 20% 测试
    rng = np.random.default_rng(0)
    n_a = min(_AUC_MAX_SAMPLES, len(pc_a))
    idx_a = rng.choice(len(pc_a), n_a, replace=False)
    n_b = len(pc_b)
    Xd = np.vstack([pc_a[idx_a], pc_b])
    yd = np.concatenate([np.zeros(n_a), np.ones(n_b)])
    m, s = Xd.mean(0), Xd.std(0) + 1e-8
    Xds = (Xd - m) / s
    perm = rng.permutation(len(Xds))
    cut = int(len(Xds) * 0.8)
    tr, te = perm[:cut], perm[cut:]
    w = logistic_regression(Xds[tr], yd[tr])
    Xte = np.hstack([Xds[te], np.ones((len(te), 1))])
    auc = roc_auc(yd[te], Xte @ w)
    return {"centroid_norm": norm, "auc": auc}


def run_data_analysis(
    datasets: dict,
    spectrum_column: str = "SpectrumData",
    preprocess: dict | None = None,
    outlier_percentile: float = 99.0,
    log=None,
) -> dict:
    """对多批数据做综合分析。

    ``datasets`` 形如 {"预训练": path1, "微调": path2, "预测": path3}。
    返回 {"summary": {name: {...}}, "pairwise": [{a, b, centroid_norm, auc}]}。
    """
    def _log(msg: str) -> None:
        logger.info(msg)
        if log:
            log(msg)

    # 1. 加载
    loaded: dict = {}
    for name, path in datasets.items():
        df = pd.read_parquet(path)
        X = parse_spectra(df, spectrum_column)
        if preprocess and any(preprocess.values()):
            X = apply_preprocessing(X, preprocess)
        loaded[name] = {"X": X, "df": df}
        _log(f"已加载 {name}: {len(X)} 样本，光谱 {X.shape[1]} 维")

    # 2. 每批分布统计 + 离群检测
    summary: dict = {}
    for name, d in loaded.items():
        X = d["X"]
        df = d["df"]
        # 离群检测（抽样提速）
        if len(X) > _OUTLIER_MAX_SAMPLES:
            idx = np.random.default_rng(0).choice(len(X), _OUTLIER_MAX_SAMPLES, replace=False)
            X_out = X[idx]
        else:
            X_out = X
        keep = cluster_outlier_mask(
            X_out, k=5, percentile=outlier_percentile, auto_k=False,
            log=lambda m: _log(f"  [{name}] {m}"),
        )
        n_out = int((~keep).sum())
        item: dict = {
            "n": int(len(X)),
            "dim": int(X.shape[1]),
            "spectrum_mean": float(X.mean()),
            "spectrum_std_mean": float(X.std(0).mean()),
            "outliers": n_out,
            "outlier_ratio": float(n_out / len(X_out)) if len(X_out) else 0.0,
            "label_stats": None,
        }
        if "RealValue" in df.columns:
            y = parse_label(df, "RealValue", 0)
            valid = y[~np.isnan(y)]
            if len(valid):
                item["label_stats"] = summarize(valid)
        summary[name] = item
        _log(
            f"[{name}] n={len(X)}, 光谱均值={X.mean():.1f}, "
            f"离群={n_out}（{item['outlier_ratio']:.1%}）"
        )

    # 3. 两两相似性
    names = list(loaded.keys())
    pairwise: list = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            res = _pair_similarity(loaded[names[i]]["X"], loaded[names[j]]["X"])
            res["a"] = names[i]
            res["b"] = names[j]
            pairwise.append(res)
            _log(
                f"相似性 {names[i]} vs {names[j]}: "
                f"质心距离={res['centroid_norm']:.2f}, 判别器 AUC={res['auc']:.3f}"
            )

    return {"summary": summary, "pairwise": pairwise}
