"""分布一致性核查（域偏移 / domain shift 检测）。

自监督预训练用大量无标签光谱，有监督微调用少量带真实标签光谱。若两批数据
分布不一致（预训练覆盖不到标签样本区域），预训练特征对下游任务帮助有限。

本模块对两批光谱做 6 项检查，返回结构化结果，供 GUI（后台线程）与
``tools/check_domain_shift.py`` 命令行共用：

  1. 逐波长均值 / 标准差对比（平均光谱是否重合）
  2. PCA 联合投影（两批质心在低维空间的距离）
  3. 马氏距离（标签样本是否落在预训练分布内部）
  4. KNN 最近邻覆盖度（标签样本离预训练样本有多远）
  5. 领域判别器（能否把两批区分开，AUC 越接近 0.5 越同分布）
  6. 标签分布对比（预训练 PredictedValue vs 标签 RealValue）

仅依赖 numpy / pandas / pyarrow，无 sklearn。
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("NIR.model")

# 绘图时预训练样本的下采样上限（避免散点过多拖慢渲染）
_PLOT_MAX_PRETRAIN_POINTS = 5000
# 平均光谱曲线下采样到的点数
_PLOT_WAVE_POINTS = 256


def parse_spectra(df: pd.DataFrame, column: str = "SpectrumData") -> np.ndarray:
    """把逗号分隔的光谱字符串解析成 (n, d) 浮点矩阵。"""
    arr = df[column].astype(str).str.split(",", expand=False)
    X = np.array([[float(v) for v in row] for row in arr], dtype=np.float32)
    return X


def parse_label(df: pd.DataFrame, column: str, index: int = 0) -> np.ndarray:
    """解析标签列：逗号分隔多分量取第 index 个，否则直接转 float。"""
    vals = df[column].astype(str)
    if vals.str.contains(",").any():
        y = np.array([float(v.split(",")[index]) for v in vals], dtype=np.float64)
    else:
        y = pd.to_numeric(vals, errors="coerce").to_numpy(dtype=np.float64)
    return y


def pca_fit(
    X: np.ndarray, n_components: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """列标准化后做 PCA（协方差特征分解），返回 (得分, 解释方差占比, 均值, 标准差)。"""
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    Xs = (X - mean) / std
    cov = Xs.T @ Xs / Xs.shape[0]
    evals, evecs = np.linalg.eigh(cov)  # 升序
    order = np.argsort(evals)[::-1]
    V = evecs[:, order][:, :n_components]
    scores = Xs @ V
    explained = evals[order][:n_components] / evals.sum()
    return scores, explained, V, mean, std


def mahalanobis_dist(pc_pretrain: np.ndarray, pc_label: np.ndarray) -> np.ndarray:
    """用预训练样本在 PC 空间的均值/协方差，算每个标签样本的马氏距离平方。"""
    mu = pc_pretrain.mean(axis=0)
    cov = np.cov(pc_pretrain.T) + np.eye(pc_pretrain.shape[1]) * 1e-6
    inv = np.linalg.inv(cov)
    diff = pc_label - mu
    return np.einsum("ij,jk,ik->i", diff, inv, diff)


def _min_dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """A 中每行到 B 中所有行的最小欧氏距离。"""
    dists = []
    for a in A:
        d = ((B - a) ** 2).sum(1)
        dists.append(float(d.min()))
    return np.array(dists)


def knn_cover(pc_pretrain: np.ndarray, pc_label: np.ndarray, n_ref: int = 8000) -> dict:
    """最近邻覆盖度：标签样本到预训练样本的最小距离 vs 预训练内部自最小距离。"""
    rng = np.random.default_rng(0)
    ref = pc_pretrain[rng.choice(len(pc_pretrain), min(n_ref, len(pc_pretrain)), replace=False)]
    d_label = _min_dist(pc_label, ref)
    queries = pc_pretrain[rng.choice(len(pc_pretrain), min(2000, len(pc_pretrain)), replace=False)]
    d_self = _min_dist(queries, ref)
    return {
        "label_to_pretrain_mean": float(d_label.mean()),
        "pretrain_self_mean": float(d_self.mean()),
        "ratio": float(d_label.mean() / (d_self.mean() + 1e-12)),
    }


def logistic_regression(
    X: np.ndarray, y: np.ndarray, iters: int = 2000, lr: float = 0.5, l2: float = 1e-3
) -> np.ndarray:
    """numpy 逻辑回归（梯度下降），返回权重向量（含偏置）。"""
    n, d = X.shape
    w = np.zeros(d + 1)
    Xb = np.hstack([X, np.ones((n, 1))])
    for _ in range(iters):
        logits = Xb @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
        grad = Xb.T @ (p - y) / n + l2 * w
        w -= lr * grad
    return w


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """用 Mann-Whitney U 统计量计算 AUC（0.5=不可区分，1=完全可分）。"""
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    n_pos, n_neg = len(pos), len(neg)
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    rank_pos = ranks[:n_pos].sum()
    return float((rank_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def summarize(vals: np.ndarray) -> dict:
    """统计分位数摘要。"""
    return {
        "n": int(len(vals)),
        "mean": float(np.nanmean(vals)),
        "std": float(np.nanstd(vals)),
        "min": float(np.nanmin(vals)),
        "p25": float(np.nanpercentile(vals, 25)),
        "p50": float(np.nanpercentile(vals, 50)),
        "p75": float(np.nanpercentile(vals, 75)),
        "p95": float(np.nanpercentile(vals, 95)),
        "max": float(np.nanmax(vals)),
    }


def _downsample_curve(y: np.ndarray, n_out: int) -> tuple[list[int], list[float]]:
    """把曲线下采样到 n_out 个点，返回 (x 序号, y 值)。"""
    if len(y) <= n_out:
        idx = np.arange(1, len(y) + 1)
        return idx.tolist(), y.tolist()
    idx = np.linspace(0, len(y) - 1, n_out).astype(int)
    return (idx + 1).tolist(), y[idx].tolist()


def _histogram(vals: np.ndarray, bins: int = 20) -> tuple[list[float], list[int]]:
    """直方图，返回 (桶左边界列表, 各桶计数)。"""
    counts, edges = np.histogram(vals, bins=bins)
    return edges.tolist(), counts.tolist()


def run_domain_shift(
    pretrain_path: str,
    label_path: str,
    spectrum_column: str = "SpectrumData",
    preprocess: dict | None = None,
    max_pretrain: int = 0,
    n_comp: int = 20,
    log=None,
) -> dict:
    """执行 6 项分布核查，返回结构化结果（含绘图数据）。

    ``preprocess`` 形如 {"smooth": True, "msc": True}，对比前对两批应用同一预处理。
    """
    def _log(msg: str) -> None:
        logger.info(msg)
        if log:
            log(msg)

    from model.preprocess import apply_preprocessing

    _log(f"加载预训练数据: {pretrain_path}")
    df_p = pd.read_parquet(pretrain_path)
    df_l = pd.read_parquet(label_path)
    if max_pretrain and len(df_p) > max_pretrain:
        df_p = df_p.sample(n=max_pretrain, random_state=0)
    Xp = parse_spectra(df_p, spectrum_column)
    Xl = parse_spectra(df_l, spectrum_column)

    if preprocess and any(preprocess.values()):
        Xp = apply_preprocessing(Xp, preprocess)
        Xl = apply_preprocessing(Xl, preprocess)
        used = ", ".join(k for k, v in preprocess.items() if v)
        _log(f"已应用预处理: {used}")

    report: dict = {
        "pretrain_file": pretrain_path,
        "label_file": label_path,
        "spectrum_column": spectrum_column,
        "preprocess": {k: bool(v) for k, v in preprocess.items()} if preprocess else None,
        "samples": {"pretrain": int(len(Xp)), "label": int(len(Xl)), "dim": int(Xp.shape[1])},
    }
    _log(f"预训练 {len(Xp)} 样本，标签 {len(Xl)} 样本，光谱 {Xp.shape[1]} 维")

    # 1. 逐波长均值对比
    mp, sp = Xp.mean(0), Xp.std(0)
    ml, sl = Xl.mean(0), Xl.std(0)
    corr_mean = float(np.corrcoef(mp, ml)[0, 1])
    mean_abs_diff = float(np.abs(mp - ml).mean())
    mean_max_diff = float(np.abs(mp - ml).max())
    report["mean_spectrum"] = {
        "corr": corr_mean,
        "mean_abs_diff": mean_abs_diff,
        "max_abs_diff": mean_max_diff,
        "pretrain_global": {"mean": float(mp.mean()), "std": float(sp.mean())},
        "label_global": {"mean": float(ml.mean()), "std": float(sl.mean())},
    }
    _log(f"1. 平均光谱相关系数 {corr_mean:.4f}，逐波长均值绝对差 {mean_abs_diff:.1f}")
    _log(f"   预训练全局均值 {mp.mean():.1f} vs 标签 {ml.mean():.1f}")

    # 2. PCA 联合投影
    n_comp_use = min(n_comp, Xp.shape[1])
    Xall = np.vstack([Xp, Xl])
    scores, explained, _, _, _ = pca_fit(Xall, n_comp_use)
    pc_p = scores[: len(Xp)]
    pc_l = scores[len(Xp):]
    centroid_dist = float(np.linalg.norm(pc_p.mean(0) - pc_l.mean(0)))
    spread_p = float(np.mean(np.linalg.norm(pc_p - pc_p.mean(0), axis=1)))
    spread_l = float(np.mean(np.linalg.norm(pc_l - pc_l.mean(0), axis=1)))
    norm_dist = float(centroid_dist / ((spread_p + spread_l) / 2 + 1e-12))
    report["pca"] = {
        "explained_var_2d": float(explained[:2].sum()),
        "centroid_distance": centroid_dist,
        "normalized_distance": norm_dist,
    }
    _log(f"2. PCA 前 2 主成分解释 {explained[:2].sum():.1%}，质心归一化距离 {norm_dist:.2f}")

    # 3. 马氏距离
    d2 = mahalanobis_dist(pc_p, pc_l)
    chi2_95 = float(np.percentile(np.random.chisquare(n_comp_use, 200000), 95))
    out_ratio = float((d2 > chi2_95).mean())
    report["mahalanobis"] = {
        "median": float(np.median(d2)),
        "p95": float(np.percentile(d2, 95)),
        "chi2_95": chi2_95,
        "outlier_ratio": out_ratio,
    }
    _log(f"3. 马氏距离超 95% 边界的标签样本比例 {out_ratio:.1%}（同分布应 ≈5%）")

    # 4. KNN 覆盖度
    knn = knn_cover(pc_p, pc_l)
    report["knn"] = knn
    _log(f"4. KNN 覆盖度比值 {knn['ratio']:.1f}（≈1 覆盖良好）")

    # 5. 领域判别器
    rng = np.random.default_rng(0)
    n_neg = min(8000, len(pc_p))
    idx_p = rng.choice(len(pc_p), n_neg, replace=False)
    n_pos = len(pc_l)
    Xd = np.vstack([pc_p[idx_p], pc_l])
    yd = np.concatenate([np.zeros(n_neg), np.ones(n_pos)])
    m, s = Xd.mean(0), Xd.std(0) + 1e-8
    Xds = (Xd - m) / s
    perm = rng.permutation(len(Xds))
    cut = int(len(Xds) * 0.8)
    tr, te = perm[:cut], perm[cut:]
    w = logistic_regression(Xds[tr], yd[tr])
    Xte = np.hstack([Xds[te], np.ones((len(te), 1))])
    auc = roc_auc(yd[te], Xte @ w)
    report["domain_classifier_auc"] = auc
    _log(f"5. 领域判别器 AUC {auc:.3f}（≈0.5 同分布，>0.8 明显差异）")

    # 6. 标签分布对比
    label_dist: dict = {}
    yp = None
    yl = None
    if "PredictedValue" in df_p.columns:
        yp = parse_label(df_p, "PredictedValue", 0)
        label_dist["predicted_value"] = summarize(yp)
    if "RealValue" in df_l.columns:
        yl = parse_label(df_l, "RealValue", 0)
        label_dist["real_value"] = summarize(yl)
    report["label_dist"] = label_dist
    if yp is not None and yl is not None:
        lo = max(float(np.nanmin(yp)), float(np.nanmin(yl)))
        hi = min(float(np.nanmax(yp)), float(np.nanmax(yl)))
        report["label_overlap"] = [lo, hi]
        _log(f"6. 仪器预测糖度 {np.nanmean(yp):.2f} vs 真实糖度 {np.nanmean(yl):.2f}，重叠范围 [{lo:.1f}, {hi:.1f}]")

    # 绘图数据
    plot: dict = {}
    wave_x, wave_p = _downsample_curve(mp, _PLOT_WAVE_POINTS)
    _, wave_l = _downsample_curve(ml, _PLOT_WAVE_POINTS)
    plot["mean_wave"] = {"x": wave_x, "pretrain": wave_p, "label": wave_l}

    # PCA 散点（预训练下采样）
    n_show = min(_PLOT_MAX_PRETRAIN_POINTS, len(pc_p))
    idx_show = rng.choice(len(pc_p), n_show, replace=False)
    plot["pca"] = {
        "pretrain_x": pc_p[idx_show, 0].tolist(),
        "pretrain_y": pc_p[idx_show, 1].tolist(),
        "label_x": pc_l[:, 0].tolist(),
        "label_y": pc_l[:, 1].tolist(),
    }

    if yp is not None:
        plot["pred_hist"] = _histogram(yp)
    if yl is not None:
        plot["real_hist"] = _histogram(yl)
    report["plot"] = plot
    return report
