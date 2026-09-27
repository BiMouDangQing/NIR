"""秋月梨 NIR 数据无监督聚类基线。

流程：
    解析光谱字符串列（Abs 或 SpectrumData）→ SNV 散射校正 → PCA 降维
    → KMeans 聚类（遍历 k=2..max_k）→ 输出各簇的糖度/果径分布。

仅依赖 numpy / pandas（已打包在 runtime 中），可离线运行。

用法：
    python -m tools.cluster_baseline <path.parquet> [--column Abs] [--k 3]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import LOG_DIR
from tools.logger import setup_logging

logger = logging.getLogger("NIR.cluster")


def find_default_file() -> str | None:
    """在常见数据目录中自动查找 NIR parquet 文件。"""
    for base in (Path("D:/model/data"), Path("data")):
        if base.exists():
            files = sorted(p for p in base.rglob("*.parquet") if "NIR" in str(p))
            if files:
                return str(files[0])
    return None


def parse_spectra(df: pd.DataFrame, column: str) -> np.ndarray:
    """把逗号分隔的字符串光谱解析为 N×W 浮点矩阵。"""
    mat = df[column].str.split(",", expand=True).astype(float).to_numpy()
    return mat


def snv(X: np.ndarray) -> np.ndarray:
    """标准正态变量变换（逐行）：(x - mean) / std。"""
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (X - mean) / std


def pca(X: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    """中心化 + SVD 实现 PCA，返回 (降维结果, 奇异值)。"""
    Xc = X - X.mean(axis=0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:n_components].T, S


def kmeans(
    X: np.ndarray, k: int, n_init: int = 10, max_iter: int = 100, seed: int = 0
) -> tuple[np.ndarray, float, np.ndarray]:
    """numpy 版 KMeans（多次初始化取惯性最小者）。"""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    best_centers: np.ndarray | None = None
    best_inertia = np.inf
    best_labels: np.ndarray | None = None

    for _ in range(n_init):
        centers = X[rng.choice(n, k, replace=False)].copy()
        for _ in range(max_iter):
            dist = (
                (X**2).sum(1, keepdims=True)
                + (centers**2).sum(1)[None, :]
                - 2 * X @ centers.T
            )
            labels = dist.argmin(1)
            new_centers = np.empty_like(centers)
            for j in range(k):
                members = X[labels == j]
                new_centers[j] = members.mean(0) if len(members) else centers[j]
            if np.allclose(centers, new_centers):
                centers = new_centers
                break
            centers = new_centers

        dist = (
            (X**2).sum(1, keepdims=True)
            + (centers**2).sum(1)[None, :]
            - 2 * X @ centers.T
        )
        labels = dist.argmin(1)
        inertia = float(dist.min(1).sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best_centers = centers.copy()
            best_labels = labels.copy()

    assert best_centers is not None and best_labels is not None
    return best_centers, best_inertia, best_labels


def silhouette(X: np.ndarray, labels: np.ndarray, sample: int = 3000, seed: int = 0) -> float:
    """在子样本上估算轮廓系数（O(n^2)，全量太慢）。"""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(sample, len(X)), replace=False)
    Xs, ls = X[idx], labels[idx]
    D = ((Xs[:, None, :] - Xs[None, :, :]) ** 2).sum(-1)
    scores = np.zeros(len(Xs))
    uniq = np.unique(ls)
    for i in range(len(Xs)):
        same = D[i, ls == ls[i]]
        a = same.sum() / max(len(same) - 1, 1)
        others = [D[i, ls == c].mean() for c in uniq if c != ls[i]]
        b = min(others) if others else 0.0
        scores[i] = (b - a) / max(a, b, 1e-12)
    return float(scores.mean())


def report_clusters(labels: np.ndarray, sugar: pd.Series, diameter: pd.Series) -> None:
    """输出各簇的样本数、糖度、果径统计。"""
    print(f"\n{'簇':>4} {'样本数':>7} {'糖度均值':>8} {'糖度范围':>14} {'果径均值':>8} {'果径范围':>10}")
    for c in sorted(set(labels)):
        s = sugar[labels == c]
        d = diameter[labels == c]
        print(
            f"{c:>4} {len(s):>7} {s.mean():>8.2f} "
            f"[{s.min():>5.1f},{s.max():>5.1f}] {d.mean():>8.1f} "
            f"[{d.min():>4.0f},{d.max():>4.0f}]"
        )


def main() -> None:
    setup_logging(LOG_DIR)
    parser = argparse.ArgumentParser(description="秋月梨 NIR 无监督聚类基线")
    parser.add_argument("path", nargs="?", default=None, help="parquet 文件路径（缺省自动搜索）")
    parser.add_argument("--column", default="Abs", help="光谱列名，默认 Abs（也可用 SpectrumData）")
    parser.add_argument("--k", type=int, default=None, help="聚类数，缺省遍历 2~max_k")
    parser.add_argument("--max-k", type=int, default=8, help="遍历的最大聚类数，默认 8")
    parser.add_argument("--pca", type=int, default=10, help="PCA 主成分数，默认 10")
    parser.add_argument("--out", default=None, help="可选：保存聚类结果的 CSV 路径")
    args = parser.parse_args()

    path = args.path or find_default_file()
    if not path:
        print("[错误] 未找到数据文件，请显式传入路径")
        return

    logger.info(f"加载数据: {path}")
    df = pd.read_parquet(path)
    print(f"数据: {df.shape[0]} 行 × {df.shape[1]} 列")

    X = parse_spectra(df, args.column)
    print(f"光谱矩阵: {X.shape[0]} × {X.shape[1]}（列: {args.column}）")

    # 糖度 = PredictedValue 第 1 个数值；果径 = Diameter
    sugar = pd.to_numeric(df["PredictedValue"].str.split(",").str[0], errors="coerce")
    diameter = pd.to_numeric(df["Diameter"], errors="coerce")
    print(f"糖度范围: {sugar.min():.2f} ~ {sugar.max():.2f}（均值 {sugar.mean():.2f}）")
    print(f"果径范围: {diameter.min():.0f} ~ {diameter.max():.0f} mm")

    # 预处理
    Xs = snv(X)
    logger.info("SNV 预处理完成")

    # PCA 降维
    Xp, S = pca(Xs, args.pca)
    explained = (S**2) / (S**2).sum()
    print(f"\nPCA 前 {args.pca} 个主成分累计解释方差: {explained[:args.pca].sum():.1%}")

    # 聚类
    ks = [args.k] if args.k else list(range(2, args.max_k + 1))
    results = {}
    print(f"\n{'k':>3} {'惯性(inertia)':>15} {'轮廓系数(抽样)':>14}")
    best_k, best_sil = None, -1.0
    for k in ks:
        _, inertia, labels = kmeans(Xp, k)
        sil = silhouette(Xp, labels)
        results[k] = labels
        print(f"{k:>3} {inertia:>15.2f} {sil:>14.3f}")
        if sil > best_sil:
            best_sil, best_k = sil, k

    # 输出最佳 k 的簇统计
    pick = args.k or best_k
    logger.info(f"最优聚类数 k={pick}（轮廓系数 {best_sil:.3f}）")
    report_clusters(results[pick], sugar, diameter)

    # 可选保存
    if args.out:
        out_df = df[["SnNumber", "Variety", "Date", "Diameter"]].copy()
        out_df["sugar"] = sugar
        out_df["cluster"] = results[pick]
        out_df.to_csv(args.out, index=False, encoding="utf-8-sig")
        logger.info(f"聚类结果已保存: {args.out}")


if __name__ == "__main__":
    main()
