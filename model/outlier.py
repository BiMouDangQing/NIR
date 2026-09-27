"""训练前的离群点剔除：SNV → PCA → KMeans → 按簇内距离分位数剔除。

在自监督预训练之前，先对光谱做无监督聚类，把距离簇中心过远
（超过给定分位数）的离群样本剔除，避免污染特征学习。
"""
from __future__ import annotations

import numpy as np

from model.preprocess import snv
from model.reduce import pca


def kmeans(
    X: np.ndarray, k: int, n_init: int = 10, max_iter: int = 100, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """numpy 版 KMeans（多次初始化取惯性最小者），返回 (簇中心, 标签)。"""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    k = max(1, min(k, n))
    best_centers: np.ndarray | None = None
    best_labels: np.ndarray | None = None
    best_inertia = np.inf

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
    return best_centers, best_labels


def silhouette(
    X: np.ndarray, labels: np.ndarray, sample: int = 3000, seed: int = 0
) -> float:
    """在子样本上估算轮廓系数（O(n²)，全量太慢）。"""
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


def choose_k(X: np.ndarray, max_k: int = 8, sample: int = 3000, seed: int = 0) -> int:
    """遍历 2..max_k，返回轮廓系数最高的聚类数。"""
    best_k, best_sil = 2, -1.0
    for k in range(2, min(max_k, len(X)) + 1):
        _, labels = kmeans(X, k, seed=seed)
        sil = silhouette(X, labels, sample=sample, seed=seed)
        if sil > best_sil:
            best_sil, best_k = sil, k
    return best_k


def cluster_outlier_mask(
    X: np.ndarray,
    k: int = 5,
    percentile: float = 99.0,
    pca_dim: int = 10,
    auto_k: bool = False,
    max_k: int = 8,
    seed: int = 0,
    log=None,
) -> np.ndarray:
    """返回保留样本的布尔掩码。

    流程：SNV 散射校正 → PCA 降维 → KMeans 聚类 →
    计算每个样本到所属簇中心的欧氏距离，距离超过 ``percentile`` 分位的判为离群。

    ``auto_k`` 为 True 时用轮廓系数自动选 k（遍历 2..max_k）。
    """
    n = X.shape[0]
    if n < 20:
        return np.ones(n, dtype=bool)  # 样本太少，不剔除

    Xs = snv(X)
    dim = min(pca_dim, X.shape[1], n)
    Xp, _ = pca(Xs, dim)
    if auto_k:
        k = choose_k(Xp, max_k=max_k, seed=seed)
        if log:
            log(f"自动选定聚类数 k={k}")
    else:
        k = max(2, min(int(k), n // 5))  # 每簇至少约 5 个样本
    centers, labels = kmeans(Xp, k, seed=seed)
    dist = np.linalg.norm(Xp - centers[labels], axis=1)
    thresh = np.percentile(dist, percentile)
    return dist <= thresh
