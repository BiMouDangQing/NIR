"""降维（PCA）：把高维特征降到 2/3 维，用于可视化。"""
from __future__ import annotations

import numpy as np


def pca(X: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """中心化 + SVD 的 PCA，返回 (低维坐标, 各主成分解释方差占比)。"""
    Xc = X - X.mean(axis=0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    coords = Xc @ Vt[:n_components].T
    explained = (S**2) / (S**2).sum()
    return coords, explained
