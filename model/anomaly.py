"""异常检测：基于重构误差的分位数阈值与 TopN 排序。"""
from __future__ import annotations

import numpy as np


def threshold(err: np.ndarray, percentile: float = 99.0) -> float:
    """返回异常阈值（重构误差的百分位数）。"""
    return float(np.percentile(err, percentile))


def detect(err: np.ndarray, percentile: float = 99.0) -> tuple[np.ndarray, float]:
    """返回 (异常标记 bool 数组, 阈值)。"""
    t = threshold(err, percentile)
    return err > t, t


def top_anomalies(err: np.ndarray, n: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """返回重构误差最大的 n 个样本的下标与误差值。"""
    idx = np.argsort(err)[::-1][:n]
    return idx, err[idx]
