"""批次校正（batch correction）：消除不同批次之间的系统性光谱差异。

两种方法：
  1. mean_variance_align —— 均值-方差对齐：把每批逐波长 z-score 后映射到参考
     批次的均值/方差尺度，消除批间的加性（均值）与乘性（方差）差异。
  2. combat —— ComBat（Johnson et al. 2007 经验贝叶斯批次校正）：逐特征估计
     批次的加性效应 γ 与乘性效应 δ，并用经验贝叶斯先验收缩，再校正回原尺度。

输入都是「样本数 × 波长数」的光谱矩阵列表（各批样本数可不同）。
"""
from __future__ import annotations

import numpy as np


def mean_variance_align(batches: list[np.ndarray], ref_idx: int = 0) -> list[np.ndarray]:
    """均值-方差对齐：各批逐波长对齐到参考批次的均值/方差尺度。

    对每批 X：``(X - mean_batch) / std_batch * std_ref + mean_ref``，
    其中 mean/std 均为逐波长统计。
    """
    ref = batches[ref_idx]
    ref_mean = ref.mean(axis=0)
    ref_std = ref.std(axis=0) + 1e-8
    out: list[np.ndarray] = []
    for X in batches:
        m = X.mean(axis=0)
        s = X.std(axis=0) + 1e-8
        out.append((X - m) / s * ref_std + ref_mean)
    return out


def _estimate_lambda_theta(delta_sq: np.ndarray) -> tuple[float, float]:
    """用 InverseGamma 分布的矩估计求 (λ, θ)。

    mean = θ/(λ-1)，var = θ²/((λ-1)²(λ-2))，解出 λ、θ。
    """
    mean = float(delta_sq.mean())
    var = float(delta_sq.var())
    if var < 1e-12:
        # 方差接近 0：退化为强先验（λ 大，θ 大）
        return 100.0, mean * 99.0
    lam = mean * mean / var + 2.0
    theta = mean * (lam - 1.0)
    return lam, theta


def combat(batches: list[np.ndarray], n_iter: int = 20) -> list[np.ndarray]:
    """ComBat 经验贝叶斯批次校正（无协变量）。

    - 先逐波长 z-score 到合并分布；
    - 对每个批次、每个波长估计加性效应 γ 与乘性效应 δ；
    - 用经验贝叶斯先验（γ 正态、δ² 逆伽马）收缩，迭代求解；
    - 校正回原尺度。
    """
    Xall = np.vstack(batches)
    g_mean = Xall.mean(axis=0)
    g_std = Xall.std(axis=0) + 1e-8
    Z = (Xall - g_mean) / g_std

    n_batch = len(batches)
    sizes = [len(b) for b in batches]
    offsets = np.cumsum([0] + sizes)
    n_feat = Xall.shape[1]

    # 1. 每个批次、每个波长的矩估计
    gamma_hat = np.zeros((n_batch, n_feat))
    delta_hat_sq = np.zeros((n_batch, n_feat))
    for i in range(n_batch):
        Zi = Z[offsets[i] : offsets[i + 1]]
        gamma_hat[i] = Zi.mean(axis=0)
        delta_hat_sq[i] = Zi.var(axis=0, ddof=1) + 1e-8

    # 2. 经验贝叶斯先验超参数
    gamma_bar = gamma_hat.mean(axis=1)  # 对波长平均
    delta_bar_sq = delta_hat_sq.mean(axis=1)
    tau_sq_bar = np.maximum(
        0.0, gamma_hat.var(axis=1) - delta_bar_sq / np.array(sizes)
    )
    lam_bar = np.zeros(n_batch)
    theta_bar = np.zeros(n_batch)
    for i in range(n_batch):
        lam_bar[i], theta_bar[i] = _estimate_lambda_theta(delta_hat_sq[i])

    # 3. 迭代收缩 γ* 与 δ*
    gamma_star = gamma_hat.copy()
    delta_star_sq = delta_hat_sq.copy()
    for _ in range(n_iter):
        # 更新 γ*（用上一步 δ*）
        for i in range(n_batch):
            num = sizes[i] * tau_sq_bar[i] * gamma_hat[i] + delta_star_sq[i] * gamma_bar[i]
            den = sizes[i] * tau_sq_bar[i] + delta_star_sq[i]
            gamma_star[i] = num / np.maximum(den, 1e-8)
        # 更新 δ*（用新 γ*）
        for i in range(n_batch):
            Zi = Z[offsets[i] : offsets[i + 1]]
            ss = ((Zi - gamma_star[i][None, :]) ** 2).sum(axis=0)
            delta_star_sq[i] = (theta_bar[i] + 0.5 * ss) / (
                sizes[i] / 2.0 + lam_bar[i] - 1.0
            )
            delta_star_sq[i] = np.maximum(delta_star_sq[i], 1e-8)

    # 4. 校正回原尺度
    corrected: list[np.ndarray] = []
    for i in range(n_batch):
        Zi = Z[offsets[i] : offsets[i + 1]]
        Yc = g_std * (Zi - gamma_star[i][None, :]) / np.sqrt(delta_star_sq[i][None, :]) + g_mean
        corrected.append(Yc)
    return corrected
