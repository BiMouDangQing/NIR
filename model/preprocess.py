"""光谱预处理：SNV / MSC / Savitzky-Golay 平滑 / 一阶二阶导数。

供自监督预训练在建模前可选启用，按固定顺序叠加：
平滑 → SNV → MSC → 一阶导数 → 二阶导数。
"""
from __future__ import annotations

import numpy as np


def savgol(X: np.ndarray, window: int = 7, poly: int = 2) -> np.ndarray:
    """Savitzky-Golay 平滑（默认窗口 7、二阶多项式）。"""
    if window % 2 == 0:
        window += 1
    half = window // 2
    x = np.arange(-half, half + 1, dtype=float)
    A = np.vander(x, poly + 1, increasing=True)
    target = np.zeros(window)
    target[half] = 1.0
    # 最小二乘拟合多项式系数，再还原为中心点的平滑卷积核（window 个权重）
    coeffs, *_ = np.linalg.lstsq(A, target, rcond=None)
    kernel = A @ coeffs
    out = np.empty_like(X)
    for i in range(len(X)):
        out[i] = np.convolve(X[i], kernel, mode="same")
    return out


def snv(X: np.ndarray) -> np.ndarray:
    """标准正态变量变换：逐样本减去均值除以标准差，消除散射与尺度差异。"""
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True) + 1e-8
    return (X - mean) / std


def msc(X: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
    """多元散射校正：逐样本对参考光谱做线性回归，校正斜率与截距。

    ``ref`` 为参考光谱（默认用输入数据的平均光谱）。跨批使用时必须传入同一
    参考光谱（训练时确定、预测时复用），否则各批用各自平均光谱会导致跨批
    不可比。
    """
    if ref is None:
        ref = X.mean(axis=0)
    ref_mean = ref.mean()
    ref_var = ((ref - ref_mean) ** 2).sum() + 1e-8
    out = np.empty_like(X)
    for i in range(len(X)):
        x = X[i]
        x_mean = x.mean()
        b1 = float(((x - x_mean) * (ref - ref_mean)).sum() / ref_var)
        b0 = x_mean - b1 * ref_mean
        out[i] = (x - b0) / (b1 + 1e-8)
    return out


def compute_msc_ref(X: np.ndarray, steps: dict | None = None) -> np.ndarray:
    """计算 MSC 参考光谱：应用 MSC 之前的预处理步骤后，取逐波长平均。

    用于训练时确定参考光谱并保存，预测时复用同一参考（避免各批各自平均）。
    """
    out = X
    steps = steps or {}
    if steps.get("smooth"):
        out = savgol(out)
    if steps.get("snv"):
        out = snv(out)
    return out.mean(axis=0)


def derivative(X: np.ndarray, order: int = 1) -> np.ndarray:
    """按波长求导（np.gradient），order=1 一阶、order=2 二阶。"""
    out = X
    for _ in range(order):
        out = np.gradient(out, axis=1)
    return out


def apply_preprocessing(X: np.ndarray, steps: dict | None = None, msc_ref: np.ndarray | None = None) -> np.ndarray:
    """按固定顺序应用勾选的预处理。

    ``steps`` 形如 {"smooth": True, "snv": True, "msc": False, "deriv1": False, "deriv2": False}。
    ``msc_ref`` 为 MSC 的参考光谱（跨批时传入训练时保存的参考，避免各批各自平均）。
    """
    if not steps:
        return X
    out = X
    if steps.get("smooth"):
        out = savgol(out)
    if steps.get("snv"):
        out = snv(out)
    if steps.get("msc"):
        out = msc(out, ref=msc_ref)
    if steps.get("deriv1"):
        out = derivative(out, 1)
    if steps.get("deriv2"):
        out = derivative(out, 2)
    return out
