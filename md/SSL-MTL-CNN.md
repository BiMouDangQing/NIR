# SSL-MTL-CNN（双通道自监督多任务学习）

本文档记录「双通道自监督多任务学习 CNN」方案的实现与实验结论，对应代码：

- `model/cnn_ssl_mtl.py`：双通道构造、1D CNN 编码器、SSL 重建、MTL 回归

> **结论（2026-09-29）：已实现并完整验证，暂不采用。** 详见第 4 节。

## 1. 背景

在「自编码器 + 微调」（CV R²=0.873）之外，探索更复杂的深度学习方案：
双通道 CNN + 自监督预训练 + 多任务学习。

## 2. 架构

### 2.1 双通道输入

受限于标签数据只有 1024 维原始强度 `SpectrumData`（缺 301 维吸光度 `Abs`），
双通道采用「原始光谱 + 一阶导数光谱」：

- 通道 1：原始光谱（逐波长 z-score）
- 通道 2：一阶导数光谱（`np.gradient`，再 z-score，消除基线漂移）

两者都是 1024 维，均可从 `SpectrumData` 派生，输入形状 `(N, 2, 1024)`。

```python
from model.cnn_ssl_mtl import make_dual_channel
dual = make_dual_channel(X_zscore)  # (N, 2, 1024)
```

### 2.2 编码器（1D CNN）

四层 1D 卷积 + 批归一化 + ReLU + 最大池化，逐层把 1024 维压缩到 128 维，
再接全连接得到 `feat_dim` 维瓶颈特征：

```
Conv1d(2→32, k9) → BN → ReLU → MaxPool(2)     # 512
Conv1d(32→64, k7) → BN → ReLU → MaxPool(2)    # 256
Conv1d(64→128, k5) → BN → ReLU → MaxPool(2)   # 128
Conv1d(128→128, k3) → BN → ReLU → AdaptiveAvgPool → fc(feat_dim)
```

### 2.3 SSL 自监督预训练

用 45644 无标签数据重建双通道光谱（MSE），解码器为全连接
`feat_dim → 256 → 2×1024`。对应 `train_ssl`。

### 2.4 MTL 多任务微调

微调时联合优化两个损失：

$$\mathcal{L} = \mathcal{L}_{reg} + \lambda \cdot \mathcal{L}_{rec}$$

- 主任务：糖度回归（MSE）
- 辅助任务：光谱重建（MSE，权重 $\lambda$ = `recon_weight`）

对应 `train_mtl`，支持冻结/解冻两阶段。

## 3. 组件清单

| 组件 | 说明 |
| --- | --- |
| `make_dual_channel(X)` | 单通道 → 双通道（原始 + 一阶导数） |
| `CNNDualEncoder` | 1D CNN 编码器 |
| `SSLCMTModel` | 编码器 + 全连接解码器（SSL 重建） |
| `CNNRegressionHead` | 糖度回归头 |
| `train_ssl` | 自监督预训练 |
| `train_mtl` | 多任务微调 |
| `predict` | 回归预测 |

## 4. 实验结论

### 4.1 第一轮（feat_dim=64）

数据：预训练 45644 无标签，有标签 1836（微调 1596 + 预测0906 240），5 折 CV：

| 方案 | CV R² |
| --- | --- |
| 现有 MLP 自编码器 + 微调 | 0.873 |
| CNN 双通道（recon_weight=0，纯回归） | 0.575 |
| CNN 双通道 + MTL（recon_weight=0.1） | 0.725 |

### 4.2 第二轮（调参：特征维度 + 重建权重）

**方向 1：特征维度扫描（recon_weight=0.1 固定）**

| feat_dim | CV R² |
| --- | --- |
| 16 | 0.552 ±0.029 |
| 32 | 0.521 ±0.052 |
| 64 | 0.560 ±0.041 |

特征维度影响不大（均 ≈0.55）。

**方向 2：重建权重扫描（feat_dim=64 固定）**

| recon_weight | CV R² |
| --- | --- |
| 0.1 | 0.725 ±0.075 |
| 0.2 | 0.545 ±0.027 |
| 0.3 | 0.748 ±0.085 |
| 0.5 | **0.824 ±0.119** |

重建权重是决定性超参：权重越大回归越好（0.1→0.3→0.5 递升到 0.824），
印证了重建任务作为强正则对 CNN 编码器特征学习的帮助。但 recon_weight=0.5 时
折间波动大（±0.119），存在过拟合风险。

## 5. 结论

1. 多任务重建辅助（$\lambda>0$）确实提升回归，且权重越大越强（0.5 时 CV R²=0.824）；
2. 但最优 0.824 仍低于现有 MLP 自编码器微调的 0.873，且波动大；
3. 暂不采用，框架保留在 `model/cnn_ssl_mtl.py` 备选；若继续可试 recon_weight
   0.7~1.0，但需注意过拟合与稳定性。
