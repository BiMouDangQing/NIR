# NIR-Pyruspyrifolia 项目设计文档

项目定位：梨（*Pyrus pyrifolia*）近红外（NIR）光谱数据处理项目。核心能力：解析
`.parquet` 光谱数据，进行**无监督自编码器建模**（特征学习、异常检测、降维可视化）
与**有监督微调**（用真实标签预测糖度等指标）。

## 📚 文档导航

| 文档 | 内容 |
| --- | --- |
| [项目结构](项目结构.md) | 目录结构、模块职责、调用关系、八个页签 |
| [无监督学习](无监督学习.md) | 自编码器原理、公式、异常检测、PCA、参数含义 |
| [数据预处理](数据预处理.md) | SNV/MSC/S-G/导数与离群剔除的公式与参数 |
| [模型微调](模型微调.md) | 有监督微调原理、公式、评估指标、预测评估 |
| [数据分布核查](数据分布核查.md) | 预训练与标签样本的域偏移检测方法与结果 |
| [模型训练日志](模型训练日志.md) | 每次训练/微调的参数、结果、分析与建议 |
| [算法与数据流](算法与数据流.md) | 字段含义、端到端数据流全景 |
| [环境配置手册](环境配置手册.md) | 运行时、依赖、启动、GPU 配置 |
| [CHANGELOG](CHANGELOG.md) | 改动日志 |

## 1. 目录结构与职责划分

| 目录 | 职责 | 对应代码 |
| --- | --- | --- |
| `config/` | 全局配置（路径、配置记忆） | `config/settings.py`、`config/memory.py` |
| `model/` | 算法核心（自编码器/预处理/离群/降维/异常/微调/分布核查/数据分析/编排） | `model/*.py` |
| `QT/` | 图形界面（8 页签） | `QT/main_window.py` 等 |
| `tools/` | 工具库（parquet 解析、xlsx 转换、日志、CLI） | `tools/*.py` |
| `md/` | 项目文档 | 本目录 |

完整目录树与模块职责见 [项目结构](项目结构.md)。

## 2. 功能模块：parquet 解析（tools/data_loader.py）

职责：识别 `.parquet` 文件并解析为表格数据，供 `QT/` 界面与后续光谱处理调用。

### 2.1 设计函数

| 函数 | 参数 | 返回 | 说明 |
| --- | --- | --- | --- |
| `is_parquet(path)` | `path: str` | `bool` | 按扩展名 + 文件头魔数判断是否为 parquet |
| `read_parquet(path)` | `path: str` | `pandas.DataFrame` | 读取 parquet 为 DataFrame |
| `read_parquet_schema(path)` | `path: str` | `pyarrow.Schema` | 读取表结构 |
| `read_parquet_metadata(path)` | `path: str` | `dict` | 行数、列数、文件大小等元信息 |
| `load_spectra(path)` | `path: str` | `tuple[DataFrame, list[str]]` | 读取光谱数据并返回数据与列名 |
| `to_numpy(df)` | `df: DataFrame` | `numpy.ndarray` | DataFrame 转 numpy 数组 |

依赖：`pyarrow`、`pandas`、`numpy`。

### 2.2 已实现代码（tools/data_loader.py）

> 实际文件包含完整 docstring 与类型注解，以下为功能等价的精简版：

```python
import os
import pandas as pd
import pyarrow.parquet as pq

MAGIC = b"PAR1"  # parquet 文件头魔数

def is_parquet(path: str) -> bool:
    if not path.lower().endswith(".parquet"):
        return False
    with open(path, "rb") as f:
        return f.read(4) == MAGIC

def read_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path, engine="pyarrow")

def read_parquet_metadata(path: str) -> dict:
    f = pq.ParquetFile(path)
    return {
        "rows": f.metadata.num_rows,
        "columns": f.metadata.num_columns,
        "size_bytes": os.path.getsize(path),
        "column_names": f.schema.names,
    }
```

### 2.3 配置文件（config/settings.py，已生成）

```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PARQUET_SUFFIX = ".parquet"
PARQUET_ENGINE = "pyarrow"
```

## 3. 功能与代码归属对照

| 功能 | 归属文件 | 说明 |
| --- | --- | --- |
| parquet 识别 | `tools/data_loader.py` | `is_parquet` |
| parquet 解析 | `tools/data_loader.py` | `read_parquet` 等 |
| 配置 | `config/settings.py` | 路径、默认参数 |
| 命令行入口 | `tools/cli.py` | 命令行解析 parquet（`python -m tools.cli`） |
| 图形界面入口 | `QT/app.py` | 启动 Qt 界面（`python -m QT.app`） |
| 主题样式 | `QT/theme.py` | 「秋月梨」QSS 配色 |
| 主窗口 | `QT/main_window.py` | 数据预览 / 自监督预训练 / 预测 / 可视化 / 分布核查 / 数据分析 / 帮助（页签） |
| 日志 | `tools/logger.py` | 运行日志 `log/app_*.log` + 崩溃日志 `log/crash_*.log` |
| 无监督聚类 | `tools/cluster_baseline.py` | 解析光谱→SNV→PCA→KMeans 基线 |
| 自编码器 | `model/autoencoder.py` | 自监督预训练、特征学习、设备选择（CPU/GPU） |
| 预处理 | `model/preprocess.py` | SNV / MSC / S-G 平滑 / 一二阶导数（可选启用） |
| 离群剔除 | `model/outlier.py` | 训练前聚类（SNV→PCA→KMeans）按距离分位剔除 |
| 降维 | `model/reduce.py` | PCA 降维（可视化） |
| 异常检测 | `model/anomaly.py` | 重构误差阈值与 TopN |
| 建模服务 | `model/service.py` | 训练/预测/可视化编排，训练返回损失曲线 |
| 分布核查 | `model/domain_shift.py` | 预训练与标签样本的 6 项域偏移检测 |
| 数据分析 | `model/data_analysis.py` | 多批数据的分布统计/离群检测/两两相似性 |
| 配置记忆 | `config/memory.py` | JSON 持久化窗口位置与上次路径 |

## 4. 实施状态

- [x] 内置便携运行时 `runtime/`（Python 3.14.6 embeddable，含 `pyarrow / pandas / numpy / PySide6`）
- [x] `tools/__init__.py`、`tools/data_loader.py`（见 §2.2）
- [x] `config/__init__.py`、`config/settings.py`
- [x] `requirements.txt`、`tools/cli.py`
- [x] 生成 `data/sample.parquet` 并端到端验证解析通过
- [x] `QT/theme.py`、`QT/main_window.py`（「秋月梨」主题界面）与 `QT/app.py` 入口
- [x] `tools/logger.py`：运行日志与崩溃日志写入 `log/`
- [x] `config/memory.py`：配置记忆（窗口位置、上次文件路径）

## 5. 使用示例

一键启动（推荐）：双击 `NIR-Pyruspyrifolia.bat`（内置便携运行时，无需安装 Python、无需联网）。

命令行解析：

```powershell
cd d:\model\model\NIR-Pyruspyrifolia
.\runtime\python.exe -m tools.cli data\sample.parquet
```

启动图形界面：

```powershell
cd d:\model\model\NIR-Pyruspyrifolia
.\runtime\python.exe -m QT.app
```

## 6. 后续扩展

- 光谱预处理（平滑、基线校正）→ `tools/spectrum.py`
- 模型推理 → `tools/model.py`
- 界面图表与光谱曲线可视化 → `QT/`
