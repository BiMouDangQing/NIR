# NIR-Pyruspyrifolia 项目设计文档

项目定位：梨（*Pyrus pyrifolia*）近红外（NIR）光谱数据处理项目。核心需求：识别并解析 `.parquet` 格式的光谱数据文件。

## 1. 目录结构与职责划分

| 目录 | 职责 | 对应代码 |
| --- | --- | --- |
| `config/` | 全局配置（路径、波段、模型参数） | `config/settings.py` |
| `tools/` | 核心工具库（parquet 解析、光谱处理） | `tools/data_loader.py` 等 |
| `QT/` | 图形界面（调用 tools 的功能） | `QT/main_window.py` 等 |
| `md/` | 项目文档 | 本目录 |

目录树：

```
NIR-Pyruspyrifolia/
├── config/
│   ├── __init__.py
│   ├── settings.py          # 数据目录、默认参数、日志/配置路径
│   └── memory.py            # 配置记忆（JSON 持久化）
├── model/
│   ├── __init__.py
│   ├── autoencoder.py       # 自编码器（自监督预训练/特征学习）
│   ├── preprocess.py        # 光谱预处理（SNV / MSC / 平滑 / 导数）
│   ├── outlier.py           # 离群剔除（训练前聚类）
│   ├── reduce.py            # 降维（PCA）
│   ├── anomaly.py           # 异常检测
│   └── service.py           # 高层编排（训练/预测/可视化）
├── tools/
│   ├── __init__.py
│   ├── data_loader.py       # parquet 识别与解析
│   ├── logger.py            # 运行日志 + 崩溃日志
│   ├── cli.py               # 命令行入口
│   ├── cluster_baseline.py  # 无监督聚类基线
│   └── autoencoder.py       # 训练 CLI（调用 model.service）
├── QT/
│   ├── __init__.py
│   ├── theme.py             # 秋月梨主题 QSS
│   ├── main_window.py       # 主窗口（5 页签）
│   ├── worker.py            # 后台任务线程
│   └── app.py               # 图形界面入口
├── data/                    # 数据目录
├── log/                     # 运行日志、崩溃日志（运行时生成）
├── md/
│   ├── 环境配置手册.md
│   ├── NIR-Pyruspyrifolia.md
│   └── CHANGELOG.md
├── NIR-Pyruspyrifolia.bat   # 一键启动脚本（双击即可）
├── requirements.txt
└── runtime/                 # 便携 Python 运行时（含全部依赖）
```

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
| 主窗口 | `QT/main_window.py` | 数据预览 / 自监督预训练 / 预测 / 可视化 / 帮助（页签） |
| 日志 | `tools/logger.py` | 运行日志 `log/app_*.log` + 崩溃日志 `log/crash_*.log` |
| 无监督聚类 | `tools/cluster_baseline.py` | 解析光谱→SNV→PCA→KMeans 基线 |
| 自编码器 | `model/autoencoder.py` | 自监督预训练、特征学习、设备选择（CPU/GPU） |
| 预处理 | `model/preprocess.py` | SNV / MSC / S-G 平滑 / 一二阶导数（可选启用） |
| 离群剔除 | `model/outlier.py` | 训练前聚类（SNV→PCA→KMeans）按距离分位剔除 |
| 降维 | `model/reduce.py` | PCA 降维（可视化） |
| 异常检测 | `model/anomaly.py` | 重构误差阈值与 TopN |
| 建模服务 | `model/service.py` | 训练/预测/可视化编排，训练返回损失曲线 |
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
