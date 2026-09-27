"""parquet 文件识别与解析模块。

提供 ``.parquet`` 格式文件的识别、读取、元信息查询与光谱数据加载能力。

依赖：``pyarrow``、``pandas``、``numpy``。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# parquet 文件头魔数（前 4 字节固定为 "PAR1"）
_MAGIC = b"PAR1"


def is_parquet(path: str | os.PathLike[str]) -> bool:
    """判断文件是否为 parquet 格式。

    先按扩展名快速判断，再读取文件头魔数做最终确认。

    参数
    ----
    path : str
        文件路径。

    返回
    ----
    bool
        是 parquet 文件返回 True，否则返回 False。
    """
    p = Path(path)
    if not p.is_file():
        return False
    if p.suffix.lower() != ".parquet":
        return False
    with open(p, "rb") as f:
        return f.read(4) == _MAGIC


def read_parquet(path: str | os.PathLike[str]) -> pd.DataFrame:
    """将 parquet 文件读取为 pandas.DataFrame。

    参数
    ----
    path : str
        parquet 文件路径。

    返回
    ----
    pandas.DataFrame
        文件中的表格数据。
    """
    return pd.read_parquet(path, engine="pyarrow")


def read_parquet_schema(path: str | os.PathLike[str]) -> pa.Schema:
    """读取 parquet 文件的表结构。

    参数
    ----
    path : str
        parquet 文件路径。

    返回
    ----
    pyarrow.Schema
        表结构（字段名与类型）。
    """
    return pq.read_schema(path)


def read_parquet_metadata(path: str | os.PathLike[str]) -> dict[str, Any]:
    """读取 parquet 文件的关键元信息。

    参数
    ----
    path : str
        parquet 文件路径。

    返回
    ----
    dict
        包含行数、列数、文件大小、列名等元信息。
    """
    f = pq.ParquetFile(path)
    return {
        "path": str(path),
        "rows": f.metadata.num_rows,
        "columns": f.metadata.num_columns,
        "size_bytes": os.path.getsize(path),
        "column_names": f.schema.names,
        "format_version": f.metadata.format_version,
    }


def load_spectra(path: str | os.PathLike[str]) -> tuple[pd.DataFrame, list[str]]:
    """读取光谱数据，返回 DataFrame 与列名列表。

    参数
    ----
    path : str
        parquet 文件路径。

    返回
    ----
    tuple[pandas.DataFrame, list[str]]
        (数据表, 列名列表)。
    """
    df = read_parquet(path)
    return df, list(df.columns)


def to_numpy(df: pd.DataFrame) -> np.ndarray:
    """将 DataFrame 转换为 numpy 数组。

    参数
    ----
    df : pandas.DataFrame
        数据表。

    返回
    ----
    numpy.ndarray
        数值矩阵。
    """
    return df.to_numpy()
