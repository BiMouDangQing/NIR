"""命令行入口：识别并解析 .parquet 文件。

用法：
    python -m tools.cli <path/to/file.parquet>
"""
import logging
import sys

from config.settings import DATA_DIR, LOG_DIR
from tools.data_loader import (
    is_parquet,
    load_spectra,
    read_parquet_metadata,
)
from tools.logger import setup_logging

logger = logging.getLogger("NIR.cli")


def main() -> None:
    setup_logging(LOG_DIR)
    if len(sys.argv) < 2:
        logger.info("未提供文件参数，打印用法")
        print("用法: python -m tools.cli <path/to/file.parquet>")
        print(f"默认数据目录: {DATA_DIR}")
        return

    path = sys.argv[1]
    if not is_parquet(path):
        logger.error(f"不是有效的 parquet 文件: {path}")
        print(f"[错误] 不是有效的 parquet 文件: {path}")
        return

    logger.info(f"开始解析: {path}")
    meta = read_parquet_metadata(path)
    print("== 文件信息 ==")
    for k, v in meta.items():
        print(f"  {k}: {v}")

    df, columns = load_spectra(path)
    print("\n== 数据预览（前 5 行） ==")
    print(df.head())
    print(f"\n列名: {columns}")
    logger.info(f"解析完成: {path}（{meta['rows']} 行 × {meta['columns']} 列）")


if __name__ == "__main__":
    main()
