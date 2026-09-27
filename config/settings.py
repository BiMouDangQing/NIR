"""全局配置：数据目录、默认参数等。"""
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 数据目录（parquet 文件默认存放位置，可按需修改）
DATA_DIR = PROJECT_ROOT / "data"

# parquet 文件扩展名
PARQUET_SUFFIX = ".parquet"

# 默认读取引擎
PARQUET_ENGINE = "pyarrow"

# 日志目录（运行日志、崩溃日志）
LOG_DIR = PROJECT_ROOT / "log"

# 配置记忆文件（记住上次参数与路径）
CONFIG_FILE = PROJECT_ROOT / "config" / "app_config.json"
