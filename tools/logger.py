"""日志模块：运行日志与崩溃日志，统一写入 log/ 目录。

- 运行日志：``log/app_YYYY-MM-DD.log``（按天生成，每天一个文件，保留最近 30 天）
- 崩溃日志：``log/crash_YYYYMMDD_HHMMSS.log``（未捕获异常时单独落盘）
"""
from __future__ import annotations

import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class DailyFileHandler(logging.Handler):
    """按天生成日志文件：``app_YYYY-MM-DD.log``，每天一个文件。"""

    def __init__(self, log_dir: Path, backup_count: int = 30) -> None:
        super().__init__()
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._backup_count = backup_count
        self._current_date: str | None = None
        self._stream = None

    def emit(self, record: logging.LogRecord) -> None:
        try:
            date_str = datetime.now().strftime("%Y-%m-%d")
            if date_str != self._current_date:
                if self._stream is not None:
                    self._stream.close()
                self._stream = open(
                    self._log_dir / f"app_{date_str}.log", "a", encoding="utf-8"
                )
                self._current_date = date_str
                self._cleanup()
            self._stream.write(self.format(record) + "\n")
            self._stream.flush()
        except Exception:  # noqa: BLE001 - 日志失败不阻断业务
            self.handleError(record)

    def _cleanup(self) -> None:
        """删除超过保留天数的旧日志。"""
        logs = sorted(self._log_dir.glob("app_*.log"))
        if len(logs) > self._backup_count:
            for old in logs[: len(logs) - self._backup_count]:
                try:
                    old.unlink()
                except OSError:
                    pass

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        super().close()


def setup_logging(log_dir: str | Path, name: str = "NIR") -> logging.Logger:
    """初始化日志系统并返回应用 logger。

    参数
    ----
    log_dir : str
        日志目录（不存在则自动创建）。
    name : str
        logger 名称。

    返回
    ----
    logging.Logger
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # 控制台编码统一为 UTF-8，避免路径含特殊字符（如不间断空格）时写日志报错
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    file_handler = DailyFileHandler(log_dir, backup_count=30)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if sys.stdout is not None:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def install_excepthook(logger: logging.Logger, log_dir: str | Path) -> None:
    """安装全局异常钩子，未捕获异常时写入运行日志与独立崩溃日志。

    参数
    ----
    logger : logging.Logger
        用于记录崩溃信息的 logger。
    log_dir : str
        崩溃日志存放目录。
    """
    log_dir = Path(log_dir)

    def _excepthook(exc_type, exc_value, exc_tb) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        logger.critical("未捕获异常（崩溃）", exc_info=(exc_type, exc_value, exc_tb))
        _write_crash_log(log_dir, exc_type, exc_value, exc_tb)

    sys.excepthook = _excepthook


def _write_crash_log(log_dir: Path, exc_type, exc_value, exc_tb) -> None:
    """将崩溃堆栈单独写入带时间戳的崩溃日志文件。"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    crash_file = log_dir / f"crash_{stamp}.log"
    content = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    try:
        crash_file.write_text(content, encoding="utf-8")
    except OSError:
        pass
