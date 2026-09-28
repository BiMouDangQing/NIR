"""图形界面入口（秋月梨主题）。

用法：
    python -m QT.app
"""
import logging
import sys

from PySide6.QtCore import QLockFile, Qt
from PySide6.QtWidgets import QApplication, QMessageBox

from config.settings import CONFIG_FILE, LOG_DIR
from QT.main_window import MainWindow
from tools.logger import install_excepthook, setup_logging

logger = logging.getLogger("NIR.app")


def main() -> None:
    root_logger = setup_logging(LOG_DIR)
    install_excepthook(root_logger, LOG_DIR)

    # 高 DPI 适配：保留非整数缩放比例（如 125%/150%），避免 Windows 高分辨率屏
    # 幕下窗口最大化/全屏时控件与文字被舍入缩放而模糊失真。
    if hasattr(Qt, "HighDpiScaleFactorRoundingPolicy"):
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )

    app = QApplication(sys.argv)

    # 单实例：防止重复启动导致多个窗口
    lock = QLockFile(str(CONFIG_FILE.parent / "app.lock"))
    lock.setStaleLockTime(10000)
    if not lock.tryLock(0):
        QMessageBox.information(None, "提示", "程序已在运行中，请勿重复启动。")
        return

    logger.info("启动图形界面")
    window = MainWindow()
    window.show()
    exit_code = app.exec()
    logger.info(f"程序退出，退出码 {exit_code}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
