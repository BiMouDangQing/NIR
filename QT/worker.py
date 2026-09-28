"""后台任务线程：在子线程中运行训练/预测，通过信号更新界面。"""
from __future__ import annotations

import inspect
import logging
import threading

from PySide6.QtCore import QThread, Signal


class ModelWorker(QThread):
    """在后台线程执行一个函数，并把日志/结果/错误通过信号送回主线程。

    提供 ``pause_event``（threading.Event），支持训练暂停/继续。
    """

    log_signal = Signal(str)
    done_signal = Signal(object)
    error_signal = Signal(str)
    epoch_signal = Signal(int, int, float, object, object)  # epoch, epochs, train_loss, val_loss, val_metrics

    def __init__(self, fn, **kwargs) -> None:
        super().__init__()
        self._fn = fn
        self._kwargs = kwargs
        self.pause_event = threading.Event()

    def run(self) -> None:  # noqa: D102 - Qt 线程入口
        try:
            kwargs = dict(self._kwargs)
            # 若目标函数声明了 pause_event 参数，则注入本线程的暂停事件
            if "pause_event" in inspect.signature(self._fn).parameters:
                kwargs["pause_event"] = self.pause_event
            # 若声明了 on_epoch，注入逐 epoch 进度回调
            if "on_epoch" in inspect.signature(self._fn).parameters:
                kwargs["on_epoch"] = self.epoch_signal.emit
            result = self._fn(log=self.log_signal.emit, **kwargs)
            self.done_signal.emit(result)
        except Exception as exc:  # noqa: BLE001 - 回传界面展示
            logging.getLogger("NIR").exception("后台任务失败")
            self.error_signal.emit(str(exc))
