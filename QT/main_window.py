"""主窗口：秋月梨近红外光谱分析前端。

整合 ``tools`` 中的 parquet 解析能力，提供文件选择、元信息展示与数据预览。
"""
from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QByteArray, Qt, QTimer, Signal
from PySide6.QtGui import QCloseEvent, QColor, QPainter, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from PySide6.QtCharts import QChart, QChartView, QLineSeries, QScatterSeries

from config.memory import AppMemory
from config.settings import CONFIG_FILE
from model.service import run_prediction, run_training, run_visualize
from QT.theme import QIYUE_THEME
from QT.worker import ModelWorker
from tools import is_parquet, load_spectra, read_parquet_metadata

logger = logging.getLogger("NIR.ui")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGO_PATH = PROJECT_ROOT / "lg.png"

_BASE_FONT_PT = 9.0
_MIN_ZOOM = 0.5
_MAX_ZOOM = 3.0
_ZOOM_STEP = 1.2
_PREVIEW_MAX_ROWS = 1000

_HELP_TEXT = """\
# 秋月梨 NIR 近红外光谱分析系统 · 使用帮助

## 项目简介
本系统解析秋月梨近红外（NIR）光谱数据（parquet 格式），通过**自监督学习**对
1024 维原始光谱建模，提供特征学习、降维可视化与异常检测。

## 数据说明
- **SpectrumData（1024 维）**：原始光谱强度，是建模输入
- **Abs（301 维）**：由 NIR 计算的吸光度
- 每行 = 一个秋月梨样本（共 24462 条）

## 页签功能
| 页签 | 功能 |
| --- | --- |
| 数据预览 | 打开 parquet、表格预览、缩放、导出 CSV |
| 自监督预训练 | 训练 / 更新自编码器模型，绘制损失曲线，支持暂停 |
| 模型预测 | 用已训练模型做异常检测，输出重构误差 |
| 降维可视化 | 瓶颈特征 PCA 到 2D 散点图 |
| 帮助 | 本页面 |

## 训练参数说明
| 参数 | 含义 | 默认值 |
| --- | --- | --- |
| 训练比例 | 训练集占比（其余为验证集） | 80% |
| 训练轮数 | 遍历全部数据的次数，越大训练越久 | 30 |
| 批大小 | 每次更新用的样本数，越大越快但占内存 | 256 |
| 学习率 | 权重更新步长，太大不收敛、太小收敛慢 | 0.001 |
| 瓶颈维度 | 编码器压缩到的特征维度 | 16 |
| 训练设备 | 自动检测 / CPU / GPU（CUDA），默认自动 | 自动 |
| 数据预处理 | SNV / MSC / S-G 平滑 / 一二阶导数，勾选启用 | 默认关闭 |
| 离群剔除 | 训练前聚类（SNV→PCA→KMeans）按距离分位剔除，支持自动选 k | 默认关闭 |

## 算法模型
- **自编码器**：1024 → 256 → 64 → 16 → 64 → 256 → 1024
- **自监督预训练**：无标签，用「重建输入」训练，更新编码器与解码器权重
- **特征学习**：瓶颈层 16 维特征
- **异常检测**：重构误差越大越可能是异常样本

## 异常阈值
- 按重构误差的**百分位数**划分，默认 99 分位（误差最大的 1% 判为异常）
"""


class ZoomableTable(QTableWidget):
    """支持 Ctrl + 滚轮缩放的表格。"""

    zoom_wheel = Signal(int)

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.zoom_wheel.emit(event.angleDelta().y())
            event.accept()
        else:
            super().wheelEvent(event)


class MainWindow(QMainWindow):
    """秋月梨主题主窗口。"""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("秋月梨 · 近红外光谱分析系统")
        self.resize(960, 640)
        self.setStyleSheet(QIYUE_THEME)

        self._memory = AppMemory(CONFIG_FILE)
        self._memory.load()

        self._df = None
        self._columns: list[str] = []
        self._zoom = 1.0

        self._build_ui()
        self._restore_state()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # 顶部标题栏（含 logo）
        header = QWidget()
        header.setObjectName("header")
        header.setFixedHeight(84)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(20, 12, 20, 12)

        logo = QLabel()
        if LOGO_PATH.exists():
            logo.setPixmap(QPixmap(str(LOGO_PATH)))
        else:
            logo.setText("🍐")
        logo.setFixedSize(183, 51)
        logo.setScaledContents(True)
        header_layout.addWidget(logo, 0, Qt.AlignLeft)

        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        title = QLabel("秋月梨")
        title.setObjectName("appTitle")
        subtitle = QLabel("NIR 近红外光谱分析系统 · Parquet 数据解析")
        subtitle.setObjectName("appSubtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header_layout.addLayout(title_box)
        header_layout.addStretch()
        root.addWidget(header)

        # 页签：数据预览 / 模型训练 / 模型预测
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_preview_tab(), "数据预览")
        self.tabs.addTab(self._build_train_tab(), "自监督预训练")
        self.tabs.addTab(self._build_predict_tab(), "模型预测")
        self.tabs.addTab(self._build_visualize_tab(), "降维可视化")
        self.tabs.addTab(self._build_help_tab(), "帮助")
        root.addWidget(self.tabs, 1)

        self.statusBar().showMessage("就绪")

    # ---------- 页签构建 ----------
    def _build_preview_tab(self) -> QWidget:
        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(20, 16, 20, 16)
        body_layout.setSpacing(12)

        file_box = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("请选择 .parquet 文件")
        self.path_edit.setReadOnly(True)
        open_btn = QPushButton("打开 parquet 文件")
        open_btn.clicked.connect(self._open_file)
        file_box.addWidget(self.path_edit, 1)
        file_box.addWidget(open_btn)
        body_layout.addLayout(file_box)

        ctrl_box = QHBoxLayout()
        zoom_in_btn = QPushButton("放大 +")
        zoom_in_btn.clicked.connect(self._zoom_in)
        zoom_out_btn = QPushButton("缩小 -")
        zoom_out_btn.clicked.connect(self._zoom_out)
        zoom_reset_btn = QPushButton("重置 100%")
        zoom_reset_btn.clicked.connect(self._zoom_reset)
        export_btn = QPushButton("导出 CSV")
        export_btn.clicked.connect(self._export_csv)
        ctrl_box.addWidget(zoom_in_btn)
        ctrl_box.addWidget(zoom_out_btn)
        ctrl_box.addWidget(zoom_reset_btn)
        ctrl_box.addStretch()
        ctrl_box.addWidget(export_btn)
        body_layout.addLayout(ctrl_box)

        info_box = QGroupBox("文件信息")
        info_layout = QVBoxLayout(info_box)
        self.info_label = QLabel("尚未加载文件")
        self.info_label.setWordWrap(True)
        info_layout.addWidget(self.info_label)
        body_layout.addWidget(info_box)

        table_box = QGroupBox("数据预览")
        table_layout = QVBoxLayout(table_box)
        self.table = ZoomableTable()
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.zoom_wheel.connect(self._on_zoom_wheel)
        table_layout.addWidget(self.table)
        body_layout.addWidget(table_box, 1)
        return body

    def _path_row(self, key: str, mode: str = "file") -> tuple[QComboBox, QWidget]:
        """构造「下拉框 + 浏览」的一行路径控件，并加载最近 5 次历史。"""
        combo = QComboBox()
        combo.setEditable(True)
        combo.addItems(self._memory.get_recent(key))
        btn = QPushButton("浏览...")
        if mode == "dir":
            btn.clicked.connect(lambda: self._browse_dir(combo, key))
        elif mode == "save":
            btn.clicked.connect(lambda: self._browse_save(combo, key))
        else:
            btn.clicked.connect(lambda: self._browse_file(combo, key))
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        h.addWidget(combo, 1)
        h.addWidget(btn)
        return combo, row

    def _browse_file(self, combo: QComboBox, key: str) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 parquet 文件", "", "Parquet 文件 (*.parquet);;所有文件 (*)"
        )
        if path:
            self._remember_path(combo, key, path)

    def _browse_dir(self, combo: QComboBox, key: str) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择目录")
        if path:
            self._remember_path(combo, key, path)

    def _browse_save(self, combo: QComboBox, key: str) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "保存结果", "reconstruction_error.csv", "CSV 文件 (*.csv)"
        )
        if path:
            self._remember_path(combo, key, path)

    def _remember_path(self, combo: QComboBox, key: str, path: str) -> None:
        """把路径写入下拉框列表（本会话内可下拉选择）并持久化到配置记忆。"""
        items = [combo.itemText(i) for i in range(combo.count())]
        if path not in items:
            combo.insertItem(0, path)
        combo.setCurrentText(path)
        self._memory.add_recent(key, path)
        self._memory.save()

    def _build_train_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        form = QFormLayout()
        form.setSpacing(10)
        self.train_data_combo, data_row = self._path_row("recent_train_data", "file")
        form.addRow("数据文件:", data_row)
        self.train_out_combo, out_row = self._path_row("recent_model_dir", "dir")
        form.addRow("模型输出目录:", out_row)

        self.ratio_spin = QSpinBox()
        self.ratio_spin.setRange(10, 90)
        self.ratio_spin.setValue(80)
        self.ratio_spin.setSuffix(" %")
        form.addRow("训练比例:", self.ratio_spin)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(30)
        form.addRow("训练轮数:", self.epochs_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(16, 4096)
        self.batch_spin.setValue(256)
        form.addRow("批大小:", self.batch_spin)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 1.0)
        self.lr_spin.setDecimals(5)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setValue(0.001)
        form.addRow("学习率:", self.lr_spin)

        self.bottleneck_spin = QSpinBox()
        self.bottleneck_spin.setRange(2, 64)
        self.bottleneck_spin.setValue(16)
        form.addRow("瓶颈维度:", self.bottleneck_spin)

        self.device_combo = QComboBox()
        self.device_combo.addItem("自动检测", None)
        self.device_combo.addItem("CPU", "cpu")
        self.device_combo.addItem("GPU (CUDA)", "cuda")
        form.addRow("训练设备:", self.device_combo)
        layout.addLayout(form)

        pre_box = QGroupBox("数据预处理（可选，勾选启用）")
        pre_layout = QHBoxLayout(pre_box)
        self.pre_smooth_cb = QCheckBox("S-G 平滑")
        self.pre_snv_cb = QCheckBox("SNV")
        self.pre_msc_cb = QCheckBox("MSC")
        self.pre_d1_cb = QCheckBox("一阶导数")
        self.pre_d2_cb = QCheckBox("二阶导数")
        for cb in (
            self.pre_smooth_cb,
            self.pre_snv_cb,
            self.pre_msc_cb,
            self.pre_d1_cb,
            self.pre_d2_cb,
        ):
            pre_layout.addWidget(cb)
        pre_layout.addStretch()
        layout.addWidget(pre_box)

        outlier_box = QGroupBox("离群点剔除（可选，训练前聚类剔除）")
        outlier_layout = QHBoxLayout(outlier_box)
        self.outlier_cb = QCheckBox("启用")
        self.outlier_k_spin = QSpinBox()
        self.outlier_k_spin.setRange(2, 20)
        self.outlier_k_spin.setValue(5)
        self.outlier_pct_spin = QSpinBox()
        self.outlier_pct_spin.setRange(90, 100)
        self.outlier_pct_spin.setValue(99)
        self.outlier_pct_spin.setSuffix(" %")
        self.outlier_auto_cb = QCheckBox("自动选 k")
        self.outlier_auto_cb.toggled.connect(self.outlier_k_spin.setDisabled)
        outlier_layout.addWidget(self.outlier_cb)
        outlier_layout.addWidget(QLabel("聚类数:"))
        outlier_layout.addWidget(self.outlier_k_spin)
        outlier_layout.addWidget(self.outlier_auto_cb)
        outlier_layout.addWidget(QLabel("剔除分位:"))
        outlier_layout.addWidget(self.outlier_pct_spin)
        outlier_layout.addStretch()
        layout.addWidget(outlier_box)

        btn_row = QHBoxLayout()
        train_btn = QPushButton("开始训练")
        train_btn.clicked.connect(self._start_train)
        self.pause_btn = QPushButton("暂停训练")
        self.pause_btn.setCheckable(True)
        self.pause_btn.setEnabled(False)
        self.pause_btn.toggled.connect(self._toggle_pause)
        btn_row.addWidget(train_btn)
        btn_row.addWidget(self.pause_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.train_chart = QChart()
        self.train_chart.setTitle("训练损失曲线")
        self.train_chart_view = QChartView(self.train_chart)
        self.train_chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        layout.addWidget(self.train_chart_view, 1)

        self.train_log = QPlainTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setPlaceholderText("训练日志将显示在这里...")
        layout.addWidget(self.train_log, 1)
        return tab

    def _build_predict_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        form = QFormLayout()
        form.setSpacing(10)
        self.pred_model_combo, model_row = self._path_row("recent_model_dir", "dir")
        form.addRow("模型目录:", model_row)
        self.pred_data_combo, data_row = self._path_row("recent_predict_data", "file")
        form.addRow("数据文件:", data_row)
        self.pred_out_combo, out_row = self._path_row("recent_predict_out", "save")
        form.addRow("结果 CSV:", out_row)
        layout.addLayout(form)

        pred_btn = QPushButton("开始预测")
        pred_btn.clicked.connect(self._start_predict)
        layout.addWidget(pred_btn)

        self.pred_log = QPlainTextEdit()
        self.pred_log.setReadOnly(True)
        self.pred_log.setPlaceholderText("预测日志将显示在这里...")
        layout.addWidget(self.pred_log, 1)
        return tab

    # ---------- 训练 / 预测 ----------
    def _start_train(self) -> None:
        data = self.train_data_combo.currentText().strip()
        out = self.train_out_combo.currentText().strip()
        if not data:
            QMessageBox.warning(self, "提示", "请选择数据文件")
            return
        if not out:
            QMessageBox.warning(self, "提示", "请选择模型输出目录")
            return
        self._remember_path(self.train_data_combo, "recent_train_data", data)
        self._remember_path(self.train_out_combo, "recent_model_dir", out)

        self.train_log.clear()
        self.train_log.appendPlainText(f"开始训练：{data}\n")

        preprocess = {
            "smooth": self.pre_smooth_cb.isChecked(),
            "snv": self.pre_snv_cb.isChecked(),
            "msc": self.pre_msc_cb.isChecked(),
            "deriv1": self.pre_d1_cb.isChecked(),
            "deriv2": self.pre_d2_cb.isChecked(),
        }
        remove_outliers = None
        if self.outlier_cb.isChecked():
            remove_outliers = {
                "k": self.outlier_k_spin.value(),
                "percentile": self.outlier_pct_spin.value(),
                "auto": self.outlier_auto_cb.isChecked(),
            }
        self.pause_btn.setChecked(False)
        self.pause_btn.setEnabled(True)
        self.pause_btn.setText("暂停训练")

        self._train_worker = ModelWorker(
            run_training,
            data_path=data,
            out_dir=out,
            epochs=self.epochs_spin.value(),
            batch_size=self.batch_spin.value(),
            lr=self.lr_spin.value(),
            bottleneck=self.bottleneck_spin.value(),
            train_ratio=self.ratio_spin.value() / 100.0,
            device=self.device_combo.currentData(),
            preprocess=preprocess,
            remove_outliers=remove_outliers,
        )
        self._train_worker.log_signal.connect(self.train_log.appendPlainText)
        self._train_worker.done_signal.connect(self._on_train_done)
        self._train_worker.error_signal.connect(self._on_task_error)
        self._train_worker.start()

    def _on_train_done(self, summary: dict) -> None:
        self.pause_btn.setChecked(False)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("暂停训练")
        brief = {k: v for k, v in summary.items() if k != "history"}
        self.train_log.appendPlainText(f"\n训练完成：{brief}\n")
        self._plot_train_history(summary.get("history"))
        QMessageBox.information(
            self, "完成", f"训练完成，模型已保存到：\n{summary.get('out_dir')}"
        )

    def _toggle_pause(self, checked: bool) -> None:
        """暂停/继续训练：设置或清除后台线程的暂停事件。"""
        worker = getattr(self, "_train_worker", None)
        if worker is None:
            return
        if checked:
            worker.pause_event.set()
            self.pause_btn.setText("继续训练")
        else:
            worker.pause_event.clear()
            self.pause_btn.setText("暂停训练")

    def _plot_train_history(self, history: dict | None) -> None:
        """把训练/验证损失画成曲线（训练可视化）。"""
        if not history:
            return
        train_vals = history.get("train") or []
        val_vals = history.get("val") or []

        chart = QChart()
        chart.setTitle("训练损失曲线")

        train_series = QLineSeries()
        train_series.setName("训练损失")
        train_series.setColor(QColor("#D8A24A"))
        for i, v in enumerate(train_vals, 1):
            train_series.append(float(i), float(v))
        chart.addSeries(train_series)

        if val_vals:
            val_series = QLineSeries()
            val_series.setName("验证损失")
            val_series.setColor(QColor("#4A90D8"))
            for i, v in enumerate(val_vals, 1):
                val_series.append(float(i), float(v))
            chart.addSeries(val_series)

        chart.createDefaultAxes()
        x_axes = chart.axes(Qt.Orientation.Horizontal)
        y_axes = chart.axes(Qt.Orientation.Vertical)
        if x_axes:
            x_axes[0].setTitleText("训练轮数")
        if y_axes:
            y_axes[0].setTitleText("MSE 损失")
        self.train_chart_view.setChart(chart)

    def _start_predict(self) -> None:
        model_dir = self.pred_model_combo.currentText().strip()
        data = self.pred_data_combo.currentText().strip()
        out = self.pred_out_combo.currentText().strip()
        if not model_dir:
            QMessageBox.warning(self, "提示", "请选择模型目录")
            return
        if not data:
            QMessageBox.warning(self, "提示", "请选择数据文件")
            return
        if not out:
            QMessageBox.warning(self, "提示", "请选择结果 CSV 路径")
            return
        self._remember_path(self.pred_model_combo, "recent_model_dir", model_dir)
        self._remember_path(self.pred_data_combo, "recent_predict_data", data)
        self._remember_path(self.pred_out_combo, "recent_predict_out", out)

        self.pred_log.clear()
        self.pred_log.appendPlainText(f"开始预测：{data}\n")
        self._predict_worker = ModelWorker(
            run_prediction,
            data_path=data,
            model_dir=model_dir,
            out_path=out,
        )
        self._predict_worker.log_signal.connect(self.pred_log.appendPlainText)
        self._predict_worker.done_signal.connect(self._on_predict_done)
        self._predict_worker.error_signal.connect(self._on_task_error)
        self._predict_worker.start()

    def _on_predict_done(self, summary: dict) -> None:
        self.pred_log.appendPlainText(f"\n预测完成：{summary}\n")
        QMessageBox.information(
            self, "完成", f"预测完成，结果已保存到：\n{summary.get('out_path')}"
        )

    def _on_task_error(self, msg: str) -> None:
        self.pause_btn.setChecked(False)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("暂停训练")
        QMessageBox.critical(self, "错误", msg)

    def _build_visualize_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        form = QFormLayout()
        form.setSpacing(10)
        self.vis_model_combo, model_row = self._path_row("recent_model_dir", "dir")
        form.addRow("模型目录:", model_row)
        self.vis_data_combo, data_row = self._path_row("recent_visualize_data", "file")
        form.addRow("数据文件:", data_row)

        self.vis_percentile_spin = QSpinBox()
        self.vis_percentile_spin.setRange(90, 100)
        self.vis_percentile_spin.setValue(99)
        self.vis_percentile_spin.setSuffix(" %")
        form.addRow("异常阈值:", self.vis_percentile_spin)
        layout.addLayout(form)

        run_btn = QPushButton("开始分析")
        run_btn.clicked.connect(self._start_visualize)
        layout.addWidget(run_btn)

        self.vis_chart = QChart()
        self.vis_chart.setTitle("NIR 瓶颈特征 2D 可视化（PCA）")
        self.vis_chart_view = QChartView(self.vis_chart)
        self.vis_chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        layout.addWidget(self.vis_chart_view, 1)

        self.vis_info = QLabel("尚未分析")
        self.vis_info.setWordWrap(True)
        layout.addWidget(self.vis_info)
        return tab

    def _start_visualize(self) -> None:
        model_dir = self.vis_model_combo.currentText().strip()
        data = self.vis_data_combo.currentText().strip()
        if not model_dir:
            QMessageBox.warning(self, "提示", "请选择模型目录")
            return
        if not data:
            QMessageBox.warning(self, "提示", "请选择数据文件")
            return
        self._remember_path(self.vis_model_combo, "recent_model_dir", model_dir)
        self._remember_path(self.vis_data_combo, "recent_visualize_data", data)

        self.vis_info.setText("正在分析...")
        self._vis_worker = ModelWorker(
            run_visualize,
            data_path=data,
            model_dir=model_dir,
            percentile=self.vis_percentile_spin.value(),
        )
        self._vis_worker.done_signal.connect(self._on_visualize_done)
        self._vis_worker.error_signal.connect(self._on_task_error)
        self._vis_worker.start()

    def _on_visualize_done(self, summary: dict) -> None:
        coords = summary["coords"]
        flags = summary["is_anomaly"]

        chart = QChart()
        chart.setTitle("NIR 瓶颈特征 2D 可视化（PCA）")
        normal = QScatterSeries()
        normal.setName("正常")
        normal.setMarkerSize(4)
        normal.setColor(QColor("#D8A24A"))
        anom = QScatterSeries()
        anom.setName("异常")
        anom.setMarkerSize(7)
        anom.setColor(QColor("#D64545"))
        for (x, y), f in zip(coords, flags):
            (anom if f else normal).append(float(x), float(y))
        chart.addSeries(normal)
        chart.addSeries(anom)
        chart.createDefaultAxes()

        self.vis_chart_view.setChart(chart)
        self.vis_info.setText(
            f"样本 {summary['samples']}，异常 {int(flags.sum())} 个"
            f"（阈值 {summary['threshold']:.4f}），"
            f"PCA 前 2 主成分解释方差 {summary['explained']:.1%}"
        )

    def _build_help_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 16, 20, 16)
        browser = QTextBrowser()
        browser.setMarkdown(_HELP_TEXT)
        browser.setOpenExternalLinks(True)
        layout.addWidget(browser)
        return tab

    def _restore_state(self) -> None:
        """从配置记忆恢复窗口位置与上次打开的文件。"""
        geometry_hex = self._memory.get("geometry")
        if geometry_hex:
            try:
                self.restoreGeometry(QByteArray(bytes.fromhex(geometry_hex)))
            except (ValueError, TypeError):
                logger.warning("窗口位置信息损坏，已忽略")

        last_file = self._memory.get("last_file")
        if last_file:
            self.path_edit.setText(str(last_file))
            if Path(last_file).exists():
                # 延迟到窗口显示后再加载，避免大文件阻塞导致窗口迟迟不出现
                QTimer.singleShot(0, lambda: self._load(str(last_file)))
            else:
                logger.info(f"上次文件已不存在，仅回填路径: {last_file}")

    def closeEvent(self, event: QCloseEvent) -> None:
        """关闭时保存窗口位置与当前文件路径。"""
        self._memory.set("geometry", bytes(self.saveGeometry()).hex())
        self._memory.set("last_file", self.path_edit.text() or "")
        self._memory.save()
        logger.info("关闭界面，已保存配置记忆")
        event.accept()

    def _open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 parquet 文件", "", "Parquet 文件 (*.parquet);;所有文件 (*)"
        )
        if not path:
            return
        if not is_parquet(path):
            logger.warning(f"选择的不是 parquet 文件: {path}")
            QMessageBox.warning(self, "提示", f"不是有效的 parquet 文件：\n{path}")
            return
        self._load(path)

    def _load(self, path: str) -> None:
        logger.info(f"加载文件: {path}")
        try:
            meta = read_parquet_metadata(path)
            df, columns = load_spectra(path)
        except Exception as exc:  # noqa: BLE001 - 展示给用户
            logger.exception(f"解析失败: {path}")
            QMessageBox.critical(self, "错误", f"文件解析失败：\n{exc}")
            return

        self._df = df
        self._columns = columns

        truncated = len(df) > _PREVIEW_MAX_ROWS
        preview_df = df.head(_PREVIEW_MAX_ROWS)

        self.path_edit.setText(path)
        note = (
            f"（仅预览前 {_PREVIEW_MAX_ROWS} 行，完整数据可导出 CSV）" if truncated else ""
        )
        self.info_label.setText(
            f"行数：{meta['rows']}    列数：{meta['columns']}    "
            f"大小：{meta['size_bytes']} 字节    格式版本：{meta['format_version']}\n"
            f"列名：{', '.join(meta['column_names'])}\n"
            f"预览：{len(preview_df)} 行{note}"
        )

        self.table.clear()
        self.table.setRowCount(len(preview_df))
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)
        for r in range(len(preview_df)):
            for c in range(len(columns)):
                value = preview_df.iat[r, c]
                item = QTableWidgetItem(str(value))
                self.table.setItem(r, c, item)

        self.statusBar().showMessage(
            f"已加载：{path}（{meta['rows']} 行 × {meta['columns']} 列）"
        )
        logger.info(f"加载完成: {path}（{meta['rows']} 行 × {meta['columns']} 列）")

    # ---------- 缩放 ----------
    def _apply_zoom(self) -> None:
        """按当前缩放系数调整表格字号与行高。"""
        font = self.table.font()
        font.setPointSizeF(_BASE_FONT_PT * self._zoom)
        self.table.setFont(font)
        self.table.verticalHeader().setDefaultSectionSize(
            max(20, int(24 * self._zoom))
        )
        self.statusBar().showMessage(f"缩放 {int(self._zoom * 100)}%")

    def _zoom_in(self) -> None:
        self._zoom = min(_MAX_ZOOM, self._zoom * _ZOOM_STEP)
        self._apply_zoom()

    def _zoom_out(self) -> None:
        self._zoom = max(_MIN_ZOOM, self._zoom / _ZOOM_STEP)
        self._apply_zoom()

    def _zoom_reset(self) -> None:
        self._zoom = 1.0
        self._apply_zoom()

    def _on_zoom_wheel(self, delta: int) -> None:
        """Ctrl + 滚轮缩放。"""
        if delta > 0:
            self._zoom_in()
        else:
            self._zoom_out()

    # ---------- CSV 导出 ----------
    def _export_csv(self) -> None:
        """将当前加载的数据导出为 CSV 文件。"""
        if self._df is None:
            QMessageBox.information(self, "提示", "请先加载 parquet 文件")
            return

        default_name = "export.csv"
        if self.path_edit.text():
            default_name = Path(self.path_edit.text()).stem + ".csv"

        path, _ = QFileDialog.getSaveFileName(
            self, "导出 CSV", default_name, "CSV 文件 (*.csv)"
        )
        if not path:
            return

        try:
            self._df.to_csv(path, index=False, encoding="utf-8-sig")
        except Exception as exc:  # noqa: BLE001 - 展示给用户
            logger.exception(f"导出 CSV 失败: {path}")
            QMessageBox.critical(self, "错误", f"导出失败：\n{exc}")
            return

        logger.info(f"已导出 CSV: {path}")
        self.statusBar().showMessage(f"已导出：{path}")
