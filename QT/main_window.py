"""主窗口：秋月梨近红外光谱分析前端。

整合 ``tools`` 中的 parquet 解析能力，提供文件选择、元信息展示与数据预览。
"""
from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QByteArray, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QCloseEvent, QColor, QFont, QImage, QPainter, QPen, QPixmap
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
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from PySide6.QtCharts import (
    QBarCategoryAxis,
    QBarSeries,
    QBarSet,
    QChart,
    QChartView,
    QLineSeries,
    QScatterSeries,
    QValueAxis,
)

from config.memory import AppMemory
from config.settings import CONFIG_FILE
from model.data_analysis import run_data_analysis
from model.domain_shift import run_domain_shift
from model.service import FINETUNE_DIR, run_finetune, run_finetune_predict, run_prediction, run_training, run_visualize
from QT.theme import QIYUE_THEME
from QT.pointcloud_3d import PointCloud3DWidget
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
| 自监督预训练 | 训练 / 更新自编码器模型，实时进度条 + 损失曲线，支持暂停 |
| 模型预测 | 用已训练模型做异常检测，输出重构误差 |
| 模型微调 | 用有标签数据微调回归头，预测指定指标（需提供真实标签） |
| 降维可视化 | 瓶颈特征 PCA 到 2D 散点图 |
| 分布核查 | 核查预训练样本与标签样本的分布一致性（域偏移检测） |
| 数据分析 | 对预训练/微调/预测多批数据做分布统计、离群检测与相似性分析 |
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
        # 所有路径下拉框 (combo, 记忆 key)，用于关闭时统一保存当前值
        self._path_combos: list[tuple[QComboBox, str]] = []

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
        self.tabs.addTab(self._build_finetune_tab(), "模型微调")
        self.tabs.addTab(self._build_visualize_tab(), "降维可视化")
        self.tabs.addTab(self._build_shift_tab(), "分布核查")
        self.tabs.addTab(self._build_analysis_tab(), "数据分析")
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
        self._path_combos.append((combo, key))
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
        self.train_out_combo, out_row = self._path_row("recent_train_out_dir", "dir")
        form.addRow("模型输出目录:", out_row)

        self.spectrum_combo = QComboBox()
        self.spectrum_combo.addItem("SpectrumData（1024 维原始光谱）", "SpectrumData")
        self.spectrum_combo.addItem("Abs（301 维吸光度）", "Abs")
        form.addRow("光谱字段:", self.spectrum_combo)

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

        self.recon_top_spin = QSpinBox()
        self.recon_top_spin.setRange(1, 10)
        self.recon_top_spin.setValue(3)
        self.recon_top_spin.setSuffix(" 个（误差最大）")
        form.addRow("重建对比样本数:", self.recon_top_spin)
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
        self.outlier_pct_spin = QDoubleSpinBox()
        self.outlier_pct_spin.setRange(90.0, 100.0)
        self.outlier_pct_spin.setDecimals(2)
        self.outlier_pct_spin.setSingleStep(0.05)
        self.outlier_pct_spin.setValue(99.0)
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
        self.train_btn = QPushButton("开始训练")
        self.train_btn.clicked.connect(self._start_train)
        self.pause_btn = QPushButton("暂停训练")
        self.pause_btn.setCheckable(True)
        self.pause_btn.setEnabled(False)
        self.pause_btn.toggled.connect(self._toggle_pause)
        btn_row.addWidget(self.train_btn)
        btn_row.addWidget(self.pause_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.train_progress = QProgressBar()
        self.train_progress.setRange(0, 1)
        self.train_progress.setValue(0)
        self.train_progress.setFormat("就绪")
        layout.addWidget(self.train_progress)

        self.train_chart = QChart()
        self.train_chart.setTitle("训练损失曲线")
        self.train_chart_view = QChartView(self.train_chart)
        self.train_chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        layout.addWidget(self.train_chart_view, 1)

        self.train_recon_chart = QChart()
        self.train_recon_chart.setTitle("重建光谱对比（训练完成后显示）")
        self.train_recon_view = QChartView(self.train_recon_chart)
        self.train_recon_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        layout.addWidget(self.train_recon_view, 1)

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
        self.pred_model_combo, model_row = self._path_row("recent_pred_model_dir", "dir")
        form.addRow("模型目录:", model_row)
        self.pred_data_combo, data_row = self._path_row("recent_predict_data", "file")
        form.addRow("数据文件:", data_row)
        layout.addLayout(form)

        pred_btn = QPushButton("开始预测")
        pred_btn.clicked.connect(self._start_predict)
        layout.addWidget(pred_btn)

        self.pred_metric = QLabel("尚未预测")
        self.pred_metric.setWordWrap(True)
        layout.addWidget(self.pred_metric)

        self.pred_chart = QChart()
        self.pred_chart.setTitle("重构误差分布")
        self.pred_chart_view = QChartView(self.pred_chart)
        self.pred_chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        layout.addWidget(self.pred_chart_view, 1)

        self.pred_log = QPlainTextEdit()
        self.pred_log.setReadOnly(True)
        self.pred_log.setPlaceholderText("预测日志将显示在这里...")
        layout.addWidget(self.pred_log, 1)
        return tab

    def _build_finetune_tab(self) -> QWidget:
        tabs = QTabWidget()

        train_tab = QWidget()
        layout = QVBoxLayout(train_tab)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        form = QFormLayout()
        form.setSpacing(10)
        self.ft_data_combo, data_row = self._path_row("recent_finetune_data", "file")
        form.addRow("数据文件:", data_row)
        self.ft_model_combo, model_row = self._path_row("recent_ft_pretrain_dir", "dir")
        form.addRow("预训练模型:", model_row)
        self.ft_out_combo, out_row = self._path_row("recent_finetune_dir", "dir")
        if not self.ft_out_combo.currentText().strip():
            self.ft_out_combo.setCurrentText(str(FINETUNE_DIR))
        form.addRow("微调输出目录:", out_row)

        label_row = QWidget()
        label_h = QHBoxLayout(label_row)
        label_h.setContentsMargins(0, 0, 0, 0)
        label_h.setSpacing(6)
        self.ft_label_combo = QComboBox()
        self.ft_label_combo.setEditable(True)
        self.ft_label_combo.addItem("RealValue", "RealValue")
        self.ft_label_combo.addItem("PredictedValue", "PredictedValue")
        self.ft_label_combo.addItem("Diameter", "Diameter")
        self.ft_label_index_spin = QSpinBox()
        self.ft_label_index_spin.setRange(1, 10)
        self.ft_label_index_spin.setValue(1)
        label_h.addWidget(self.ft_label_combo, 1)
        label_h.addWidget(QLabel("第"))
        label_h.addWidget(self.ft_label_index_spin)
        label_h.addWidget(QLabel("分量"))
        form.addRow("标签字段:", label_row)

        self.ft_epochs_spin = QSpinBox()
        self.ft_epochs_spin.setRange(1, 1000)
        self.ft_epochs_spin.setValue(100)
        form.addRow("训练轮数:", self.ft_epochs_spin)

        self.ft_batch_spin = QSpinBox()
        self.ft_batch_spin.setRange(16, 4096)
        self.ft_batch_spin.setValue(64)
        form.addRow("批大小:", self.ft_batch_spin)

        self.ft_lr_spin = QDoubleSpinBox()
        self.ft_lr_spin.setRange(0.00001, 1.0)
        self.ft_lr_spin.setDecimals(5)
        self.ft_lr_spin.setSingleStep(0.0001)
        self.ft_lr_spin.setValue(0.001)
        form.addRow("学习率:", self.ft_lr_spin)

        self.ft_ratio_spin = QSpinBox()
        self.ft_ratio_spin.setRange(10, 90)
        self.ft_ratio_spin.setValue(80)
        self.ft_ratio_spin.setSuffix(" %")
        form.addRow("训练比例:", self.ft_ratio_spin)

        self.ft_max_samples_spin = QSpinBox()
        self.ft_max_samples_spin.setRange(0, 24462)
        self.ft_max_samples_spin.setValue(0)
        self.ft_max_samples_spin.setSingleStep(100)
        self.ft_max_samples_spin.setSpecialValueText("全部")
        self.ft_max_samples_spin.setSuffix(" 个（0=全部）")
        form.addRow("微调样本数:", self.ft_max_samples_spin)

        self.ft_cv_spin = QSpinBox()
        self.ft_cv_spin.setRange(0, 10)
        self.ft_cv_spin.setValue(5)
        self.ft_cv_spin.setSpecialValueText("关闭")
        self.ft_cv_spin.setSuffix(" 折")
        form.addRow("交叉验证折数:", self.ft_cv_spin)

        self.ft_device_combo = QComboBox()
        self.ft_device_combo.addItem("自动检测", None)
        self.ft_device_combo.addItem("CPU", "cpu")
        self.ft_device_combo.addItem("GPU (CUDA)", "cuda")
        form.addRow("训练设备:", self.ft_device_combo)
        layout.addLayout(form)

        self.ft_freeze_cb = QCheckBox("冻结编码器（先只训回归头）")
        self.ft_freeze_cb.setChecked(True)
        self.ft_unfreeze_spin = QSpinBox()
        self.ft_unfreeze_spin.setRange(0, 500)
        self.ft_unfreeze_spin.setValue(30)
        self.ft_unfreeze_spin.setSpecialValueText("0=纯冻结")
        self.ft_unfreeze_spin.setSuffix(" 轮解冻联合微调")
        freeze_row = QWidget()
        freeze_h = QHBoxLayout(freeze_row)
        freeze_h.setContentsMargins(0, 0, 0, 0)
        freeze_h.setSpacing(6)
        freeze_h.addWidget(self.ft_freeze_cb)
        freeze_h.addWidget(QLabel("最后"))
        freeze_h.addWidget(self.ft_unfreeze_spin)
        freeze_h.addWidget(QLabel("解冻学习率:"))
        self.ft_unfreeze_lr_spin = QDoubleSpinBox()
        self.ft_unfreeze_lr_spin.setRange(0.0, 1.0)
        self.ft_unfreeze_lr_spin.setDecimals(5)
        self.ft_unfreeze_lr_spin.setSingleStep(0.0001)
        self.ft_unfreeze_lr_spin.setValue(0.0001)
        self.ft_unfreeze_lr_spin.setSpecialValueText("0=自动(学习率×0.1)")
        freeze_h.addWidget(self.ft_unfreeze_lr_spin)
        freeze_h.addStretch()
        layout.addWidget(freeze_row)

        btn_row = QHBoxLayout()
        self.finetune_btn = QPushButton("开始微调")
        self.finetune_btn.clicked.connect(self._start_finetune)
        self.ft_pause_btn = QPushButton("暂停微调")
        self.ft_pause_btn.setCheckable(True)
        self.ft_pause_btn.setEnabled(False)
        self.ft_pause_btn.toggled.connect(self._toggle_finetune_pause)
        btn_row.addWidget(self.finetune_btn)
        btn_row.addWidget(self.ft_pause_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.ft_progress = QProgressBar()
        self.ft_progress.setRange(0, 1)
        self.ft_progress.setValue(0)
        self.ft_progress.setFormat("就绪")
        layout.addWidget(self.ft_progress)

        self.ft_chart = QChart()
        self.ft_chart.setTitle("微调损失曲线")
        self.ft_chart_view = QChartView(self.ft_chart)
        self.ft_chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        layout.addWidget(self.ft_chart_view, 1)

        self.ft_result = QLabel("尚未微调")
        self.ft_result.setWordWrap(True)
        layout.addWidget(self.ft_result)

        self.ft_log = QPlainTextEdit()
        self.ft_log.setReadOnly(True)
        self.ft_log.setPlaceholderText("微调日志将显示在这里...")
        layout.addWidget(self.ft_log, 1)
        tabs.addTab(train_tab, "模型微调")

        eval_tab = QWidget()
        eval_layout = QVBoxLayout(eval_tab)
        eval_layout.setContentsMargins(20, 16, 20, 16)
        eval_layout.setSpacing(12)
        eval_form = QFormLayout()
        self.ft_eval_model_combo, em_row = self._path_row("recent_finetune_model", "dir")
        eval_form.addRow("微调模型目录:", em_row)
        self.ft_eval_data_combo, ed_row = self._path_row("recent_ft_eval_data", "file")
        eval_form.addRow("数据文件:", ed_row)
        eval_layout.addLayout(eval_form)
        self.eval_btn = QPushButton("预测并评估")
        self.eval_btn.clicked.connect(self._start_ft_eval)
        eval_layout.addWidget(self.eval_btn)
        self.ft_eval_metric = QLabel("尚未评估")
        self.ft_eval_metric.setWordWrap(True)
        eval_layout.addWidget(self.ft_eval_metric)
        self.ft_eval_tabs = QTabWidget()
        self.ft_eval_scatter_chart = QChart()
        self.ft_eval_scatter_view = QChartView(self.ft_eval_scatter_chart)
        self.ft_eval_scatter_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.ft_eval_resid_chart = QChart()
        self.ft_eval_resid_view = QChartView(self.ft_eval_resid_chart)
        self.ft_eval_resid_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.ft_eval_hist_chart = QChart()
        self.ft_eval_hist_view = QChartView(self.ft_eval_hist_chart)
        self.ft_eval_hist_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.ft_eval_tabs.addTab(self.ft_eval_scatter_view, "预测 vs 真实")
        self.ft_eval_tabs.addTab(self.ft_eval_resid_view, "残差图")
        self.ft_eval_tabs.addTab(self.ft_eval_hist_view, "误差分布")
        eval_layout.addWidget(self.ft_eval_tabs, 1)
        tabs.addTab(eval_tab, "预测评估")

        return tabs

    # ---------- 训练 / 预测 ----------
    def _start_train(self) -> None:
        worker = getattr(self, "_train_worker", None)
        if worker is not None and worker.isRunning():
            QMessageBox.warning(self, "提示", "训练正在进行中，请等待完成后再试。")
            return
        data = self.train_data_combo.currentText().strip()
        out = self.train_out_combo.currentText().strip()
        if not data:
            QMessageBox.warning(self, "提示", "请选择数据文件")
            return
        if not out:
            QMessageBox.warning(self, "提示", "请选择模型输出目录")
            return
        self._remember_path(self.train_data_combo, "recent_train_data", data)
        self._remember_path(self.train_out_combo, "recent_train_out_dir", out)

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
        self.train_btn.setEnabled(False)

        self.train_progress.setRange(0, self.epochs_spin.value())
        self.train_progress.setValue(0)
        self._init_train_chart()

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
            spectrum_column=self.spectrum_combo.currentData(),
            recon_top_n=self.recon_top_spin.value(),
        )
        self._train_worker.log_signal.connect(self.train_log.appendPlainText)
        self._train_worker.epoch_signal.connect(self._on_train_epoch)
        self._train_worker.done_signal.connect(self._on_train_done)
        self._train_worker.error_signal.connect(self._on_task_error)
        self._train_worker.start()

    def _on_train_done(self, summary: dict) -> None:
        self.pause_btn.setChecked(False)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("暂停训练")
        self.train_btn.setEnabled(True)
        self.train_progress.setValue(self.train_progress.maximum())
        self.train_progress.setFormat("完成")
        self.train_btn.setEnabled(True)
        brief = {k: v for k, v in summary.items() if k not in ("history", "recon_samples")}
        self.train_log.appendPlainText(f"\n训练完成：{brief}\n")
        self._plot_train_history(summary.get("history"))
        self._plot_recon_comparison(summary.get("recon_samples"))
        out_dir = summary.get("out_dir")
        if out_dir:
            # 自动填入预测页签的模型目录并记忆，便于直接使用新训练模型
            self.pred_model_combo.setCurrentText(out_dir)
            self._remember_path(self.pred_model_combo, "recent_pred_model_dir", out_dir)
            self._save_chart_png(self.train_chart_view, Path(out_dir) / "loss_curve.png")
            self._save_chart_png(self.train_recon_view, Path(out_dir) / "recon_comparison.png")
        QMessageBox.information(
            self, "完成", f"训练完成，模型已保存到：\n{summary.get('out_dir')}"
        )

    def _append_recon_pair(self, chart: QChart, s: dict, color: QColor, name: str) -> None:
        """向图表追加一对原始/重建光谱曲线（原始实线、重建虚线）。"""
        orig = QLineSeries()
        orig.setName(f"{name} - 原始")
        pen_orig = QPen(color)
        pen_orig.setWidth(2)
        orig.setPen(pen_orig)
        recon = QLineSeries()
        recon.setName(f"{name} - 重建")
        pen_recon = QPen(color)
        pen_recon.setWidth(1)
        pen_recon.setStyle(Qt.PenStyle.DashLine)
        recon.setPen(pen_recon)
        for i, v in enumerate(s["original"], 1):
            orig.append(float(i), float(v))
        for i, v in enumerate(s["recon"], 1):
            recon.append(float(i), float(v))
        chart.addSeries(orig)
        chart.addSeries(recon)

    def _plot_recon_comparison(self, recon_samples: dict | None) -> None:
        """重建光谱对比图：原始 vs 重建，越重合说明重建越好。"""
        if not recon_samples:
            return
        chart = QChart()
        chart.setTitle("重建光谱对比（原始 vs 重建，越重合越好）")
        worst_list = recon_samples.get("worst") or []
        for rank, s in enumerate(worst_list, 1):
            self._append_recon_pair(chart, s, QColor("#D64545"), f"误差第{rank}大")
        best = recon_samples.get("best")
        if best:
            self._append_recon_pair(chart, best, QColor("#3BA776"), "误差最小")
        chart.createDefaultAxes()
        x_axes = chart.axes(Qt.Orientation.Horizontal)
        y_axes = chart.axes(Qt.Orientation.Vertical)
        if x_axes:
            x_axes[0].setTitleText("波长序号")
        if y_axes:
            y_axes[0].setTitleText("光谱强度")
        self.train_recon_view.setChart(chart)

    def _apply_chart_cn_font(self, chart: QChart) -> None:
        """给图表所有文本元素设置中文字体，避免离屏保存 PNG 时中文乱码。"""
        font = QFont("Microsoft YaHei", 10)
        chart.setTitleFont(font)
        legend = chart.legend()
        if legend is not None:
            legend.setFont(font)
        for axis in chart.axes():
            axis.setTitleFont(font)
            axis.setLabelsFont(font)

    def _save_chart_png(self, view: QChartView, path: Path, width: int = 1600, height: int = 1000) -> None:
        """把图表离屏渲染为固定尺寸 PNG（显式中文字体 + 白底）。

        直接离屏渲染图表场景而非 grab 屏幕 widget，避免：
        ① 中文因字体回退失败而乱码；② 子页签未激活导致渲染尺寸/内容与界面不一致。
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            chart = view.chart()
            self._apply_chart_cn_font(chart)
            # 按目标尺寸重排图表，保证渲染填满且与目标比例一致（不变形、无白边）
            chart.resize(width, height)
            image = QImage(width, height, QImage.Format.Format_ARGB32)
            image.fill(Qt.GlobalColor.white)
            painter = QPainter(image)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            view.render(painter, QRectF(0, 0, width, height))
            painter.end()
            if not image.save(str(path)):
                logger.warning(f"图表保存失败: {path}")
        except Exception as exc:  # noqa: BLE001 - 图表保存失败不应中断主流程
            logger.warning(f"图表保存异常: {path}: {exc}")

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

    def _init_train_chart(self) -> None:
        """训练开始时初始化空曲线，用于逐 epoch 实时更新。"""
        chart = QChart()
        chart.setTitle("训练损失曲线")
        self._train_series = QLineSeries()
        self._train_series.setName("训练损失")
        self._train_series.setColor(QColor("#D8A24A"))
        self._val_series = QLineSeries()
        self._val_series.setName("验证损失")
        self._val_series.setColor(QColor("#4A90D8"))
        chart.addSeries(self._train_series)
        chart.addSeries(self._val_series)
        chart.createDefaultAxes()
        self.train_chart_view.setChart(chart)

    def _on_train_epoch(self, epoch: int, epochs: int, train_loss: float, val_loss, metrics=None) -> None:
        """每个 epoch 结束实时更新进度条与损失曲线。"""
        self.train_progress.setRange(0, epochs)
        self.train_progress.setValue(epoch)
        self.train_progress.setFormat(f"epoch {epoch}/{epochs}  %p%")

        self._train_series.append(float(epoch), float(train_loss))
        if val_loss is not None:
            self._val_series.append(float(epoch), float(val_loss))

        chart = self.train_chart_view.chart()
        x_axes = chart.axes(Qt.Orientation.Horizontal)
        y_axes = chart.axes(Qt.Orientation.Vertical)
        if x_axes:
            x_axes[0].setRange(1, epochs)
        if y_axes:
            points = self._train_series.pointsVector()
            max_y = max(p.y() for p in points) if points else 1.0
            y_axes[0].setRange(0, max_y * 1.1)

    def _start_predict(self) -> None:
        model_dir = self.pred_model_combo.currentText().strip()
        data = self.pred_data_combo.currentText().strip()
        if not model_dir:
            QMessageBox.warning(self, "提示", "请选择模型目录")
            return
        if not data:
            QMessageBox.warning(self, "提示", "请选择数据文件")
            return
        self._remember_path(self.pred_model_combo, "recent_pred_model_dir", model_dir)
        self._remember_path(self.pred_data_combo, "recent_predict_data", data)

        self.pred_log.clear()
        self.pred_log.appendPlainText(f"开始预测：{data}\n")
        self._predict_worker = ModelWorker(
            run_prediction,
            data_path=data,
            model_dir=model_dir,
        )
        self._predict_worker.log_signal.connect(self.pred_log.appendPlainText)
        self._predict_worker.done_signal.connect(self._on_predict_done)
        self._predict_worker.error_signal.connect(self._on_task_error)
        self._predict_worker.start()

    def _on_predict_done(self, summary: dict) -> None:
        self.pred_log.appendPlainText(f"\n预测完成：{summary}\n")
        self._plot_pred_histogram(summary)
        out_dir = summary.get("out_dir")
        if out_dir:
            self._save_chart_png(
                self.pred_chart_view, Path(out_dir) / "reconstruction_error_histogram.png"
            )
        self.pred_metric.setText(
            f"样本 {summary['samples']} | 均值 {summary['recon_err_mean']:.4f} | "
            f"中位数 {summary['recon_err_median']:.4f} | "
            f"95分位 {summary['recon_err_p95']:.4f} | "
            f"99分位 {summary['recon_err_p99']:.4f} | "
            f"异常 {summary['anomaly_count']} 个"
        )
        QMessageBox.information(
            self, "完成", f"预测完成，结果已保存到：\n{summary.get('out_path')}"
        )

    def _plot_pred_histogram(self, summary: dict) -> None:
        """把重构误差分布画成直方图（对数横轴）。"""
        centers = summary.get("hist_centers") or []
        counts = summary.get("hist_counts") or []
        if not centers:
            return
        bar_set = QBarSet("样本数")
        bar_set.setColor(QColor("#D8A24A"))
        labels = []
        for c, n in zip(centers, counts):
            bar_set.append(float(n))
            labels.append(f"{c:.2g}")
        series = QBarSeries()
        series.append(bar_set)

        chart = QChart()
        chart.addSeries(series)
        chart.setTitle("重构误差分布（对数横轴，右侧为异常尾）")
        axis_x = QBarCategoryAxis()
        axis_x.append(labels)
        chart.addAxis(axis_x, Qt.AlignBottom)
        series.attachAxis(axis_x)
        axis_y = QValueAxis()
        axis_y.setLabelFormat("%.0f")
        chart.addAxis(axis_y, Qt.AlignLeft)
        series.attachAxis(axis_y)
        self.pred_chart_view.setChart(chart)

    def _start_finetune(self) -> None:
        worker = getattr(self, "_ft_worker", None)
        if worker is not None and worker.isRunning():
            QMessageBox.warning(self, "提示", "微调正在进行中，请等待完成后再试。")
            return
        data = self.ft_data_combo.currentText().strip()
        model_dir = self.ft_model_combo.currentText().strip()
        out = self.ft_out_combo.currentText().strip()
        if not data:
            QMessageBox.warning(self, "提示", "请选择数据文件")
            return
        if not model_dir:
            QMessageBox.warning(self, "提示", "请选择预训练模型目录")
            return
        if not out:
            QMessageBox.warning(self, "提示", "请选择微调输出目录")
            return
        self._remember_path(self.ft_data_combo, "recent_finetune_data", data)
        self._remember_path(self.ft_model_combo, "recent_ft_pretrain_dir", model_dir)
        self._remember_path(self.ft_out_combo, "recent_finetune_dir", out)

        self.ft_log.clear()
        self.ft_log.appendPlainText(f"开始微调：{data}\n")

        max_samples = self.ft_max_samples_spin.value()
        self.ft_pause_btn.setChecked(False)
        self.ft_pause_btn.setEnabled(True)
        self.ft_pause_btn.setText("暂停微调")
        self.finetune_btn.setEnabled(False)
        self.ft_progress.setRange(0, self.ft_epochs_spin.value())
        self.ft_progress.setValue(0)
        self._init_finetune_chart()

        self._ft_worker = ModelWorker(
            run_finetune,
            data_path=data,
            model_dir=model_dir,
            out_dir=out,
            label_column=self.ft_label_combo.currentText().strip() or "RealValue",
            label_index=self.ft_label_index_spin.value() - 1,
            epochs=self.ft_epochs_spin.value(),
            batch_size=self.ft_batch_spin.value(),
            lr=self.ft_lr_spin.value(),
            freeze=self.ft_freeze_cb.isChecked(),
            unfreeze_epochs=self.ft_unfreeze_spin.value() if self.ft_freeze_cb.isChecked() else 0,
            unfreeze_lr=self.ft_unfreeze_lr_spin.value() or None,
            train_ratio=self.ft_ratio_spin.value() / 100.0,
            max_samples=max_samples or None,
            k_folds=self.ft_cv_spin.value(),
            device=self.ft_device_combo.currentData(),
        )
        self._ft_worker.log_signal.connect(self.ft_log.appendPlainText)
        self._ft_worker.epoch_signal.connect(self._on_finetune_epoch)
        self._ft_worker.done_signal.connect(self._on_finetune_done)
        self._ft_worker.error_signal.connect(self._on_finetune_error)
        self._ft_worker.start()

    def _on_finetune_done(self, summary: dict) -> None:
        self.ft_pause_btn.setChecked(False)
        self.ft_pause_btn.setEnabled(False)
        self.ft_pause_btn.setText("暂停微调")
        self.finetune_btn.setEnabled(True)
        self.ft_progress.setValue(self.ft_progress.maximum())
        self.ft_progress.setFormat("完成")
        # 先保存实时图（含验证 R² 双轴），再覆盖为静态损失曲线
        out_dir = summary.get("out_dir")
        if out_dir:
            self._save_chart_png(self.ft_chart_view, Path(out_dir) / "loss_curve.png")
        self._plot_finetune_history(summary.get("history"))
        cv = summary.get("cv")
        cv_text = ""
        if cv:
            cv_text = (
                f"\n{cv['k']} 折交叉验证: RMSE {cv['mean']['rmse']:.4f}±{cv['std']['rmse']:.4f} | "
                f"R² {cv['mean']['r2']:.4f}±{cv['std']['r2']:.4f} | corr {cv['mean']['corr']:.4f}±{cv['std']['corr']:.4f}"
            )
        self.ft_result.setText(
            f"样本 {summary['samples']}（训练 {summary['train_samples']} / 验证 {summary['val_samples']}） | "
            f"验证 RMSE {summary['rmse']:.4f} | MAE {summary['mae']:.4f} | R² {summary['r2']:.4f} | "
            f"corr {summary.get('corr', 0.0):.4f} | "
            f"{'冻结编码器' if summary['freeze'] else '全模型微调'}" + cv_text
        )
        QMessageBox.information(
            self, "完成", f"微调完成，模型已保存到：\n{summary.get('out_dir')}"
        )
        # 自动填入评估区的微调模型目录并记忆，方便直接预测评估
        if out_dir:
            self.ft_eval_model_combo.setCurrentText(out_dir)
            self._remember_path(self.ft_eval_model_combo, "recent_finetune_model", out_dir)

    def _start_ft_eval(self) -> None:
        worker = getattr(self, "_ft_eval_worker", None)
        if worker is not None and worker.isRunning():
            QMessageBox.warning(self, "提示", "预测评估正在进行中，请等待完成。")
            return
        model_dir = self.ft_eval_model_combo.currentText().strip()
        data = self.ft_eval_data_combo.currentText().strip()
        if not model_dir:
            QMessageBox.warning(self, "提示", "请选择微调模型目录")
            return
        if not data:
            QMessageBox.warning(self, "提示", "请选择数据文件")
            return
        self._remember_path(self.ft_eval_model_combo, "recent_finetune_model", model_dir)
        self._remember_path(self.ft_eval_data_combo, "recent_ft_eval_data", data)
        self.ft_eval_metric.setText("正在评估...")
        self.eval_btn.setEnabled(False)
        self._ft_eval_worker = ModelWorker(
            run_finetune_predict,
            data_path=data,
            model_dir=model_dir,
        )
        self._ft_eval_worker.log_signal.connect(self.ft_log.appendPlainText)
        self._ft_eval_worker.done_signal.connect(self._on_ft_eval_done)
        self._ft_eval_worker.error_signal.connect(self._on_finetune_error)
        self._ft_eval_worker.start()

    def _on_ft_eval_done(self, summary: dict) -> None:
        self.eval_btn.setEnabled(True)
        metrics = summary.get("metrics")
        scatters = summary.get("scatters")
        if metrics:
            self.ft_eval_metric.setText(
                f"样本 {summary['samples']} | RMSE {metrics['rmse']:.4f} | MAE {metrics['mae']:.4f} | "
                f"R² {metrics['r2']:.4f} | 相关系数 {metrics['corr']:.4f} | MAPE {metrics['mape']:.2f}% | "
                f"偏差 {metrics['bias']:+.4f}\n"
                f"误差: 均值 {metrics['bias']:+.4f}, 标准差 {metrics['err_std']:.4f}, "
                f"5%~95% 区间 [{metrics['err_p5']:+.3f}, {metrics['err_p95']:+.3f}]\n"
                f"准确度: ±0.5 度内 {metrics.get('within_0p5', 0.0):.1%}，±1 度内 {metrics.get('within_1p0', 0.0):.1%}"
            )
        else:
            self.ft_eval_metric.setText(
                f"样本 {summary['samples']} | 数据无标签列，仅输出预测值\n"
                f"预测均值 {summary['mean']:.3f}，标准差 {summary['std']:.3f}，"
                f"范围 [{summary['min']:.3f}, {summary['max']:.3f}]"
            )
        if scatters:
            self._plot_ft_eval_scatter(scatters)
            self._plot_ft_residual(scatters)
            self._plot_ft_error_hist(scatters)
            out_dir = summary.get("out_dir")
            if out_dir:
                self._save_chart_png(
                    self.ft_eval_scatter_view, Path(out_dir) / "scatter_pred_vs_true.png"
                )
                self._save_chart_png(
                    self.ft_eval_resid_view, Path(out_dir) / "residual.png"
                )
                self._save_chart_png(
                    self.ft_eval_hist_view, Path(out_dir) / "error_histogram.png"
                )
        QMessageBox.information(
            self, "完成", f"预测完成，结果已保存到：\n{summary.get('out_path')}"
        )

    def _plot_ft_eval_scatter(self, scatters: dict) -> None:
        """预测值 vs 真实值散点图 + y=x 理想线。"""
        y_true = scatters["y_true"]
        y_pred = scatters["y_pred"]
        chart = QChart()
        chart.setTitle("预测值 vs 真实值（越接近对角线越好）")
        series = QScatterSeries()
        series.setName("样本")
        series.setMarkerSize(5)
        series.setColor(QColor("#D8A24A"))
        for t, p in zip(y_true, y_pred):
            series.append(float(t), float(p))
        chart.addSeries(series)
        lo = min(min(y_true), min(y_pred))
        hi = max(max(y_true), max(y_pred))
        line = QLineSeries()
        line.setName("y=x 理想线")
        line.setColor(QColor("#D64545"))
        line.append(float(lo), float(lo))
        line.append(float(hi), float(hi))
        chart.addSeries(line)
        chart.createDefaultAxes()
        x_axes = chart.axes(Qt.Orientation.Horizontal)
        y_axes = chart.axes(Qt.Orientation.Vertical)
        if x_axes:
            x_axes[0].setTitleText("真实值")
        if y_axes:
            y_axes[0].setTitleText("预测值")
        self.ft_eval_scatter_view.setChart(chart)

    def _plot_ft_residual(self, scatters: dict) -> None:
        """残差图：误差（预测-真实）随真实值的变化，应围绕 0 随机分布。"""
        y_true = scatters["y_true"]
        y_pred = scatters["y_pred"]
        chart = QChart()
        chart.setTitle("残差图（误差 = 预测 - 真实，应围绕 0 随机分布）")
        series = QScatterSeries()
        series.setName("残差")
        series.setMarkerSize(5)
        series.setColor(QColor("#4A90D8"))
        for t, p in zip(y_true, y_pred):
            series.append(float(t), float(p - t))
        chart.addSeries(series)
        zero = QLineSeries()
        zero.setName("零线")
        zero.setColor(QColor("#D64545"))
        zero.append(float(min(y_true)), 0.0)
        zero.append(float(max(y_true)), 0.0)
        chart.addSeries(zero)
        chart.createDefaultAxes()
        x_axes = chart.axes(Qt.Orientation.Horizontal)
        y_axes = chart.axes(Qt.Orientation.Vertical)
        if x_axes:
            x_axes[0].setTitleText("真实值")
        if y_axes:
            y_axes[0].setTitleText("残差（预测-真实）")
        self.ft_eval_resid_view.setChart(chart)

    def _plot_ft_error_hist(self, scatters: dict) -> None:
        """误差分布直方图，应近似以 0 为中心的钟形。"""
        y_true = scatters["y_true"]
        y_pred = scatters["y_pred"]
        import numpy as np

        errs = np.array([p - t for t, p in zip(y_true, y_pred)], dtype=float)
        counts, edges = np.histogram(errs, bins=20)
        bar_set = QBarSet("样本数")
        bar_set.setColor(QColor("#D8A24A"))
        labels = []
        for i, c in enumerate(counts):
            bar_set.append(float(c))
            labels.append(f"{edges[i]:.2f}")
        series = QBarSeries()
        series.append(bar_set)
        chart = QChart()
        chart.addSeries(series)
        chart.setTitle("误差分布（应近似以 0 为中心的钟形）")
        axis_x = QBarCategoryAxis()
        axis_x.append(labels)
        chart.addAxis(axis_x, Qt.AlignBottom)
        series.attachAxis(axis_x)
        axis_y = QValueAxis()
        chart.addAxis(axis_y, Qt.AlignLeft)
        series.attachAxis(axis_y)
        self.ft_eval_hist_view.setChart(chart)

    def _toggle_finetune_pause(self, checked: bool) -> None:
        worker = getattr(self, "_ft_worker", None)
        if worker is None:
            return
        if checked:
            worker.pause_event.set()
            self.ft_pause_btn.setText("继续微调")
        else:
            worker.pause_event.clear()
            self.ft_pause_btn.setText("暂停微调")

    def _init_finetune_chart(self) -> None:
        chart = QChart()
        chart.setTitle("微调损失与验证 R² 曲线")
        self._ft_train_series = QLineSeries()
        self._ft_train_series.setName("训练损失")
        self._ft_train_series.setColor(QColor("#D8A24A"))
        self._ft_val_series = QLineSeries()
        self._ft_val_series.setName("验证损失")
        self._ft_val_series.setColor(QColor("#4A90D8"))
        self._ft_val_r2_series = QLineSeries()
        self._ft_val_r2_series.setName("验证 R²")
        self._ft_val_r2_series.setColor(QColor("#3BA776"))
        chart.addSeries(self._ft_train_series)
        chart.addSeries(self._ft_val_series)
        chart.addSeries(self._ft_val_r2_series)
        self._ft_axis_x = QValueAxis()
        self._ft_axis_x.setTitleText("训练轮数")
        self._ft_axis_loss = QValueAxis()
        self._ft_axis_loss.setTitleText("MSE 损失")
        self._ft_axis_r2 = QValueAxis()
        self._ft_axis_r2.setTitleText("验证 R²")
        chart.addAxis(self._ft_axis_x, Qt.AlignBottom)
        chart.addAxis(self._ft_axis_loss, Qt.AlignLeft)
        chart.addAxis(self._ft_axis_r2, Qt.AlignRight)
        self._ft_train_series.attachAxis(self._ft_axis_x)
        self._ft_val_series.attachAxis(self._ft_axis_x)
        self._ft_val_r2_series.attachAxis(self._ft_axis_x)
        self._ft_train_series.attachAxis(self._ft_axis_loss)
        self._ft_val_series.attachAxis(self._ft_axis_loss)
        self._ft_val_r2_series.attachAxis(self._ft_axis_r2)
        self.ft_chart_view.setChart(chart)

    def _on_finetune_epoch(self, epoch: int, epochs: int, train_loss: float, val_loss, metrics=None) -> None:
        self.ft_progress.setRange(0, epochs)
        self.ft_progress.setValue(epoch)
        self.ft_progress.setFormat(f"epoch {epoch}/{epochs}  %p%")
        self._ft_train_series.append(float(epoch), float(train_loss))
        if val_loss is not None:
            self._ft_val_series.append(float(epoch), float(val_loss))
        if metrics is not None:
            self._ft_val_r2_series.append(float(epoch), float(metrics.get("r2", 0.0)))
        self._ft_axis_x.setRange(1, epochs)
        train_pts = self._ft_train_series.pointsVector()
        if train_pts:
            self._ft_axis_loss.setRange(0, max(p.y() for p in train_pts) * 1.1)
        r2_pts = self._ft_val_r2_series.pointsVector()
        if r2_pts:
            vals = [p.y() for p in r2_pts]
            self._ft_axis_r2.setRange(min(0.0, min(vals)) - 0.05, max(1.0, max(vals)) + 0.05)

    def _plot_finetune_history(self, history: dict | None) -> None:
        if not history:
            return
        train_vals = history.get("train") or []
        val_vals = history.get("val") or []
        chart = QChart()
        chart.setTitle("微调损失曲线")
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
        self.ft_chart_view.setChart(chart)

    def _on_finetune_error(self, msg: str) -> None:
        self.ft_pause_btn.setChecked(False)
        self.ft_pause_btn.setEnabled(False)
        self.ft_pause_btn.setText("暂停微调")
        self.finetune_btn.setEnabled(True)
        QMessageBox.critical(self, "错误", msg)

    def _on_task_error(self, msg: str) -> None:
        self.pause_btn.setChecked(False)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("暂停训练")
        self.train_btn.setEnabled(True)
        QMessageBox.critical(self, "错误", msg)

    def _build_visualize_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        form = QFormLayout()
        form.setSpacing(10)
        self.vis_model_combo, model_row = self._path_row("recent_vis_model_dir", "dir")
        form.addRow("模型目录:", model_row)
        self.vis_data_combo, data_row = self._path_row("recent_visualize_data", "file")
        form.addRow("数据文件:", data_row)

        self.vis_percentile_spin = QDoubleSpinBox()
        self.vis_percentile_spin.setRange(90.0, 100.0)
        self.vis_percentile_spin.setDecimals(2)
        self.vis_percentile_spin.setSingleStep(0.05)
        self.vis_percentile_spin.setValue(99.0)
        self.vis_percentile_spin.setSuffix(" %")
        form.addRow("异常阈值:", self.vis_percentile_spin)

        self.vis_samples_spin = QSpinBox()
        self.vis_samples_spin.setRange(100, 24462)
        self.vis_samples_spin.setValue(5000)
        self.vis_samples_spin.setSingleStep(500)
        self.vis_samples_spin.setSuffix(" 个")
        form.addRow("最大样本数:", self.vis_samples_spin)

        self.vis_dim_combo = QComboBox()
        self.vis_dim_combo.addItem("2D")
        self.vis_dim_combo.addItem("3D")
        self.vis_dim_combo.currentTextChanged.connect(self._on_vis_dim_changed)
        form.addRow("维度:", self.vis_dim_combo)
        layout.addLayout(form)

        run_btn = QPushButton("开始分析")
        run_btn.clicked.connect(self._start_visualize)
        layout.addWidget(run_btn)

        self.vis_chart = QChart()
        self.vis_chart.setTitle("NIR 瓶颈特征 2D 可视化（PCA）")
        self.vis_chart_view = QChartView(self.vis_chart)
        self.vis_chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        layout.addWidget(self.vis_chart_view, 1)

        self.vis_scatter_3d = None
        try:
            self.vis_scatter_3d = PointCloud3DWidget()
            self.vis_scatter_3d.setVisible(False)
            layout.addWidget(self.vis_scatter_3d, 1)
        except Exception as exc:  # noqa: BLE001 - 环境不支持 OpenGL 时降级
            logger.warning(f"3D 可视化初始化失败，已降级为仅 2D: {exc}")
            self.vis_dim_combo.setEnabled(False)
            self.vis_dim_combo.setCurrentText("2D")

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
        self._remember_path(self.vis_model_combo, "recent_vis_model_dir", model_dir)
        self._remember_path(self.vis_data_combo, "recent_visualize_data", data)

        self.vis_info.setText("正在分析...")
        self._vis_worker = ModelWorker(
            run_visualize,
            data_path=data,
            model_dir=model_dir,
            percentile=self.vis_percentile_spin.value(),
            max_samples=self.vis_samples_spin.value(),
            n_components=3 if self.vis_dim_combo.currentText() == "3D" else 2,
        )
        self._vis_worker.done_signal.connect(self._on_visualize_done)
        self._vis_worker.error_signal.connect(self._on_task_error)
        self._vis_worker.start()

    def _on_visualize_done(self, summary: dict) -> None:
        coords = summary["coords"]
        flags = summary["is_anomaly"]

        # 2D（QScatterSeries）与 3D（自定义 OpenGL 点云）均可全量渲染
        n_comp = summary.get("n_components", 2)
        use_3d = n_comp == 3 and self.vis_scatter_3d is not None

        if use_3d:
            self._plot_scatter_3d(coords, flags)
        else:
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
            f"样本 {summary['samples']}，异常 {int(summary['is_anomaly'].sum())} 个"
            f"（阈值 {summary['threshold']:.4f}），"
            f"PCA 前 {n_comp} 主成分解释方差 {summary['explained']:.1%}"
        )

    def _on_vis_dim_changed(self, text: str) -> None:
        """切换 2D / 3D 视图。"""
        is_3d = text == "3D"
        self.vis_chart_view.setVisible(not is_3d)
        if self.vis_scatter_3d is not None:
            self.vis_scatter_3d.setVisible(is_3d)

    def _plot_scatter_3d(self, coords, flags) -> None:
        """用自定义 OpenGL 点云绘制三维散点图（PC1/PC2/PC3），支持全量样本。"""
        self.vis_scatter_3d.set_points(coords, flags)

    # ---------- 分布核查 ----------
    def _build_shift_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        form = QFormLayout()
        form.setSpacing(10)
        self.shift_pretrain_combo, p_row = self._path_row("recent_shift_pretrain", "file")
        form.addRow("预训练数据文件:", p_row)
        self.shift_label_combo, l_row = self._path_row("recent_shift_label", "file")
        form.addRow("标签数据文件:", l_row)

        self.shift_spectrum_combo = QComboBox()
        self.shift_spectrum_combo.addItem("SpectrumData（1024 维）", "SpectrumData")
        self.shift_spectrum_combo.addItem("Abs（301 维）", "Abs")
        form.addRow("光谱字段:", self.shift_spectrum_combo)
        layout.addLayout(form)

        pre_box = QGroupBox("对比前预处理（可选，消除散射/尺度后再比较）")
        pre_layout = QHBoxLayout(pre_box)
        self.shift_smooth_cb = QCheckBox("S-G 平滑")
        self.shift_smooth_cb.setChecked(True)
        self.shift_msc_cb = QCheckBox("MSC")
        self.shift_msc_cb.setChecked(True)
        self.shift_snv_cb = QCheckBox("SNV")
        for cb in (self.shift_smooth_cb, self.shift_msc_cb, self.shift_snv_cb):
            pre_layout.addWidget(cb)
        pre_layout.addStretch()
        layout.addWidget(pre_box)

        btn_row = QHBoxLayout()
        self.shift_btn = QPushButton("开始核查")
        self.shift_btn.clicked.connect(self._start_shift)
        btn_row.addWidget(self.shift_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.shift_result = QLabel("尚未核查")
        self.shift_result.setWordWrap(True)
        layout.addWidget(self.shift_result)

        self.shift_tabs = QTabWidget()
        self.shift_wave_chart = QChart()
        self.shift_wave_view = QChartView(self.shift_wave_chart)
        self.shift_wave_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.shift_pca_chart = QChart()
        self.shift_pca_view = QChartView(self.shift_pca_chart)
        self.shift_pca_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.shift_hist_chart = QChart()
        self.shift_hist_view = QChartView(self.shift_hist_chart)
        self.shift_hist_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.shift_tabs.addTab(self.shift_wave_view, "平均光谱对比")
        self.shift_tabs.addTab(self.shift_pca_view, "PCA 重叠")
        self.shift_tabs.addTab(self.shift_hist_view, "标签分布")
        layout.addWidget(self.shift_tabs, 1)

        self.shift_log = QPlainTextEdit()
        self.shift_log.setReadOnly(True)
        self.shift_log.setPlaceholderText("核查日志将显示在这里...")
        layout.addWidget(self.shift_log, 1)
        return tab

    def _start_shift(self) -> None:
        worker = getattr(self, "_shift_worker", None)
        if worker is not None and worker.isRunning():
            QMessageBox.warning(self, "提示", "分布核查正在进行中，请等待完成。")
            return
        pretrain = self.shift_pretrain_combo.currentText().strip()
        label = self.shift_label_combo.currentText().strip()
        if not pretrain:
            QMessageBox.warning(self, "提示", "请选择预训练数据文件")
            return
        if not label:
            QMessageBox.warning(self, "提示", "请选择标签数据文件")
            return
        self._remember_path(self.shift_pretrain_combo, "recent_shift_pretrain", pretrain)
        self._remember_path(self.shift_label_combo, "recent_shift_label", label)

        preprocess = {
            "smooth": self.shift_smooth_cb.isChecked(),
            "msc": self.shift_msc_cb.isChecked(),
            "snv": self.shift_snv_cb.isChecked(),
        }
        self.shift_log.clear()
        self.shift_log.appendPlainText(f"开始分布核查：\n预训练 {pretrain}\n标签 {label}\n")
        self.shift_result.setText("正在核查...")
        self.shift_btn.setEnabled(False)
        self._shift_worker = ModelWorker(
            run_domain_shift,
            pretrain_path=pretrain,
            label_path=label,
            spectrum_column=self.shift_spectrum_combo.currentData(),
            preprocess=preprocess,
        )
        self._shift_worker.log_signal.connect(self.shift_log.appendPlainText)
        self._shift_worker.done_signal.connect(self._on_shift_done)
        self._shift_worker.error_signal.connect(self._on_shift_error)
        self._shift_worker.start()

    def _on_shift_done(self, summary: dict) -> None:
        self.shift_btn.setEnabled(True)
        ms = summary["mean_spectrum"]
        pca = summary["pca"]
        mah = summary["mahalanobis"]
        knn = summary["knn"]
        auc = summary["domain_classifier_auc"]
        ld = summary.get("label_dist") or {}
        text = (
            f"预训练 {summary['samples']['pretrain']} 样本 / 标签 {summary['samples']['label']} 样本\n"
            f"① 平均光谱相关系数 {ms['corr']:.3f}；全局均值 预训练 {ms['pretrain_global']['mean']:.0f} "
            f"vs 标签 {ms['label_global']['mean']:.0f}\n"
            f"② PCA 质心归一化距离 {pca['normalized_distance']:.2f}（<1 重叠，>3 分离）\n"
            f"③ 马氏距离超 95% 边界比例 {mah['outlier_ratio']:.0%}（同分布 ≈5%）\n"
            f"④ KNN 覆盖度比值 {knn['ratio']:.1f}（≈1 覆盖良好）\n"
            f"⑤ 领域判别器 AUC {auc:.3f}（≈0.5 同分布，>0.8 差异明显）"
        )
        if ld.get("predicted_value") and ld.get("real_value"):
            pv = ld["predicted_value"]
            rv = ld["real_value"]
            text += f"\n⑥ 仪器预测糖度 {pv['mean']:.2f} vs 真实糖度 {rv['mean']:.2f}"
        self.shift_result.setText(text)

        plot = summary.get("plot") or {}
        if plot.get("mean_wave"):
            self._plot_shift_mean_wave(plot["mean_wave"])
        if plot.get("pca"):
            self._plot_shift_pca(plot["pca"])
        if plot.get("pred_hist") or plot.get("real_hist"):
            self._plot_shift_label_hist(plot)
        self.shift_log.appendPlainText("核查完成。")

    def _plot_shift_mean_wave(self, wave: dict) -> None:
        """平均光谱对比图。"""
        chart = QChart()
        chart.setTitle("平均光谱对比（预训练 vs 标签，越重合越同分布）")
        p = QLineSeries()
        p.setName("预训练")
        p.setColor(QColor("#D8A24A"))
        for x, y in zip(wave["x"], wave["pretrain"]):
            p.append(float(x), float(y))
        l = QLineSeries()
        l.setName("标签")
        l.setColor(QColor("#4A90D8"))
        for x, y in zip(wave["x"], wave["label"]):
            l.append(float(x), float(y))
        chart.addSeries(p)
        chart.addSeries(l)
        chart.createDefaultAxes()
        x_axes = chart.axes(Qt.Orientation.Horizontal)
        y_axes = chart.axes(Qt.Orientation.Vertical)
        if x_axes:
            x_axes[0].setTitleText("波长序号")
        if y_axes:
            y_axes[0].setTitleText("光谱强度")
        self.shift_wave_view.setChart(chart)

    def _plot_shift_pca(self, pca: dict) -> None:
        """PCA 联合投影散点图（两批重叠=同分布，分离=偏移）。"""
        chart = QChart()
        chart.setTitle("PCA 联合投影（重叠=同分布，分离=分布偏移）")
        normal = QScatterSeries()
        normal.setName("预训练")
        normal.setMarkerSize(4)
        normal.setColor(QColor("#D8A24A"))
        for x, y in zip(pca["pretrain_x"], pca["pretrain_y"]):
            normal.append(float(x), float(y))
        label = QScatterSeries()
        label.setName("标签")
        label.setMarkerSize(7)
        label.setColor(QColor("#D64545"))
        for x, y in zip(pca["label_x"], pca["label_y"]):
            label.append(float(x), float(y))
        chart.addSeries(normal)
        chart.addSeries(label)
        chart.createDefaultAxes()
        x_axes = chart.axes(Qt.Orientation.Horizontal)
        y_axes = chart.axes(Qt.Orientation.Vertical)
        if x_axes:
            x_axes[0].setTitleText("PC1")
        if y_axes:
            y_axes[0].setTitleText("PC2")
        self.shift_pca_view.setChart(chart)

    def _plot_shift_label_hist(self, plot: dict) -> None:
        """标签分布对比图（仪器预测糖度 vs 真实糖度，归一化密度折线）。"""
        chart = QChart()
        chart.setTitle("标签分布对比（仪器预测糖度 vs 真实糖度）")
        for key, color, name in (
            ("pred_hist", QColor("#D8A24A"), "仪器预测糖度"),
            ("real_hist", QColor("#4A90D8"), "真实糖度"),
        ):
            hist = plot.get(key)
            if not hist:
                continue
            edges, counts = hist
            total = sum(counts) or 1
            series = QLineSeries()
            series.setName(name)
            series.setColor(color)
            for i, c in enumerate(counts):
                center = (edges[i] + edges[i + 1]) / 2.0
                series.append(float(center), float(c) / total)
            chart.addSeries(series)
        chart.createDefaultAxes()
        x_axes = chart.axes(Qt.Orientation.Horizontal)
        y_axes = chart.axes(Qt.Orientation.Vertical)
        if x_axes:
            x_axes[0].setTitleText("糖度")
        if y_axes:
            y_axes[0].setTitleText("占比")
        self.shift_hist_view.setChart(chart)

    def _on_shift_error(self, msg: str) -> None:
        self.shift_btn.setEnabled(True)
        self.shift_result.setText("核查失败")
        QMessageBox.critical(self, "错误", msg)

    # ---------- 数据分析（多批：分布 / 异常 / 相似性） ----------
    def _build_analysis_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        form = QFormLayout()
        form.setSpacing(10)
        self.ana_pretrain_combo, p_row = self._path_row("recent_ana_pretrain", "file")
        form.addRow("预训练数据文件:", p_row)
        self.ana_finetune_combo, f_row = self._path_row("recent_ana_finetune", "file")
        form.addRow("微调数据文件:", f_row)
        self.ana_predict_combo, pr_row = self._path_row("recent_ana_predict", "file")
        form.addRow("预测数据文件:", pr_row)
        layout.addLayout(form)

        pre_box = QGroupBox("预处理（可选，消除散射/尺度后再分析）")
        pre_layout = QHBoxLayout(pre_box)
        self.ana_smooth_cb = QCheckBox("S-G 平滑")
        self.ana_smooth_cb.setChecked(True)
        self.ana_msc_cb = QCheckBox("MSC")
        self.ana_msc_cb.setChecked(True)
        self.ana_snv_cb = QCheckBox("SNV")
        for cb in (self.ana_smooth_cb, self.ana_msc_cb, self.ana_snv_cb):
            pre_layout.addWidget(cb)
        pre_layout.addStretch()
        layout.addWidget(pre_box)

        btn_row = QHBoxLayout()
        self.ana_btn = QPushButton("开始分析")
        self.ana_btn.clicked.connect(self._start_analysis)
        btn_row.addWidget(self.ana_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.ana_report = QPlainTextEdit()
        self.ana_report.setReadOnly(True)
        self.ana_report.setPlaceholderText(
            "分析报告将显示在这里：各批数据的分布统计、离群（异常）样本、以及两两之间的相似性。"
        )
        layout.addWidget(self.ana_report, 1)
        return tab

    def _start_analysis(self) -> None:
        worker = getattr(self, "_ana_worker", None)
        if worker is not None and worker.isRunning():
            QMessageBox.warning(self, "提示", "数据分析正在进行中，请等待完成。")
            return
        datasets: dict = {}
        for name, combo, key in (
            ("预训练", self.ana_pretrain_combo, "recent_ana_pretrain"),
            ("微调", self.ana_finetune_combo, "recent_ana_finetune"),
            ("预测", self.ana_predict_combo, "recent_ana_predict"),
        ):
            path = combo.currentText().strip()
            if path:
                datasets[name] = path
                self._remember_path(combo, key, path)
        if len(datasets) < 2:
            QMessageBox.warning(self, "提示", "请至少选择 2 个数据文件")
            return

        preprocess = {
            "smooth": self.ana_smooth_cb.isChecked(),
            "msc": self.ana_msc_cb.isChecked(),
            "snv": self.ana_snv_cb.isChecked(),
        }
        self.ana_report.clear()
        self.ana_report.appendPlainText(
            "开始数据分析：\n" + "\n".join(f"  {k}: {v}" for k, v in datasets.items()) + "\n"
        )
        self.ana_btn.setEnabled(False)
        self._ana_worker = ModelWorker(
            run_data_analysis,
            datasets=datasets,
            spectrum_column="SpectrumData",
            preprocess=preprocess,
        )
        self._ana_worker.log_signal.connect(self.ana_report.appendPlainText)
        self._ana_worker.done_signal.connect(self._on_analysis_done)
        self._ana_worker.error_signal.connect(self._on_analysis_error)
        self._ana_worker.start()

    def _on_analysis_done(self, summary: dict) -> None:
        self.ana_btn.setEnabled(True)
        lines = ["", "===== 各数据分布与异常检测 ====="]
        for name, s in summary["summary"].items():
            lines.append(
                f"【{name}】样本 {s['n']} | 光谱均值 {s['spectrum_mean']:.0f} | "
                f"逐波长std均值 {s['spectrum_std_mean']:.0f}"
            )
            lines.append(f"  离群(异常)样本: {s['outliers']} 个（{s['outlier_ratio']:.1%}）")
            ls = s.get("label_stats")
            if ls:
                lines.append(
                    f"  标签 RealValue: 均值 {ls['mean']:.2f}, std {ls['std']:.3f}, "
                    f"范围 [{ls['min']:.1f}, {ls['max']:.1f}]"
                )
        lines.append("")
        lines.append("===== 两两相似性 =====")
        for p in summary["pairwise"]:
            lines.append(
                f"{p['a']} vs {p['b']}: 质心距离 {p['centroid_norm']:.2f}, "
                f"判别器 AUC {p['auc']:.3f}"
            )
        lines.append("")
        lines.append(
            "说明: 质心距离 <1 高度重叠，1~3 部分重叠，>3 明显分离；"
            "判别器 AUC ≈0.5 表示同分布，>0.8 表示明显分布差异。"
        )
        self.ana_report.appendPlainText("\n".join(lines))

    def _on_analysis_error(self, msg: str) -> None:
        self.ana_btn.setEnabled(True)
        QMessageBox.critical(self, "错误", msg)

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

        self._restore_train_config()

    def closeEvent(self, event: QCloseEvent) -> None:
        """关闭时保存窗口位置、当前文件路径、各页签路径与训练参数。"""
        self._memory.set("geometry", bytes(self.saveGeometry()).hex())
        self._memory.set("last_file", self.path_edit.text() or "")
        # 统一保存所有路径下拉框的当前值（含手动输入/粘贴的路径）
        for combo, key in self._path_combos:
            path = combo.currentText().strip()
            if path:
                self._memory.add_recent(key, path)
        self._save_train_config()
        self._memory.save()
        logger.info("关闭界面，已保存配置记忆")
        event.accept()

    def _save_train_config(self) -> None:
        """保存训练页签的所有参数到配置记忆。"""
        self._memory.set(
            "train_config",
            {
                "epochs": self.epochs_spin.value(),
                "batch_size": self.batch_spin.value(),
                "lr": self.lr_spin.value(),
                "bottleneck": self.bottleneck_spin.value(),
                "train_ratio": self.ratio_spin.value(),
                "device": self.device_combo.currentData(),
                "spectrum": self.spectrum_combo.currentData(),
                "pre_smooth": self.pre_smooth_cb.isChecked(),
                "pre_snv": self.pre_snv_cb.isChecked(),
                "pre_msc": self.pre_msc_cb.isChecked(),
                "pre_deriv1": self.pre_d1_cb.isChecked(),
                "pre_deriv2": self.pre_d2_cb.isChecked(),
                "outlier_enabled": self.outlier_cb.isChecked(),
                "outlier_k": self.outlier_k_spin.value(),
                "outlier_auto": self.outlier_auto_cb.isChecked(),
                "outlier_pct": self.outlier_pct_spin.value(),
                "recon_top": self.recon_top_spin.value(),
                "vis_percentile": self.vis_percentile_spin.value(),
                "vis_samples": self.vis_samples_spin.value(),
                "shift_spectrum": self.shift_spectrum_combo.currentData(),
                "shift_smooth": self.shift_smooth_cb.isChecked(),
                "shift_msc": self.shift_msc_cb.isChecked(),
                "shift_snv": self.shift_snv_cb.isChecked(),
                "ft_label_column": self.ft_label_combo.currentText(),
                "ft_label_index": self.ft_label_index_spin.value(),
                "ft_epochs": self.ft_epochs_spin.value(),
                "ft_batch": self.ft_batch_spin.value(),
                "ft_lr": self.ft_lr_spin.value(),
                "ft_ratio": self.ft_ratio_spin.value(),
                "ft_max_samples": self.ft_max_samples_spin.value(),
                "ft_freeze": self.ft_freeze_cb.isChecked(),
                "ft_unfreeze": self.ft_unfreeze_spin.value(),
                "ft_unfreeze_lr": self.ft_unfreeze_lr_spin.value(),
                "ft_cv": self.ft_cv_spin.value(),
                "ft_device": self.ft_device_combo.currentData(),
            },
        )

    def _restore_train_config(self) -> None:
        """恢复训练页签上次的参数。"""
        cfg = self._memory.get("train_config")
        if not isinstance(cfg, dict):
            return
        self.epochs_spin.setValue(int(cfg.get("epochs", self.epochs_spin.value())))
        self.batch_spin.setValue(int(cfg.get("batch_size", self.batch_spin.value())))
        self.lr_spin.setValue(float(cfg.get("lr", self.lr_spin.value())))
        self.bottleneck_spin.setValue(int(cfg.get("bottleneck", self.bottleneck_spin.value())))
        self.ratio_spin.setValue(int(cfg.get("train_ratio", self.ratio_spin.value())))
        dev_idx = self.device_combo.findData(cfg.get("device"))
        if dev_idx >= 0:
            self.device_combo.setCurrentIndex(dev_idx)
        spec_idx = self.spectrum_combo.findData(cfg.get("spectrum"))
        if spec_idx >= 0:
            self.spectrum_combo.setCurrentIndex(spec_idx)
        self.pre_smooth_cb.setChecked(bool(cfg.get("pre_smooth", False)))
        self.pre_snv_cb.setChecked(bool(cfg.get("pre_snv", False)))
        self.pre_msc_cb.setChecked(bool(cfg.get("pre_msc", False)))
        self.pre_d1_cb.setChecked(bool(cfg.get("pre_deriv1", False)))
        self.pre_d2_cb.setChecked(bool(cfg.get("pre_deriv2", False)))
        self.outlier_cb.setChecked(bool(cfg.get("outlier_enabled", False)))
        self.outlier_k_spin.setValue(int(cfg.get("outlier_k", self.outlier_k_spin.value())))
        self.outlier_pct_spin.setValue(float(cfg.get("outlier_pct", self.outlier_pct_spin.value())))
        self.outlier_auto_cb.setChecked(bool(cfg.get("outlier_auto", False)))
        self.recon_top_spin.setValue(int(cfg.get("recon_top", self.recon_top_spin.value())))
        self.vis_percentile_spin.setValue(float(cfg.get("vis_percentile", self.vis_percentile_spin.value())))
        self.vis_samples_spin.setValue(int(cfg.get("vis_samples", self.vis_samples_spin.value())))
        shift_spec_idx = self.shift_spectrum_combo.findData(cfg.get("shift_spectrum"))
        if shift_spec_idx >= 0:
            self.shift_spectrum_combo.setCurrentIndex(shift_spec_idx)
        self.shift_smooth_cb.setChecked(bool(cfg.get("shift_smooth", True)))
        self.shift_msc_cb.setChecked(bool(cfg.get("shift_msc", True)))
        self.shift_snv_cb.setChecked(bool(cfg.get("shift_snv", False)))
        ft_col = cfg.get("ft_label_column")
        if ft_col:
            ft_col_idx = self.ft_label_combo.findText(str(ft_col))
            if ft_col_idx >= 0:
                self.ft_label_combo.setCurrentIndex(ft_col_idx)
        self.ft_label_index_spin.setValue(int(cfg.get("ft_label_index", self.ft_label_index_spin.value())))
        self.ft_epochs_spin.setValue(int(cfg.get("ft_epochs", self.ft_epochs_spin.value())))
        self.ft_batch_spin.setValue(int(cfg.get("ft_batch", self.ft_batch_spin.value())))
        self.ft_lr_spin.setValue(float(cfg.get("ft_lr", self.ft_lr_spin.value())))
        self.ft_ratio_spin.setValue(int(cfg.get("ft_ratio", self.ft_ratio_spin.value())))
        self.ft_max_samples_spin.setValue(int(cfg.get("ft_max_samples", self.ft_max_samples_spin.value())))
        self.ft_freeze_cb.setChecked(bool(cfg.get("ft_freeze", True)))
        self.ft_unfreeze_spin.setValue(int(cfg.get("ft_unfreeze", self.ft_unfreeze_spin.value())))
        self.ft_unfreeze_lr_spin.setValue(float(cfg.get("ft_unfreeze_lr", self.ft_unfreeze_lr_spin.value())))
        self.ft_cv_spin.setValue(int(cfg.get("ft_cv", self.ft_cv_spin.value())))
        ft_dev_idx = self.ft_device_combo.findData(cfg.get("ft_device"))
        if ft_dev_idx >= 0:
            self.ft_device_combo.setCurrentIndex(ft_dev_idx)

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
