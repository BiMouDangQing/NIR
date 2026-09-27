"""「秋月梨」主题样式表（QSS）。

秋月梨（Pyrus pyrifolia cv. Qiuyue）果皮金黄带褐、口感脆甜，
本主题取其「金黄果皮 + 秋叶暖色 + 奶油底色」的配色：
  - 主色（果皮金棕） #D8A24A
  - 高亮（秋叶橙黄） #EBCB7E
  - 浅底（梨肉奶白） #FAF6EC
  - 文字（深棕）     #4A3A24
"""

QIYUE_THEME = """
* {
    font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
}

QMainWindow, QDialog {
    background-color: #FAF6EC;
}

QWidget#header {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #F3E2B3, stop:0.5 #EBCB7E, stop:1 #F3E2B3);
    border-bottom: 2px solid #D8A24A;
}

QLabel#appTitle {
    color: #4A3A24;
    font-size: 22px;
    font-weight: bold;
}

QLabel#appSubtitle {
    color: #7A6138;
    font-size: 12px;
}

QPushButton {
    background-color: #D8A24A;
    color: #FFFFFF;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    font-size: 13px;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #C58F3C;
}

QPushButton:pressed {
    background-color: #A8752F;
}

QLineEdit {
    background-color: #FFFFFF;
    border: 1px solid #E2D3AE;
    border-radius: 4px;
    padding: 6px;
    color: #4A3A24;
}

QGroupBox {
    border: 1px solid #E2D3AE;
    border-radius: 6px;
    margin-top: 12px;
    background-color: #FFFDF7;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: #8A6B2F;
    font-weight: bold;
}

QTableWidget {
    background-color: #FFFFFF;
    border: 1px solid #E2D3AE;
    gridline-color: #EFE6CF;
    color: #4A3A24;
    selection-background-color: #F3E2B3;
    selection-color: #4A3A24;
}

QHeaderView::section {
    background-color: #F3E2B3;
    color: #4A3A24;
    padding: 6px;
    border: 1px solid #E2D3AE;
    font-weight: bold;
}

QStatusBar {
    background-color: #F3E2B3;
    color: #7A6138;
}
"""
