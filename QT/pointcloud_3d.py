"""自定义 OpenGL 3D 点云渲染器。

用 GPU 点精灵（GL_POINTS + 圆形着色器）直接绘制全部样本，
替代 ``Q3DScatter``（后者在 2.4 万点以上会崩溃），支持鼠标旋转与滚轮缩放。
"""
from __future__ import annotations

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QMatrix4x4, QQuaternion, QSurfaceFormat, QVector3D
from PySide6.QtOpenGL import (
    QOpenGLBuffer,
    QOpenGLShader,
    QOpenGLShaderProgram,
    QOpenGLVertexArrayObject,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget

# OpenGL 常量（避免引入额外依赖）
GL_POINTS = 0x0000
GL_FLOAT = 0x1406
GL_DEPTH_TEST = 0x0B71
GL_COLOR_BUFFER_BIT = 0x00004000
GL_DEPTH_BUFFER_BIT = 0x00000100

_VERT = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;
layout(location = 2) in float aSize;
uniform mat4 uMVP;
out vec3 vColor;
void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
    gl_PointSize = aSize;
    vColor = aColor;
}
"""

_FRAG = """
#version 330 core
in vec3 vColor;
out vec4 FragColor;
void main() {
    vec2 c = gl_PointCoord - vec2(0.5);
    if (dot(c, c) > 0.25) discard;
    FragColor = vec4(vColor, 1.0);
}
"""

_NORMAL = np.array([216, 162, 74], dtype=np.float32) / 255.0  # #D8A24A
_ANOMALY = np.array([214, 69, 69], dtype=np.float32) / 255.0  # #D64545


class PointCloud3DWidget(QOpenGLWidget):
    """全量 3D 点云视图（PC1/PC2/PC3），鼠标拖动旋转、滚轮缩放。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.CoreProfile)
        fmt.setDepthBufferSize(24)
        self.setFormat(fmt)

        self._rot = QQuaternion()
        self._zoom = 1.0
        self._last = None
        self._program = None
        self._vbo = None
        self._vao = None
        self._n = 0
        self._dirty = False
        self._data = None
        self._broken = False

    def set_points(self, coords, flags) -> None:
        """设置点云数据：coords 为 (N, 3) 坐标，flags 为布尔数组标记异常点。"""
        pts = np.asarray(coords, dtype=np.float32)
        flags = np.asarray(flags, dtype=bool)
        if pts.ndim != 2 or pts.shape[1] < 3 or len(pts) != len(flags):
            return
        center = pts.mean(axis=0)
        scale = float(np.abs(pts - center).max()) or 1.0
        pts = (pts - center) / scale

        colors = np.where(flags[:, None], _ANOMALY, _NORMAL).astype(np.float32)
        sizes = np.where(flags, 7.0, 4.0).astype(np.float32)
        self._data = np.hstack([pts[:, :3], colors, sizes[:, None]]).astype(np.float32)
        self._n = len(self._data)
        self._dirty = True
        self.update()

    def initializeGL(self) -> None:
        gl = self.context().functions()
        gl.glClearColor(0.12, 0.12, 0.14, 1.0)
        gl.glEnable(GL_DEPTH_TEST)
        try:
            self._program = QOpenGLShaderProgram(self)
            self._program.addShaderFromSourceCode(QOpenGLShader.Vertex, _VERT)
            self._program.addShaderFromSourceCode(QOpenGLShader.Fragment, _FRAG)
            self._program.link()
            self._program.bind()

            self._vao = QOpenGLVertexArrayObject(self)
            self._vao.create()
            self._vao.bind()
            self._vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
            self._vbo.create()
            self._vbo.bind()
            self._vbo.setUsagePattern(QOpenGLBuffer.DynamicDraw)
            stride = 7 * 4
            self._program.enableAttributeArray(0)
            self._program.setAttributeBuffer(0, GL_FLOAT, 0, 3, stride)
            self._program.enableAttributeArray(1)
            self._program.setAttributeBuffer(1, GL_FLOAT, 3 * 4, 3, stride)
            self._program.enableAttributeArray(2)
            self._program.setAttributeBuffer(2, GL_FLOAT, 6 * 4, 1, stride)
            self._vao.release()
            self._vbo.release()
            self._program.release()
        except Exception:  # noqa: BLE001 - 着色器/OpenGL 初始化失败时静默降级
            self._broken = True
            self._program = None

    def resizeGL(self, w: int, h: int) -> None:
        self.context().functions().glViewport(0, 0, max(1, w), max(1, h))

    def paintGL(self) -> None:
        gl = self.context().functions()
        gl.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        if self._broken or self._program is None or self._n == 0:
            return

        self._program.bind()
        if self._dirty:
            self._vbo.bind()
            self._vbo.allocate(self._data.nbytes)
            self._vbo.write(0, self._data.tobytes(), self._data.nbytes)
            self._dirty = False

        self._program.setUniformValue("uMVP", self._compute_mvp())
        self._vao.bind()
        gl.glDrawArrays(GL_POINTS, 0, self._n)
        self._vao.release()
        self._program.release()

    def _compute_mvp(self) -> QMatrix4x4:
        proj = QMatrix4x4()
        proj.ortho(-2.5, 2.5, -2.5, 2.5, 0.1, 100.0)
        view = QMatrix4x4()
        view.lookAt(QVector3D(0.0, 0.0, 6.0), QVector3D(0.0, 0.0, 0.0), QVector3D(0.0, 1.0, 0.0))
        model = QMatrix4x4()
        model.rotate(self._rot)
        model.scale(self._zoom)
        return proj * view * model

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._last = event.position()

    def mouseReleaseEvent(self, event) -> None:
        self._last = None

    def mouseMoveEvent(self, event) -> None:
        if self._last is None:
            return
        pos = event.position()
        dx = pos.x() - self._last.x()
        dy = pos.y() - self._last.y()
        self._last = pos
        rot = QQuaternion.fromAxisAndAngle(0.0, 1.0, 0.0, dx * 0.5) * QQuaternion.fromAxisAndAngle(1.0, 0.0, 0.0, dy * 0.5)
        self._rot = rot * self._rot
        self.update()

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        self._zoom *= 1.1 if delta > 0 else 0.9
        self._zoom = max(0.2, min(10.0, self._zoom))
        self.update()
