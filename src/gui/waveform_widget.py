"""波形绘制与交互控件"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QPoint, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QMouseEvent, QPainter, QPen, QWheelEvent
from PyQt6.QtWidgets import QWidget

from src.core.waveform_data import WaveformData


class WaveformWidget(QWidget):
    """波形显示与交互控件。

    交互：
      - 单击           → seek_to 信号
      - 左键拖拽       → 平移视图
      - 拖拽分割点手柄 → 调整分割点
      - 双击           → 添加分割点
      - 右键分割点     → 删除
      - 滚轮           → 以鼠标为中心缩放
    """

    seek_to = pyqtSignal(float)              # 秒
    marker_moved = pyqtSignal(int, float)    # 索引, 秒
    marker_added = pyqtSignal(float)         # 秒
    marker_removed = pyqtSignal(int)         # 索引
    view_changed = pyqtSignal()

    HANDLE_W = 10
    HANDLE_H = 12
    MIN_VIEW_SEC = 0.5

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(220)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.ArrowCursor)

        self._wave: WaveformData | None = None
        self._markers: list[float] = []      # 分割点（秒，升序）
        self._playhead: float = 0.0

        self._view_start = 0.0
        self._view_dur = 10.0

        # 交互状态
        self._drag_mode: str | None = None   # 'pan' | 'marker' | None
        self._drag_marker_idx: int = -1
        self._drag_start_x: int = 0
        self._drag_start_view: float = 0.0
        self._drag_moved: bool = False
        self._hover_marker: int = -1

    # ---------- 数据接口 ----------

    def set_waveform(self, wave: WaveformData) -> None:
        self._wave = wave
        self._view_start = 0.0
        self._view_dur = wave.duration
        self._playhead = 0.0
        self.update()

    def set_markers(self, markers: list[float]) -> None:
        self._markers = sorted(markers)
        self.update()

    def markers(self) -> list[float]:
        return list(self._markers)

    def set_playhead(self, sec: float, follow: bool = False) -> None:
        self._playhead = sec
        if follow and self._wave:
            # 播放头居中
            if not (self._view_start <= sec <= self._view_start + self._view_dur):
                self._center_on(sec)
            else:
                half = self._view_dur / 2
                if abs(sec - (self._view_start + half)) > half * 0.6:
                    self._center_on(sec)
        self.update()

    def view_range(self) -> tuple[float, float]:
        return self._view_start, self._view_start + self._view_dur

    # ---------- 坐标换算 ----------

    def _sec_to_x(self, sec: float) -> float:
        if self._view_dur <= 0:
            return 0.0
        return (sec - self._view_start) / self._view_dur * self.width()

    def _x_to_sec(self, x: float) -> float:
        return self._view_start + (x / max(1, self.width())) * self._view_dur

    def _clamp_view(self) -> None:
        if not self._wave:
            return
        dur = self._wave.duration
        self._view_dur = max(self.MIN_VIEW_SEC, min(self._view_dur, dur))
        max_start = max(0.0, dur - self._view_dur)
        self._view_start = max(0.0, min(self._view_start, max_start))

    def _center_on(self, sec: float) -> None:
        self._view_start = sec - self._view_dur / 2
        self._clamp_view()

    # ---------- 绘制 ----------

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(28, 30, 38))

        if not self._wave:
            p.setPen(QColor(180, 180, 180))
            p.drawText(QRectF(0, 0, w, h), Qt.AlignmentFlag.AlignCenter, "加载波形中...")
            return

        self._draw_grid(p, w, h)
        self._draw_wave(p, w, h)
        self._draw_markers(p, w, h)
        self._draw_playhead(p, w, h)

    def _draw_grid(self, p: QPainter, w: int, h: int) -> None:
        # 中线
        p.setPen(QPen(QColor(70, 74, 88), 1))
        p.drawLine(0, h // 2, w, h // 2)
        # 时间刻度：自适应步长
        step = self._nice_time_step(self._view_dur / max(1, w // 80))
        t = (int(self._view_start / step) + 1) * step
        p.setPen(QPen(QColor(90, 94, 110), 1))
        font = p.font()
        font.setPointSize(8)
        p.setFont(font)
        while t < self._view_start + self._view_dur:
            x = int(self._sec_to_x(t))
            if 0 <= x < w:
                p.drawLine(x, h - 18, x, h)
                p.drawText(x + 2, h - 4, self._fmt(t))
            t += step

    def _draw_wave(self, p: QPainter, w: int, h: int) -> None:
        assert self._wave is not None
        mid = h / 2
        amp = (h / 2) * 0.9
        start = self._view_start
        end = start + self._view_dur
        mins, maxs, bucket = self._wave.peaks(start, end, w)
        if mins.size == 0:
            return
        p.setPen(QPen(QColor(140, 170, 255), 1))
        px_per_bucket = w / mins.size
        for i in range(mins.size):
            x = int(i * px_per_bucket)
            y1 = mid - maxs[i] * amp
            y2 = mid - mins[i] * amp
            p.drawLine(x, int(y1), x, int(max(y2, y1 + 1)))

    def _draw_markers(self, p: QPainter, w: int, h: int) -> None:
        for i, m in enumerate(self._markers):
            x = self._sec_to_x(m)
            if x < -self.HANDLE_W or x > w + self.HANDLE_W:
                continue
            color = QColor(255, 100, 100) if i == self._hover_marker else QColor(230, 80, 80)
            p.setPen(QPen(color, 2))
            p.drawLine(int(x), 0, int(x), h)
            # 顶部三角手柄
            p.setBrush(color)
            p.setPen(Qt.PenStyle.NoPen)
            hw, hh = self.HANDLE_W, self.HANDLE_H
            points = [
                QPoint(int(x - hw / 2), 0),
                QPoint(int(x + hw / 2), 0),
                QPoint(int(x), hh),
            ]
            p.drawPolygon(points)

    def _draw_playhead(self, p: QPainter, w: int, h: int) -> None:
        x = self._sec_to_x(self._playhead)
        if 0 <= x <= w:
            p.setPen(QPen(QColor(255, 220, 100), 2))
            p.drawLine(int(x), 0, int(x), h)

    @staticmethod
    def _fmt(sec: float) -> str:
        m = int(sec // 60)
        s = sec - m * 60
        if sec < 60:
            return f"{s:.1f}s"
        return f"{m}:{s:04.1f}"

    @staticmethod
    def _nice_time_step(raw: float) -> float:
        for c in (0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600):
            if raw <= c:
                return c
        return 900

    # ---------- 事件 ----------

    def wheelEvent(self, e: QWheelEvent) -> None:
        if not self._wave:
            return
        factor = 1.25 if e.angleDelta().y() < 0 else 0.8
        mouse_sec = self._x_to_sec(e.position().x())
        new_dur = self._view_dur * factor
        new_dur = max(self.MIN_VIEW_SEC, min(new_dur, self._wave.duration))
        # 以鼠标位置为锚点缩放
        ratio = (mouse_sec - self._view_start) / self._view_dur
        self._view_start = mouse_sec - new_dur * ratio
        self._view_dur = new_dur
        self._clamp_view()
        self.view_changed.emit()
        self.update()

    def mousePressEvent(self, e: QMouseEvent) -> None:
        if not self._wave:
            return
        self._drag_moved = False
        if e.button() == Qt.MouseButton.LeftButton:
            idx = self._hit_marker(e.position().x(), e.position().y())
            if idx >= 0:
                self._drag_mode = 'marker'
                self._drag_marker_idx = idx
            else:
                self._drag_mode = 'pan'
                self._drag_start_x = int(e.position().x())
                self._drag_start_view = self._view_start
        elif e.button() == Qt.MouseButton.RightButton:
            idx = self._hit_marker(e.position().x(), e.position().y())
            if idx >= 0:
                self.marker_removed.emit(idx)

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        if not self._wave:
            return
        x, y = e.position().x(), e.position().y()

        if self._drag_mode == 'pan':
            dx = x - self._drag_start_x
            if abs(dx) > 3:
                self._drag_moved = True
            sec_per_px = self._view_dur / max(1, self.width())
            self._view_start = self._drag_start_view - dx * sec_per_px
            self._clamp_view()
            self.update()
        elif self._drag_mode == 'marker':
            self._drag_moved = True
            new_sec = self._x_to_sec(x)
            new_sec = max(0.0, min(self._wave.duration, new_sec))
            self._markers[self._drag_marker_idx] = new_sec
            self.update()
        else:
            # 悬停检测
            idx = self._hit_marker(x, y)
            if idx != self._hover_marker:
                self._hover_marker = idx
                self.setCursor(Qt.CursorShape.SizeHorCursor if idx >= 0 else Qt.CursorShape.ArrowCursor)
                self.update()

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:
        if not self._wave:
            return
        if e.button() == Qt.MouseButton.LeftButton:
            if self._drag_mode == 'marker':
                idx = self._drag_marker_idx
                self._markers.sort()
                self.marker_moved.emit(idx, self._markers[idx] if idx < len(self._markers) else 0.0)
                self.update()
            elif self._drag_mode == 'pan' and not self._drag_moved:
                # 单击 → 跳转
                self.seek_to.emit(self._x_to_sec(e.position().x()))
            self._drag_mode = None
            self._drag_marker_idx = -1

    def mouseDoubleClickEvent(self, e: QMouseEvent) -> None:
        if not self._wave:
            return
        if e.button() == Qt.MouseButton.LeftButton:
            idx = self._hit_marker(e.position().x(), e.position().y())
            if idx < 0:
                self.marker_added.emit(self._x_to_sec(e.position().x()))

    def _hit_marker(self, x: float, y: float) -> int:
        """命中检测：顶部手柄区域 或 竖线附近 3px。"""
        for i, m in enumerate(self._markers):
            mx = self._sec_to_x(m)
            # 手柄命中
            if y <= self.HANDLE_H and abs(x - mx) <= self.HANDLE_W / 2:
                return i
            # 竖线命中
            if abs(x - mx) <= 3:
                return i
        return -1
