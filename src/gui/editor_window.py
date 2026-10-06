"""单曲剪辑工作台"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.core.analyzer import FileAnalysis, TrackFlag, TrackSegment, build_segments
from src.core.audio_player import AudioPlayer
from src.core.waveform_data import WaveformData, WaveformError
from src.gui.waveform_widget import WaveformWidget


class _WaveLoadWorker(QThread):
    loaded = pyqtSignal(object)  # WaveformData
    failed = pyqtSignal(str)

    def __init__(self, path: Path, duration: float) -> None:
        super().__init__()
        self.path = path
        self.duration = duration

    def run(self) -> None:
        try:
            self.loaded.emit(WaveformData.from_file(self.path, self.duration))
        except WaveformError as e:
            self.failed.emit(str(e))
        except Exception as e:
            self.failed.emit(f"未知错误: {e}")


class EditorWindow(QDialog):
    """单曲剪辑工作台。

    打开方式：EditorWindow(analysis, parent).exec()
    结果：self.result_segments — 若用户确认则为新 segments，否则为 None。
    """

    def __init__(
        self,
        analysis: FileAnalysis,
        min_track_sec: float = 60.0,
        long_track_sec: float = 600.0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"剪辑工作台 - {analysis.src_path.name}")
        self.resize(1200, 720)

        self._analysis = analysis
        self._min_track = min_track_sec
        self._long_track = long_track_sec
        self.result_segments: tuple[TrackSegment, ...] | None = None

        # 初始分割点 = segments 内部边界
        initial_markers = [s.end for s in analysis.segments[:-1]] if analysis.segments else []

        self._player = AudioPlayer(self)
        self._player.load(analysis.src_path)
        self._player.position_changed.connect(self._on_position)
        self._player.state_changed.connect(self._on_play_state)
        self._player.error_occurred.connect(self._on_player_error)

        self._build_ui(initial_markers)

        # 异步加载波形
        self._wave_worker = _WaveLoadWorker(analysis.src_path, analysis.total_duration)
        self._wave_worker.loaded.connect(self._on_wave_loaded)
        self._wave_worker.failed.connect(self._on_wave_failed)
        self._wave_worker.start()

        # 播放头刷新由 position_changed 驱动

    # ---------- UI ----------

    def _build_ui(self, markers: list[float]) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        header = QLabel(f"{self._analysis.src_path.name}   总时长 {self._fmt(self._analysis.total_duration)}")
        header.setStyleSheet("font-weight: bold;")
        layout.addWidget(header)

        # 波形 + 右侧分割点列表
        splitter = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.wave = WaveformWidget()
        self.wave.set_markers(markers)
        self.wave.seek_to.connect(self._on_seek)
        self.wave.marker_moved.connect(self._on_marker_changed)
        self.wave.marker_added.connect(self._on_marker_added)
        self.wave.marker_removed.connect(self._on_marker_removed)
        left_layout.addWidget(self.wave, 1)

        # 播放控制栏
        ctrl = QHBoxLayout()
        self.btn_play = QPushButton("▶ 播放")
        self.btn_play.clicked.connect(self._toggle_play)
        self.btn_stop = QPushButton("■ 停止")
        self.btn_stop.clicked.connect(self._stop)
        self.lbl_time = QLabel("0:00 / 0:00")
        ctrl.addWidget(self.btn_play)
        ctrl.addWidget(self.btn_stop)
        ctrl.addWidget(self.lbl_time)
        ctrl.addStretch(1)

        self.btn_zoom_out = QPushButton("−")
        self.btn_zoom_in = QPushButton("+")
        self.btn_zoom_all = QPushButton("适配全曲")
        self.btn_zoom_out.clicked.connect(lambda: self._zoom(1.5))
        self.btn_zoom_in.clicked.connect(lambda: self._zoom(1 / 1.5))
        self.btn_zoom_all.clicked.connect(self._zoom_all)
        ctrl.addWidget(QLabel("缩放:"))
        ctrl.addWidget(self.btn_zoom_out)
        ctrl.addWidget(self.btn_zoom_in)
        ctrl.addWidget(self.btn_zoom_all)
        left_layout.addLayout(ctrl)

        hint = QLabel("操作: 单击跳转播放 | 拖拽平移 | 滚轮缩放 | 双击添加分割点 | 右键分割点删除 | 拖拽红色手柄调整")
        hint.setStyleSheet("color: gray; font-size: 11px;")
        left_layout.addWidget(hint)

        splitter.addWidget(left)

        # 右侧分割点列表
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(QLabel("分割点"))
        self.marker_list = QListWidget()
        right_layout.addWidget(self.marker_list, 1)
        btn_row = QHBoxLayout()
        self.btn_del_marker = QPushButton("删除选中")
        self.btn_del_marker.clicked.connect(self._delete_selected_marker)
        self.btn_clear = QPushButton("清空")
        self.btn_clear.clicked.connect(self._clear_markers)
        btn_row.addWidget(self.btn_del_marker)
        btn_row.addWidget(self.btn_clear)
        right_layout.addLayout(btn_row)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter, 1)

        # 底部确认/取消
        bottom = QHBoxLayout()
        self.lbl_segments = QLabel("")
        bottom.addWidget(self.lbl_segments)
        bottom.addStretch(1)
        self.btn_ok = QPushButton("确认应用")
        self.btn_cancel = QPushButton("取消")
        self.btn_ok.clicked.connect(self._accept)
        self.btn_cancel.clicked.connect(self.reject)
        bottom.addWidget(self.btn_ok)
        bottom.addWidget(self.btn_cancel)
        layout.addLayout(bottom)

        self._refresh_marker_list()

    # ---------- 波形加载 ----------

    def _on_wave_loaded(self, wave: WaveformData) -> None:
        self.wave.set_waveform(wave)
        self.lbl_time.setText(f"0:00 / {self._fmt(wave.duration)}")

    def _on_wave_failed(self, msg: str) -> None:
        QMessageBox.critical(self, "波形加载失败", msg)

    # ---------- 播放 ----------

    def _toggle_play(self) -> None:
        if self._player.is_playing:
            self._player.pause()
        else:
            self._player.play()

    def _stop(self) -> None:
        self._player.stop()
        self.wave.set_playhead(0.0)

    def _on_seek(self, sec: float) -> None:
        self._player.seek_ms(int(sec * 1000))
        self.wave.set_playhead(sec)
        if not self._player.is_playing:
            self._player.play()

    def _on_position(self, ms: int) -> None:
        sec = ms / 1000.0
        self.wave.set_playhead(sec, follow=True)
        dur_s = self._analysis.total_duration
        self.lbl_time.setText(f"{self._fmt(sec)} / {self._fmt(dur_s)}")

    def _on_play_state(self, playing: bool) -> None:
        self.btn_play.setText("⏸ 暂停" if playing else "▶ 播放")

    def _on_player_error(self, msg: str) -> None:
        QMessageBox.warning(self, "播放错误", msg)

    # ---------- 缩放 ----------

    def _zoom(self, factor: float) -> None:
        if self.wave._wave is None:
            return
        playhead = self.wave._playhead
        new_dur = self.wave._view_dur * factor
        new_dur = max(WaveformWidget.MIN_VIEW_SEC, min(new_dur, self.wave._wave.duration))
        self.wave._view_dur = new_dur
        self.wave._center_on(playhead)
        self.wave.update()

    def _zoom_all(self) -> None:
        if self.wave._wave:
            self.wave._view_start = 0.0
            self.wave._view_dur = self.wave._wave.duration
            self.wave.update()

    # ---------- 分割点 ----------

    def _on_marker_changed(self, idx: int, sec: float) -> None:
        self._refresh_marker_list()

    def _on_marker_added(self, sec: float) -> None:
        markers = self.wave.markers()
        markers.append(sec)
        self.wave.set_markers(markers)
        self._refresh_marker_list()

    def _on_marker_removed(self, idx: int) -> None:
        markers = self.wave.markers()
        if 0 <= idx < len(markers):
            markers.pop(idx)
            self.wave.set_markers(markers)
            self._refresh_marker_list()

    def _delete_selected_marker(self) -> None:
        row = self.marker_list.currentRow()
        if row >= 0:
            self._on_marker_removed(row)

    def _clear_markers(self) -> None:
        if QMessageBox.question(self, "确认", "清空所有分割点？") == QMessageBox.StandardButton.Yes:
            self.wave.set_markers([])
            self._refresh_marker_list()

    def _refresh_marker_list(self) -> None:
        markers = self.wave.markers()
        self.marker_list.clear()
        for i, m in enumerate(markers):
            item = QListWidgetItem(f"#{i + 1:02d}   {self._fmt(m)}")
            self.marker_list.addItem(item)
        # 预览曲目数
        n = len(markers) + 1
        self.lbl_segments.setText(f"将切出 {n} 首")

    # ---------- 确认 ----------

    def _accept(self) -> None:
        markers = self.wave.markers()
        duration = self._analysis.total_duration
        if duration <= 0:
            QMessageBox.warning(self, "错误", "音频时长无效")
            return

        # 用分割点作为"伪静音中点"构造 segments
        from src.core.ffmpeg_ops import SilenceInterval
        fake_silences = [SilenceInterval(start=m - 0.001, end=m + 0.001) for m in markers]
        segments = build_segments(
            duration,
            fake_silences,
            min_track_sec=self._min_track,
            long_track_sec=self._long_track,
        )
        self.result_segments = segments
        self._player.stop()
        self.accept()

    def reject(self) -> None:
        self._player.stop()
        super().reject()

    def closeEvent(self, event) -> None:
        self._player.stop()
        if self._wave_worker.isRunning():
            self._wave_worker.terminate()
            self._wave_worker.wait(2000)
        super().closeEvent(event)

    @staticmethod
    def _fmt(sec: float) -> str:
        m = int(sec // 60)
        s = int(sec % 60)
        return f"{m}:{s:02d}"
