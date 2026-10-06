"""音频播放器封装"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QObject, QUrl, pyqtSignal
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer


class AudioPlayer(QObject):
    """基于 QMediaPlayer 的播放器。

    position_ms / duration_ms 单位为毫秒。
    """
    position_changed = pyqtSignal(int)   # ms
    state_changed = pyqtSignal(bool)     # True=播放中
    duration_changed = pyqtSignal(int)   # ms
    error_occurred = pyqtSignal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._player = QMediaPlayer(self)
        self._audio = QAudioOutput(self)
        self._player.setAudioOutput(self._audio)
        self._player.positionChanged.connect(self._on_position)
        self._player.durationChanged.connect(self._on_duration)
        self._player.playbackStateChanged.connect(self._on_state)
        self._player.errorOccurred.connect(self._on_error)

    def load(self, path: Path) -> None:
        self._player.setSource(QUrl.fromLocalFile(str(path)))

    def play(self) -> None:
        self._player.play()

    def pause(self) -> None:
        self._player.pause()

    def stop(self) -> None:
        self._player.stop()

    def seek_ms(self, ms: int) -> None:
        self._player.setPosition(max(0, ms))

    @property
    def is_playing(self) -> bool:
        return self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState

    @property
    def position_ms(self) -> int:
        return self._player.position()

    @property
    def duration_ms(self) -> int:
        return self._player.duration()

    def set_volume(self, volume: float) -> None:
        self._audio.setVolume(max(0.0, min(1.0, volume)))

    def _on_position(self, ms: int) -> None:
        self.position_changed.emit(int(ms))

    def _on_duration(self, ms: int) -> None:
        self.duration_changed.emit(int(ms))

    def _on_state(self, state) -> None:
        self.state_changed.emit(state == QMediaPlayer.PlaybackState.PlayingState)

    def _on_error(self, error, msg: str) -> None:
        self.error_occurred.emit(msg or str(error))
