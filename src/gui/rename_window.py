"""独立批量重命名窗口：AI 起名 + 原地重命名"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.core.rename_pipeline import NamingRecord, run_batch
from src.core.renamer import (
    RenameResult,
    default_log_path,
    execute_renames,
    plan_renames,
    rollback_from_log,
)
from src.core.settings import Settings, load_settings
from src.core.title_registry import (
    TitleRegistry,
    load_registry,
    register_titles,
    registry_path,
)
from src.gui.settings_dialog import SettingsDialog

_COL_STATUS = 0
_COL_OLD = 1
_COL_NEW = 2
_COL_LANG = 3
_COL_REASON = 4
_COL_SRC = 5

_LANG_LABEL = {"zh": "中文", "en": "英文", "unknown": "无人声/未知"}


class RenameWorker(QThread):
    """后台执行：起名 → 规划 → 写标签 + 重命名。"""
    progress = pyqtSignal(int, int, object)       # current, total, NamingRecord
    renamed = pyqtSignal(object, object, object)  # results, log_path, registry
    failed = pyqtSignal(str)

    def __init__(
        self,
        settings: Settings,
        files: list[Path],
        registry: TitleRegistry,
    ) -> None:
        super().__init__()
        self.settings = settings
        self.files = files
        self.registry = registry
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            records = run_batch(
                self.settings,
                self.files,
                self.registry,
                progress=lambda c, t, r: self.progress.emit(c, t, r),
                should_cancel=lambda: self._cancelled,
            )
            if self._cancelled:
                return
            ok_records = [r for r in records if r.ok]
            if not ok_records:
                self.failed.emit("没有任何曲目成功起名，未执行重命名")
                return
            plans = plan_renames([(r.path, r.title) for r in ok_records])
            log_path = default_log_path(self.files[0].parent)
            results = execute_renames(
                plans, log_path=log_path, write_metadata=self.settings.write_metadata
            )
            new_registry = register_titles(
                self.registry, tuple(r.title for r in ok_records)
            )
            self.renamed.emit(results, log_path, new_registry)
        except Exception as e:  # noqa: BLE001 - 后台线程需兜底并回报
            self.failed.emit(str(e))


class RenameWindow(QDialog):
    """批量重命名主窗口。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("批量重命名 - AI 起名")
        self.resize(1120, 660)
        self._settings = load_settings()
        self._registry = load_registry()
        self._worker: RenameWorker | None = None
        self._last_log: Path | None = None
        self._build_ui()

    # ---------- UI ----------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        dir_row = QHBoxLayout()
        self.dir_edit = QLineEdit()
        self.dir_edit.setPlaceholderText("选择包含 MP3 的目录...")
        self.btn_browse = QPushButton("浏览...")
        self.btn_browse.clicked.connect(self._choose_dir)
        dir_row.addWidget(QLabel("目录:"))
        dir_row.addWidget(self.dir_edit, 1)
        dir_row.addWidget(self.btn_browse)
        layout.addLayout(dir_row)

        opt_row = QHBoxLayout()
        self.lbl_found = QLabel("未扫描")
        self.btn_scan = QPushButton("扫描")
        self.btn_scan.clicked.connect(self._scan)
        self.btn_settings = QPushButton("API 设置")
        self.btn_settings.clicked.connect(self._open_settings)
        opt_row.addWidget(self.btn_scan)
        opt_row.addWidget(self.lbl_found)
        opt_row.addStretch(1)
        self.lbl_registry = QLabel(self._registry_hint())
        self.lbl_registry.setStyleSheet("color: gray; font-size: 11px;")
        opt_row.addWidget(self.lbl_registry)
        opt_row.addWidget(self.btn_settings)
        layout.addLayout(opt_row)

        warn = QLabel("注意：确认后将对原文件【原地重命名】并把歌名写入 ID3 标题（不可撤销），"
                      "但会生成 CSV 回滚日志，可用下方按钮还原文件名。")
        warn.setStyleSheet("color: #a33; font-size: 12px;")
        warn.setWordWrap(True)
        layout.addWidget(warn)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["状态", "原文件名", "新歌名", "语种", "命名依据", "歌词来源"]
        )
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(_COL_STATUS, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(_COL_OLD, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(_COL_NEW, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(_COL_LANG, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(_COL_REASON, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(_COL_SRC, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table, 1)

        self.progress = QProgressBar()
        layout.addWidget(self.progress)

        bottom = QHBoxLayout()
        self.btn_start = QPushButton("开始（起名并重命名）")
        self.btn_start.clicked.connect(self._on_start)
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._on_cancel)
        self.btn_rollback = QPushButton("回滚上次重命名")
        self.btn_rollback.setEnabled(False)
        self.btn_rollback.clicked.connect(self._on_rollback)
        bottom.addWidget(self.btn_start)
        bottom.addWidget(self.btn_cancel)
        bottom.addWidget(self.btn_rollback)
        bottom.addStretch(1)
        self.lbl_status = QLabel("就绪")
        bottom.addWidget(self.lbl_status)
        layout.addLayout(bottom)

    # ---------- 交互 ----------

    def _choose_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择目录")
        if path:
            self.dir_edit.setText(path)
            self._scan()

    def _scan(self) -> int:
        files = self._collect_files()
        self.lbl_found.setText(f"找到 {len(files)} 个 MP3")
        return len(files)

    def _collect_files(self) -> list[Path]:
        text = self.dir_edit.text().strip()
        if not text:
            return []
        folder = Path(text)
        if not folder.is_dir():
            return []
        return sorted(f for f in folder.glob("*.mp3") if f.is_file())

    def _registry_hint(self) -> str:
        return f"歌名库：已有 {len(self._registry.titles)} 个（全局去重）"

    def _open_settings(self) -> None:
        dialog = SettingsDialog(self._settings, self)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.result_settings:
            self._settings = dialog.result_settings
            self.lbl_status.setText("API 设置已保存")

    def _on_start(self) -> None:
        files = self._collect_files()
        if not files:
            QMessageBox.warning(self, "提示", "目录下未找到 MP3 文件")
            return
        if not self._settings.is_configured:
            QMessageBox.warning(self, "提示", "请先完成 API 设置")
            self._open_settings()
            if not self._settings.is_configured:
                return

        asr_state = "启用" if (self._settings.asr_enabled and self._settings.asr_configured) \
            else "未启用（无法判定语种）"
        meta_state = "写入 ID3 标题" if self._settings.write_metadata else "不写标签"
        reply = QMessageBox.question(
            self,
            "确认执行",
            f"将对 {len(files)} 个文件调用 AI 起名并【原地重命名】。\n"
            f"目录：{files[0].parent}\n"
            f"语音识别：{asr_state}\n"
            f"元数据：{meta_state}\n"
            f"歌名库：已有 {len(self._registry.titles)} 个，本次将全局去重\n\n"
            "确认执行？",
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.table.setRowCount(0)
        self.progress.setMaximum(len(files))
        self.progress.setValue(0)
        self._set_running(True)

        self._worker = RenameWorker(self._settings, files, self._registry)
        self._worker.progress.connect(self._on_progress)
        self._worker.renamed.connect(self._on_renamed)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _set_running(self, running: bool) -> None:
        self.btn_start.setEnabled(not running)
        self.btn_scan.setEnabled(not running)
        self.btn_browse.setEnabled(not running)
        self.btn_settings.setEnabled(not running)
        self.btn_cancel.setEnabled(running)

    def _on_progress(self, current: int, total: int, record: object) -> None:
        assert isinstance(record, NamingRecord)
        self.progress.setValue(current)
        self._append_row(record)
        self.lbl_status.setText(f"起名中... {current}/{total}")

    def _append_row(self, record: NamingRecord) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        lang = _LANG_LABEL.get(record.detected_language, record.detected_language)
        if record.ok:
            status, color = "成功", QColor("green")
            new_name = f"{record.title}{record.path.suffix}"
            reason, note = record.reason, record.lyrics_source
        else:
            status, color = "失败", QColor("red")
            new_name, reason, note = "-", record.error or "", record.lyrics_source

        for col, text in (
            (_COL_STATUS, status),
            (_COL_OLD, record.path.name),
            (_COL_NEW, new_name),
            (_COL_LANG, lang),
            (_COL_REASON, reason),
            (_COL_SRC, note),
        ):
            item = QTableWidgetItem(text)
            if col == _COL_STATUS:
                item.setForeground(color)
            self.table.setItem(row, col, item)
        self.table.scrollToBottom()

    def _on_renamed(self, results: object, log_path: object, registry: object) -> None:
        self._set_running(False)
        result_list = list(results)  # type: ignore[arg-type]
        self._last_log = Path(str(log_path))
        self._registry = registry  # type: ignore[assignment]
        self.lbl_registry.setText(self._registry_hint())
        self.btn_rollback.setEnabled(True)

        ok = sum(1 for r in result_list if r.success)
        fail = len(result_list) - ok
        tagged = sum(1 for r in result_list if r.tagged)
        tag_fail = sum(1 for r in result_list if r.tag_error)
        self.lbl_status.setText(f"完成：重命名 {ok} 成功 / {fail} 失败；标签 {tagged} 写入 / {tag_fail} 失败")
        QMessageBox.information(
            self,
            "完成",
            f"重命名：成功 {ok} 个，失败 {fail} 个\n"
            f"ID3 标题：写入 {tagged} 个，失败 {tag_fail} 个\n\n"
            f"歌名库已更新至 {len(self._registry.titles)} 个\n\n"
            f"回滚日志：\n{self._last_log}",
        )

    def _on_failed(self, msg: str) -> None:
        self._set_running(False)
        self.lbl_status.setText("执行失败")
        QMessageBox.critical(self, "执行失败", msg)

    def _on_cancel(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self.lbl_status.setText("正在取消...")

    def _on_rollback(self) -> None:
        if self._last_log is None or not self._last_log.exists():
            QMessageBox.warning(self, "提示", "没有可用的回滚日志")
            return
        reply = QMessageBox.question(
            self, "确认回滚", f"依据日志还原文件名？\n{self._last_log}"
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            ok, fail = rollback_from_log(self._last_log)
        except (OSError, FileNotFoundError) as e:
            QMessageBox.critical(self, "回滚失败", str(e))
            return
        QMessageBox.information(self, "回滚完成", f"还原成功 {ok} 个，失败 {fail} 个")

    def closeEvent(self, event) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(5000)
        super().closeEvent(event)
