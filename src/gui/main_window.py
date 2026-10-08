"""MusiCut 主窗口"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QContextMenuEvent
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src import __version__
from src.core.analyzer import (
    FileAnalysis,
    TrackFlag,
    build_segments,
    short_title_for_file,
)
from src.core.ffmpeg_ops import detect_silences
from src.core.splitter import SplitResult, SplitTask, execute_task, plan_output_paths
from src.gui.editor_window import EditorWindow
from src.gui.rename_window import RenameWindow


_COL_FLAG = 0
_COL_DURATION = 1
_COL_TITLE = 2
_COL_STATUS = 3


class AnalysisWorker(QThread):
    progress = pyqtSignal(int, int)  # current, total
    finished_file = pyqtSignal(FileAnalysis)
    done = pyqtSignal()

    def __init__(
        self,
        files: list[Path],
        noise_db: float,
        min_silence: float,
        min_track: float,
        long_track: float,
    ) -> None:
        super().__init__()
        self.files = files
        self.noise_db = noise_db
        self.min_silence = min_silence
        self.min_track = min_track
        self.long_track = long_track
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        total = len(self.files)
        for i, f in enumerate(self.files):
            if self._cancelled:
                break
            self.progress.emit(i, total)
            try:
                duration, silences = detect_silences(
                    f,
                    noise_db=self.noise_db,
                    min_duration=self.min_silence,
                )
                segments = build_segments(
                    duration,
                    silences,
                    min_track_sec=self.min_track,
                    long_track_sec=self.long_track,
                )
                self.finished_file.emit(
                    FileAnalysis(
                        src_path=f,
                        total_duration=duration,
                        segments=segments,
                    )
                )
            except Exception as exc:
                self.finished_file.emit(
                    FileAnalysis(
                        src_path=f,
                        total_duration=0.0,
                        error=str(exc),
                    )
                )
        self.progress.emit(total, total)
        self.done.emit()


class SplitWorker(QThread):
    progress = pyqtSignal(int, int)
    finished_task = pyqtSignal(SplitResult)
    done = pyqtSignal()

    def __init__(self, tasks: list[SplitTask]) -> None:
        super().__init__()
        self.tasks = tasks
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        total = len(self.tasks)
        for i, task in enumerate(self.tasks):
            if self._cancelled:
                break
            self.progress.emit(i, total)
            result = execute_task(task)
            self.finished_task.emit(result)
        self.progress.emit(total, total)
        self.done.emit()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"MusiCut v{__version__} - 歌单自动切分")
        self.resize(1100, 750)

        self._analyses: dict[Path, FileAnalysis] = {}
        self._analysis_worker: AnalysisWorker | None = None
        self._split_worker: SplitWorker | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(12)

        # --- 目录选择 ---
        dir_row = QHBoxLayout()
        self.src_edit = QLineEdit()
        self.src_edit.setPlaceholderText("选择包含 MP3 歌单的目录...")
        self.btn_src = QPushButton("浏览...")
        self.btn_src.clicked.connect(self._choose_src)
        dir_row.addWidget(QLabel("源目录:"))
        dir_row.addWidget(self.src_edit, 1)
        dir_row.addWidget(self.btn_src)
        layout.addLayout(dir_row)

        out_row = QHBoxLayout()
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("切分后的输出目录...")
        self.btn_out = QPushButton("浏览...")
        self.btn_out.clicked.connect(self._choose_out)
        out_row.addWidget(QLabel("输出目录:"))
        out_row.addWidget(self.out_edit, 1)
        out_row.addWidget(self.btn_out)
        layout.addLayout(out_row)

        # --- 参数 ---
        param_row = QHBoxLayout()
        param_row.setSpacing(16)

        self.spin_noise = QDoubleSpinBox()
        self.spin_noise.setRange(-60.0, -10.0)
        self.spin_noise.setValue(-28.0)
        self.spin_noise.setDecimals(1)
        self.spin_noise.setSuffix(" dB")
        param_row.addWidget(QLabel("静音阈值:"))
        param_row.addWidget(self.spin_noise)

        self.spin_silence = QDoubleSpinBox()
        self.spin_silence.setRange(0.3, 5.0)
        self.spin_silence.setValue(1.2)
        self.spin_silence.setDecimals(1)
        self.spin_silence.setSuffix(" s")
        param_row.addWidget(QLabel("最短静音:"))
        param_row.addWidget(self.spin_silence)

        self.spin_min = QSpinBox()
        self.spin_min.setRange(10, 300)
        self.spin_min.setValue(60)
        self.spin_min.setSuffix(" s")
        param_row.addWidget(QLabel("最短曲目:"))
        param_row.addWidget(self.spin_min)

        self.spin_long = QSpinBox()
        self.spin_long.setRange(120, 3600)
        self.spin_long.setValue(600)
        self.spin_long.setSuffix(" s")
        param_row.addWidget(QLabel("过长阈值:"))
        param_row.addWidget(self.spin_long)

        param_row.addStretch(1)
        layout.addLayout(param_row)

        # --- 主区域：文件列表 + 预览 ---
        splitter = QSplitter(Qt.Orientation.Vertical)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["#", "时长", "源文件", "状态"])
        self.tree.header().setSectionResizeMode(_COL_FLAG, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.header().setSectionResizeMode(_COL_DURATION, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.header().setSectionResizeMode(_COL_TITLE, QHeaderView.ResizeMode.Stretch)
        self.tree.header().setSectionResizeMode(_COL_STATUS, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.currentItemChanged.connect(self._on_selection_changed)
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        splitter.addWidget(self.tree)

        self.preview = QTreeWidget()
        self.preview.setHeaderLabels(["曲目", "开始", "结束", "时长", "备注"])
        self.preview.header().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.preview.setMaximumHeight(220)
        splitter.addWidget(self.preview)

        layout.addWidget(splitter, 1)

        # --- 底部按钮 + 进度 ---
        bottom = QHBoxLayout()
        self.btn_analyze = QPushButton("开始分析")
        self.btn_analyze.clicked.connect(self._on_analyze)
        self.btn_split = QPushButton("执行切分")
        self.btn_split.setEnabled(False)
        self.btn_split.clicked.connect(self._on_split)
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._on_cancel)
        bottom.addWidget(self.btn_analyze)
        bottom.addWidget(self.btn_split)
        bottom.addWidget(self.btn_cancel)
        self.btn_rename = QPushButton("批量重命名（AI 起名）")
        self.btn_rename.clicked.connect(self._open_rename_window)
        bottom.addWidget(self.btn_rename)
        bottom.addStretch(1)
        self.lbl_status = QLabel("就绪")
        bottom.addWidget(self.lbl_status)
        layout.addLayout(bottom)

        self.progress = QProgressBar()
        layout.addWidget(self.progress)

    def _choose_src(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择源目录")
        if path:
            self.src_edit.setText(path)
            if not self.out_edit.text():
                self.out_edit.setText(str(Path(path) / "output"))

    def _choose_out(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if path:
            self.out_edit.setText(path)

    def _load_files(self) -> list[Path]:
        src = self.src_edit.text().strip()
        if not src:
            QMessageBox.warning(self, "提示", "请先选择源目录")
            return []
        files = sorted(Path(src).glob("*.mp3"))
        if not files:
            QMessageBox.information(self, "提示", "源目录下未找到 MP3 文件")
            return []
        return files

    def _on_analyze(self) -> None:
        files = self._load_files()
        if not files:
            return
        self._analyses.clear()
        self.tree.clear()
        self.preview.clear()
        self.btn_analyze.setEnabled(False)
        self.btn_split.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress.setMaximum(len(files))
        self.progress.setValue(0)

        self._analysis_worker = AnalysisWorker(
            files=files,
            noise_db=self.spin_noise.value(),
            min_silence=self.spin_silence.value(),
            min_track=self.spin_min.value(),
            long_track=self.spin_long.value(),
        )
        self._analysis_worker.progress.connect(self._on_analyze_progress)
        self._analysis_worker.finished_file.connect(self._on_analyze_file)
        self._analysis_worker.done.connect(self._on_analyze_done)
        self._analysis_worker.start()

    def _on_analyze_progress(self, current: int, total: int) -> None:
        self.progress.setValue(current)
        self.lbl_status.setText(f"分析中... {current}/{total}")

    def _on_analyze_file(self, analysis: FileAnalysis) -> None:
        self._analyses[analysis.src_path] = analysis
        item = QTreeWidgetItem(self.tree)
        total_m = int(analysis.total_duration // 60)
        total_s = int(analysis.total_duration % 60)
        item.setText(_COL_FLAG, "")
        item.setText(_COL_DURATION, f"{total_m}:{total_s:02d}")
        item.setText(_COL_TITLE, analysis.src_path.name)
        if analysis.error:
            item.setText(_COL_STATUS, f"错误: {analysis.error}")
            item.setForeground(_COL_STATUS, QColor("red"))
        else:
            cnt = len(analysis.segments)
            flags = [s.flag for s in analysis.segments if s.flag != TrackFlag.OK]
            if TrackFlag.TOO_LONG in flags:
                status = f"{cnt} 首 (⚠️ 有长曲目)"
                item.setForeground(_COL_STATUS, QColor("red"))
            elif TrackFlag.TOO_SHORT in flags:
                status = f"{cnt} 首 (⚠️ 有短曲目)"
                item.setForeground(_COL_STATUS, QColor("orange"))
            else:
                status = f"{cnt} 首"
            item.setText(_COL_STATUS, status)
        item.setData(0, Qt.ItemDataRole.UserRole, str(analysis.src_path))

    def _on_analyze_done(self) -> None:
        self.btn_analyze.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        has_ok = any(a.error is None for a in self._analyses.values())
        self.btn_split.setEnabled(has_ok)
        self.lbl_status.setText("分析完成，请检查异常标记后执行切分")

    def _on_selection_changed(self, current: QTreeWidgetItem | None) -> None:
        self.preview.clear()
        if current is None:
            return
        path_str = current.data(0, Qt.ItemDataRole.UserRole)
        if not path_str:
            return
        analysis = self._analyses.get(Path(path_str))
        if analysis is None or analysis.error:
            return
        for seg in analysis.segments:
            pi = QTreeWidgetItem(self.preview)
            pi.setText(0, f"{seg.index:02d}")
            pi.setText(1, self._fmt_time(seg.start))
            pi.setText(2, self._fmt_time(seg.end))
            dur_m = int(seg.duration // 60)
            dur_s = int(seg.duration % 60)
            pi.setText(3, f"{dur_m}:{dur_s:02d}")
            if seg.flag == TrackFlag.TOO_SHORT:
                pi.setText(4, "过短")
                for c in range(5):
                    pi.setBackground(c, QColor(255, 200, 100, 120))
            elif seg.flag == TrackFlag.TOO_LONG:
                pi.setText(4, "过长")
                for c in range(5):
                    pi.setBackground(c, QColor(255, 100, 100, 120))
            else:
                pi.setText(4, "正常")
        self.preview.expandAll()

    @staticmethod
    def _fmt_time(sec: float) -> str:
        m = int(sec // 60)
        s = int(sec % 60)
        return f"{m}:{s:02d}"

    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        """双击打开剪辑工作台"""
        path_str = item.data(0, Qt.ItemDataRole.UserRole)
        if not path_str:
            return
        path = Path(path_str)
        analysis = self._analyses.get(path)
        if analysis is None or analysis.error:
            QMessageBox.warning(self, "提示", "该文件分析失败，无法打开编辑器")
            return
        editor = EditorWindow(
            analysis,
            min_track_sec=self.spin_min.value(),
            long_track_sec=self.spin_long.value(),
            parent=self,
        )
        if editor.exec() == EditorWindow.DialogCode.Accepted and editor.result_segments is not None:
            # 更新分析结果
            new_analysis = FileAnalysis(
                src_path=analysis.src_path,
                total_duration=analysis.total_duration,
                segments=editor.result_segments,
            )
            self._analyses[path] = new_analysis
            self._update_tree_item(item, new_analysis)
            self._on_selection_changed(item)
            self.lbl_status.setText(f"已更新: {path.name}")

    def _update_tree_item(self, item: QTreeWidgetItem, analysis: FileAnalysis) -> None:
        """根据分析结果刷新树中条目显示"""
        total_m = int(analysis.total_duration // 60)
        total_s = int(analysis.total_duration % 60)
        item.setText(_COL_DURATION, f"{total_m}:{total_s:02d}")
        if analysis.error:
            item.setText(_COL_STATUS, f"错误: {analysis.error}")
            item.setForeground(_COL_STATUS, QColor("red"))
        else:
            cnt = len(analysis.segments)
            flags = [s.flag for s in analysis.segments if s.flag != TrackFlag.OK]
            if TrackFlag.TOO_LONG in flags:
                status = f"{cnt} 首 (⚠️ 有长曲目)"
                item.setForeground(_COL_STATUS, QColor("red"))
            elif TrackFlag.TOO_SHORT in flags:
                status = f"{cnt} 首 (⚠️ 有短曲目)"
                item.setForeground(_COL_STATUS, QColor("orange"))
            else:
                status = f"{cnt} 首"
                item.setForeground(_COL_STATUS, QColor("black"))
            item.setText(_COL_STATUS, status)

    def _open_rename_window(self) -> None:
        """打开独立批量重命名窗口"""
        window = RenameWindow(self)
        window.exec()

    def _on_split(self) -> None:
        analyses = [a for a in self._analyses.values() if a.error is None]
        if not analyses:
            return
        out_dir = Path(self.out_edit.text().strip())
        if not out_dir:
            QMessageBox.warning(self, "提示", "请设置输出目录")
            return
        out_dir.mkdir(parents=True, exist_ok=True)

        tasks = plan_output_paths(analyses, out_dir)
        if not tasks:
            QMessageBox.information(self, "提示", "没有可切分的曲目")
            return

        reply = QMessageBox.question(
            self,
            "确认",
            f"将切分 {len(tasks)} 首曲目到:\n{out_dir}\n\n确认执行？",
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.btn_analyze.setEnabled(False)
        self.btn_split.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress.setMaximum(len(tasks))
        self.progress.setValue(0)

        self._split_worker = SplitWorker(tasks)
        self._split_worker.progress.connect(self._on_split_progress)
        self._split_worker.finished_task.connect(self._on_split_task)
        self._split_worker.done.connect(self._on_split_done)
        self._split_worker.start()

    def _on_split_progress(self, current: int, total: int) -> None:
        self.progress.setValue(current)
        self.lbl_status.setText(f"切分中... {current}/{total}")

    def _on_split_task(self, result: SplitResult) -> None:
        if not result.success:
            self.lbl_status.setText(f"失败: {result.task.dst_path.name}")

    def _on_split_done(self) -> None:
        self.btn_analyze.setEnabled(True)
        self.btn_split.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.lbl_status.setText("切分完成")
        QMessageBox.information(self, "完成", "切分操作已结束。")

    def _on_cancel(self) -> None:
        if self._analysis_worker and self._analysis_worker.isRunning():
            self._analysis_worker.cancel()
            self._analysis_worker.wait(3000)
        if self._split_worker and self._split_worker.isRunning():
            self._split_worker.cancel()
            self._split_worker.wait(3000)
        self.btn_analyze.setEnabled(True)
        self.btn_split.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.lbl_status.setText("已取消")

    def closeEvent(self, event) -> None:
        self._on_cancel()
        event.accept()
