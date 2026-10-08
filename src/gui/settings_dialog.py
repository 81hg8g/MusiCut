"""API 配置对话框"""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.core.ai_namer import AiNamerError, NamingInput, suggest_title
from src.core.settings import (
    DEFAULT_ASR_BASE_URL,
    DEFAULT_ASR_MODEL,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    Settings,
    save_settings,
    settings_path,
)


class SettingsDialog(QDialog):
    """编辑并保存 API 配置。"""

    def __init__(self, settings: Settings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("API 设置")
        self.setMinimumWidth(460)
        self._settings = settings
        self.result_settings: Settings | None = None
        self._build_ui(settings)

    def _build_ui(self, s: Settings) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(self._build_llm_group(s))
        layout.addWidget(self._build_asr_group(s))
        layout.addWidget(self._build_output_group(s))

        hint = QLabel(f"配置保存位置：{settings_path()}")
        hint.setStyleSheet("color: gray; font-size: 11px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        note = QLabel(
            "隐私提示：启用语音识别后，音频片段会上传到 ASR 服务商以判定语种；"
            "音频特征与歌词文本会上传到起名模型。不上传完整音频。"
        )
        note.setStyleSheet("color: #946200; font-size: 11px;")
        note.setWordWrap(True)
        layout.addWidget(note)

        btn_row = QHBoxLayout()
        self.btn_test = QPushButton("测试起名连接")
        self.btn_test.clicked.connect(self._on_test)
        btn_row.addWidget(self.btn_test)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _build_llm_group(self, s: Settings) -> QGroupBox:
        box = QGroupBox("大模型（起名）")
        form = QFormLayout(box)

        self.edit_key = QLineEdit(s.api_key)
        self.edit_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.edit_key.setPlaceholderText("sk-...")
        key_row = QHBoxLayout()
        key_row.addWidget(self.edit_key, 1)
        self.btn_eye = QPushButton("显示")
        self.btn_eye.setCheckable(True)
        self.btn_eye.setFixedWidth(56)
        self.btn_eye.toggled.connect(self._toggle_echo)
        key_row.addWidget(self.btn_eye)
        key_widget = QWidget()
        key_widget.setLayout(key_row)
        form.addRow("API Key:", key_widget)

        self.edit_url = QLineEdit(s.base_url or DEFAULT_BASE_URL)
        form.addRow("Base URL:", self.edit_url)

        self.edit_model = QLineEdit(s.model or DEFAULT_MODEL)
        form.addRow("模型:", self.edit_model)

        self.combo_lang = QComboBox()
        self.combo_lang.addItem("中文", "zh")
        self.combo_lang.addItem("英文", "en")
        self.combo_lang.setCurrentIndex(0 if s.language == "zh" else 1)
        form.addRow("无人声时语言:", self.combo_lang)

        self.spin_conc = QSpinBox()
        self.spin_conc.setRange(1, 16)
        self.spin_conc.setValue(s.concurrency)
        form.addRow("并发数:", self.spin_conc)
        return box

    def _build_asr_group(self, s: Settings) -> QGroupBox:
        box = QGroupBox("语音识别（判定中文/英文演唱）")
        form = QFormLayout(box)

        self.chk_asr = QCheckBox("启用（关闭则无法判定语种）")
        self.chk_asr.setChecked(s.asr_enabled)
        form.addRow("", self.chk_asr)

        self.edit_asr_key = QLineEdit(s.asr_api_key)
        self.edit_asr_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.edit_asr_key.setPlaceholderText("SiliconFlow API Key")
        form.addRow("ASR Key:", self.edit_asr_key)

        self.edit_asr_url = QLineEdit(s.asr_base_url or DEFAULT_ASR_BASE_URL)
        form.addRow("ASR URL:", self.edit_asr_url)

        self.edit_asr_model = QLineEdit(s.asr_model or DEFAULT_ASR_MODEL)
        form.addRow("ASR 模型:", self.edit_asr_model)

        self.spin_sample = QSpinBox()
        self.spin_sample.setRange(20, 300)
        self.spin_sample.setSuffix(" 秒")
        self.spin_sample.setValue(s.asr_sample_sec)
        form.addRow("截取样长:", self.spin_sample)

        self.chk_vocal = QCheckBox("本地人声预筛（Silero VAD；无本地歌词且判定无人声时跳过 ASR，省费用）")
        self.chk_vocal.setChecked(s.vocal_filter_enabled)
        form.addRow("", self.chk_vocal)

        self.spin_vocal = QDoubleSpinBox()
        self.spin_vocal.setRange(0.01, 0.50)
        self.spin_vocal.setSingleStep(0.01)
        self.spin_vocal.setDecimals(2)
        self.spin_vocal.setValue(s.vocal_threshold)
        form.addRow("人声占比阈值:", self.spin_vocal)

        from src.core import vocal_detect
        ok = vocal_detect.is_available()
        status = "可用" if ok else f"不可用（{vocal_detect.load_error() or '未知原因'}）"
        lbl_vad = QLabel(f"本地人声检测（Silero VAD）：{status}")
        lbl_vad.setWordWrap(True)
        lbl_vad.setStyleSheet(
            f"color: {'#1a7f37' if ok else '#a33'}; font-size: 11px;"
        )
        form.addRow("", lbl_vad)
        return box

    def _build_output_group(self, s: Settings) -> QGroupBox:
        box = QGroupBox("输出")
        form = QFormLayout(box)
        self.chk_meta = QCheckBox("将歌名写入 MP3 的 ID3 标题（Title）")
        self.chk_meta.setChecked(s.write_metadata)
        form.addRow("", self.chk_meta)
        return box

    def _toggle_echo(self, shown: bool) -> None:
        self.edit_key.setEchoMode(
            QLineEdit.EchoMode.Normal if shown else QLineEdit.EchoMode.Password
        )
        self.btn_eye.setText("隐藏" if shown else "显示")

    def _collect(self) -> Settings:
        return self._settings.with_updates(
            api_key=self.edit_key.text().strip(),
            base_url=self.edit_url.text().strip() or DEFAULT_BASE_URL,
            model=self.edit_model.text().strip() or DEFAULT_MODEL,
            language=self.combo_lang.currentData(),
            concurrency=self.spin_conc.value(),
            asr_enabled=self.chk_asr.isChecked(),
            asr_api_key=self.edit_asr_key.text().strip(),
            asr_base_url=self.edit_asr_url.text().strip() or DEFAULT_ASR_BASE_URL,
            asr_model=self.edit_asr_model.text().strip() or DEFAULT_ASR_MODEL,
            asr_sample_sec=self.spin_sample.value(),
            vocal_filter_enabled=self.chk_vocal.isChecked(),
            vocal_threshold=self.spin_vocal.value(),
            write_metadata=self.chk_meta.isChecked(),
        )

    def _on_test(self) -> None:
        candidate = self._collect()
        if not candidate.is_configured:
            QMessageBox.warning(self, "提示", "请先填写 API Key 与 Base URL")
            return
        self.btn_test.setEnabled(False)
        self.btn_test.setText("测试中...")
        try:
            result = suggest_title(
                candidate,
                NamingInput(file_name="测试音频.mp3", duration_sec=180.0,
                            features_desc="时长 3分0秒；节奏约 90 BPM（中速）"),
            )
            QMessageBox.information(self, "连接成功", f"模型返回示例歌名：{result.title}")
        except AiNamerError as e:
            QMessageBox.critical(self, "连接失败", str(e))
        finally:
            self.btn_test.setEnabled(True)
            self.btn_test.setText("测试连接")

    def _on_save(self) -> None:
        new_settings = self._collect()
        try:
            save_settings(new_settings)
        except OSError as e:
            QMessageBox.critical(self, "保存失败", str(e))
            return
        self.result_settings = new_settings
        self.accept()
