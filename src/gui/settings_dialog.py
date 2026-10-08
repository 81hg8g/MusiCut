"""API 配置对话框"""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
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
        form = QFormLayout()

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
        form.addRow("起名语言:", self.combo_lang)

        self.spin_conc = QSpinBox()
        self.spin_conc.setRange(1, 16)
        self.spin_conc.setValue(s.concurrency)
        form.addRow("并发数:", self.spin_conc)

        layout.addLayout(form)

        hint = QLabel(f"配置保存位置：{settings_path()}")
        hint.setStyleSheet("color: gray; font-size: 11px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        note = QLabel("提示：调用云端接口时，歌词与音频特征描述会上传到服务商，音频文件本身不会上传。")
        note.setStyleSheet("color: #946200; font-size: 11px;")
        note.setWordWrap(True)
        layout.addWidget(note)

        btn_row = QHBoxLayout()
        self.btn_test = QPushButton("测试连接")
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
