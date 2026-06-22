from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox, QSizePolicy,
)

from ui.info.MistakeWidget import _svg_icon


class TolerancePanel(QWidget):
    """
    Small panel beneath the MistakeWidget for tuning the string-edit
    `tolerance` — the semitone slack within which a played note still counts as
    correct.

    Mirrors the InstrumentPanel's Range / Tuning rows: a labeled input plus a
    "check" Apply button. The widget only collects/validates the value and emits
    `tolerance_applied`; app.py owns the Config and re-runs detect_mistakes.
    """
    tolerance_applied = pyqtSignal(float)  # emits the new tolerance (semitones)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._check_icon = _svg_icon("check.svg")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        layout.addWidget(QLabel("Tolerance:"))
        self.tolerance_input = QLineEdit()
        self.tolerance_input.setPlaceholderText("0.3")
        self.tolerance_input.setText("0.3")
        layout.addWidget(self.tolerance_input, 1)

        self.apply_button = QPushButton()
        self.apply_button.setIcon(self._check_icon)
        self.apply_button.setIconSize(QSize(16, 16))
        self.apply_button.setFixedSize(QSize(28, 28))
        self.apply_button.setToolTip("Apply tolerance")
        self.apply_button.clicked.connect(self._on_apply)
        layout.addWidget(self.apply_button)

        # stay compact: this sits in a vertical splitter under the MistakeWidget,
        # which should take the vertical slack.
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self.setMaximumHeight(self.sizeHint().height())

    def set_tolerance(self, tolerance: float):
        """Reflect the active recording's current tolerance in the input (no emit)."""
        self.tolerance_input.setText(f"{tolerance:g}")

    def _on_apply(self):
        try:
            tolerance = float(self.tolerance_input.text().strip())
        except ValueError:
            tolerance = None
        if tolerance is None or tolerance < 0:
            QMessageBox.warning(
                self, "Invalid tolerance",
                "Enter a non-negative number for the tolerance (e.g. 0.3).",
            )
            return
        self.tolerance_applied.emit(tolerance)
