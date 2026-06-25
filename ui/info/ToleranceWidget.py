from PyQt6.QtCore import Qt, QSize, QEvent, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox,
    QSizePolicy, QToolTip, QSlider
)

from ui.info.MistakeWidget import _svg_icon

class ToleranceWidget(QWidget):
    """
    Small panel beneath the MistakeWidget for tuning the string-edit
    `tolerance` — the semitone slack within which a played note still counts as
    correct.

    Presented to the user purely in **semitones**: a graduated slider (1-5) plus
    an editable readout that is the source of truth. The box accepts manual
    values, including ones beyond the slider's range. The widget converts back to
    raw `tolerance` and emits `tolerance_applied`; app.py owns the Config and
    re-runs detect_mistakes.
    """
    tolerance_applied = pyqtSignal(float)  # emits the new tolerance (raw units)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._HELP_TEXT = "How close to the intended note (in semitones)\nthe user can play to be counted correct.\n1 = Nearest semitone, 2 = Nearest whole step, etc."
        # The user only ever sees/edits *semitones*. The Config stores a raw `tolerance`
        # (the string-edit semitone slack); the two are linearly related: each whole
        # semitone step on the slider is worth `_TOL_PER_SEMITONE` of raw tolerance.
        #   semitones 1 2 3 4 5  <->  tolerance 0.25 0.5 0.75 1.0 1.25
        self._TOL_PER_SEMITONE = 0.25
        self._SEMITONE_MIN = 1
        self._SEMITONE_MAX = 5
        self._DEFAULT_SEMITONES = 2  # -> 0.5 tolerance

        self.init_ui()
    
    def init_ui(self):
        self._check_icon = _svg_icon("check.svg")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        help_icon = _svg_icon("circle-help.svg", px=32)
        self.help_label = QLabel()
        self.help_label.setPixmap(help_icon.pixmap(QSize(14, 14)))
        self.help_label.setToolTip(self._HELP_TEXT)
        self.help_label.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        # hover tooltips can silently fail to appear on macOS (see below), so the
        # icon is also clickable: a click forces the same text into a small popup.
        self.help_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self.help_label.installEventFilter(self)
        layout.addWidget(self.help_label)

        layout.addWidget(QLabel("Tolerance:"))

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(self._SEMITONE_MIN)
        self.slider.setMaximum(self._SEMITONE_MAX)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(1)
        self.slider.setValue(self._DEFAULT_SEMITONES)

        # square off the round themed handle into a rectangle
        self._ACCENT = "#8ab4f8" # match qdarktheme
        self.slider.setStyleSheet(
            "QSlider::handle:horizontal {"
            f"  background-color: {self._ACCENT};"
            "  width: 12px;"
            "  border-radius: 2px;"
            "}"
        )
        self.slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self.slider, 1)

        # editable semitone readout — the source of truth on Apply.
        self.value_box = QLineEdit()
        self.value_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.value_box.setFixedWidth(48)
        self.value_box.setToolTip("Tolerance in semitones")
        self.value_box.setText(self._fmt(self._DEFAULT_SEMITONES))
        self.value_box.editingFinished.connect(self._on_text_edited)
        layout.addWidget(self.value_box)

        self.apply_button = QPushButton()
        self.apply_button.setIcon(self._check_icon)
        self.apply_button.setIconSize(QSize(16, 16))
        self.apply_button.setFixedSize(QSize(28, 28))
        self.apply_button.setToolTip("Apply")
        self.apply_button.clicked.connect(self._on_apply)
        layout.addWidget(self.apply_button)

        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self.setMaximumHeight(self.sizeHint().height())

    def eventFilter(self, obj, event):
        """Clicking the help icon shows its text in a small popup, right on top
        of the icon — a reliable fallback for when hover tooltips don't fire."""
        if obj is self.help_label and event.type() == QEvent.Type.MouseButtonPress:
            QToolTip.showText(
                self.help_label.mapToGlobal(self.help_label.rect().topLeft()),
                self._HELP_TEXT,
                self.help_label,
            )
            return True
        return super().eventFilter(obj, event)

    @staticmethod
    def _fmt(semitones: float) -> str:
        return f"{semitones:g}"

    def _read_semitones(self):
        """Parse the box (the source of truth); None if blank/invalid/negative."""
        try:
            semitones = float(self.value_box.text().strip())
        except ValueError:
            return None
        return semitones if semitones >= 0 else None

    def set_tolerance(self, tolerance: float):
        """Reflect the active recording's current tolerance in the box + slider
        (no emit). Box is authoritative; slider clamps to its range."""
        semitones = self._tolerance_to_semitones(tolerance)
        self.value_box.setText(self._fmt(semitones))
        self._sync_slider_to(semitones)

    def _sync_slider_to(self, semitones: float):
        """Move the slider to match `semitones`, clamped to its range, without
        looping back into the box (values past the range stay only in the box)."""
        clamped = round(min(max(semitones, self._SEMITONE_MIN), self._SEMITONE_MAX))
        blocked = self.slider.blockSignals(True)
        self.slider.setValue(clamped)
        self.slider.blockSignals(blocked)

    def _on_slider_changed(self, semitones: int):
        # user dragged the slider -> it becomes the source of truth.
        self.value_box.setText(self._fmt(semitones))

    def _on_text_edited(self):
        semitones = self._read_semitones()
        if semitones is None:
            return
        self._sync_slider_to(semitones)

    def _on_apply(self):
        semitones = self._read_semitones()
        if semitones is None:
            QMessageBox.warning(
                self, "Invalid tolerance",
                "Enter a non-negative number of semitones (e.g. 2).",
            )
            return
        self.tolerance_applied.emit(self._semitones_to_tolerance(semitones))

    # helper helpers
    def _semitones_to_tolerance(self, semitones: float) -> float:
        return semitones * self._TOL_PER_SEMITONE
    def _tolerance_to_semitones(self, tolerance: float) -> float:
        return tolerance / self._TOL_PER_SEMITONE