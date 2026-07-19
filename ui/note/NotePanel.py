from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QStackedWidget
)

from app_logic.user.ds.Recording import Recording
from ui.guitarhero.GuitarHero import GuitarHero
from ui.Icons import svg_pixmap
from ui.note.NoteCurveWidget import NoteCurveWidget
from ui.note.VolumeWidget import VolumeWidget
from ui.note.VibratoWidget import VibratoWidget
from ui.note.TimbreWidget import TimbreWidget


class NotePanel(QWidget):
    """Right-column per-note inspector: a combo picks which graph (Volume /
    Vibrato / Timbre) shows the note under the cursor; the host and tabs
    drive it like the other view surfaces (set_recording / set_live /
    update_time / refresh). Orchestration only — each graph owns its own
    rendering, legends included."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(6, 6, 6, 6)
        self._layout.setSpacing(4)
        self._t = 0.0

        self.widgets: dict[str, NoteCurveWidget] = {
            "Volume": VolumeWidget(),
            "Vibrato": VibratoWidget(),
            "Timbre": TimbreWidget(),
        }

        header = QHBoxLayout()
        header.setSpacing(6)
        self.combo = QComboBox()
        self.combo.addItems(list(self.widgets))
        self.combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.combo.setStyleSheet(GuitarHero._COMBO_STYLE)
        self.combo.currentTextChanged.connect(self._on_widget_changed)
        header.addWidget(self.combo)
        # every graph's extra controls live in the header row; only the
        # active graph's are visible (see _on_widget_changed)
        for w in self.widgets.values():
            for extra in w.header_widgets():
                extra.hide()
                header.addWidget(extra)
        header.addStretch(1)
        self.help_icon = QLabel()
        self.help_icon.setPixmap(svg_pixmap("circle-help.svg", 14))
        header.addWidget(self.help_icon)
        self._layout.addLayout(header)

        self.stack = QStackedWidget()
        for w in self.widgets.values():
            self.stack.addWidget(w)
        self._layout.addWidget(self.stack)

        self._on_widget_changed(self.combo.currentText())

    # --- host/tab-driven API (mirrors the other view surfaces) ---
    def set_recording(self, rec: Recording | None):
        for w in self.widgets.values():
            w.set_recording(rec)

    def set_live(self, live: bool):
        for w in self.widgets.values():
            w.set_live(live)

    def update_time(self, t: float):
        self._t = t
        self._active().update_time(t)

    def refresh(self):
        for w in self.widgets.values():
            w.refresh()

    def clear(self):
        """The active take's analysis/audio was wiped — re-resolve (to blank)."""
        self.refresh()

    # --- internals ---
    def _active(self) -> NoteCurveWidget:
        return self.widgets[self.combo.currentText()]

    def _on_widget_changed(self, name: str):
        active = self.widgets[name]
        self.stack.setCurrentWidget(active)
        self.help_icon.setToolTip(
            f"<div style='max-width: 260px;'>{active.HELP}</div>")
        for key, w in self.widgets.items():
            for extra in w.header_widgets():
                extra.setVisible(key == name)
        active.update_time(self._t)
