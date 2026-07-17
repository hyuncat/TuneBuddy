from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel

from app_logic.user.NoteInfo import NoteInfo
from ui.Icons import svg_pixmap
from ui.info.Gradient import VolumeGradient


class NotePopupGH(QFrame):
    """Small popup next to the cursor with a clicked GuitarHero user note's
    characteristics (see NoteInfo). Qt.Popup, so any click outside dismisses
    it. (The ScoreViewer's counterpart is NotePopupSV, rendered by the JS.)"""

    # mistake.type -> display label, mirroring the MistakeWidget timing tab
    _TIMING_LABELS = {"early": "Early", "late": "Late",
                      "long": "Too long", "short": "Too short"}

    def __init__(self, chars: NoteInfo, parent=None):
        super().__init__(parent, flags=Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.setStyleSheet("""
            NotePopupGH {
                background-color: rgb(32, 33, 38);
                border: 1px solid rgb(95, 95, 105);
                border-radius: 6px;
            }
            QLabel {
                color: rgb(228, 231, 235);
                font-size: 12px;
                background: transparent;
                border: none;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 9, 12, 9)
        layout.setSpacing(3)

        for text in self._text_rows(chars):
            label = QLabel(text)
            label.setTextFormat(Qt.TextFormat.RichText)
            layout.addWidget(label)

    def _text_rows(self, chars: NoteInfo) -> list[str]:
        rows = [f"<b>Pitch:</b> {chars.note_name} {chars.cents:+.0f}¢"]

        onset = f"<b>Onset:</b> {chars.onset:.2f}s"
        if chars.onset_mistake:
            onset += f" ({self._TIMING_LABELS[chars.onset_mistake]})"
        rows.append(onset)

        duration = f"<b>Duration:</b> {chars.duration:.2f}s"
        if chars.duration_mistake:
            duration += f" ({self._TIMING_LABELS[chars.duration_mistake]})"
        rows.append(duration)

        rows.append(f"<b>Volume:</b> {chars.volume_abs_db:.1f} dB" if chars.volume_abs_db is not None else "— dB")

        if chars.vibrato_rate_hz is not None:
            rows.append(f"<b>Vibrato:</b> f={chars.vibrato_rate_hz:.1f}Hz, "
                        f"A={chars.vibrato_extent_cents:.0f}¢")
        else:
            rows.append("<b>Vibrato:</b> —")
        return rows

    def popup_at(self, global_pos: QPoint):
        """Show next to the cursor, nudged to stay on screen."""
        self.adjustSize()
        screen = QGuiApplication.screenAt(global_pos) or QGuiApplication.primaryScreen()
        geo = screen.availableGeometry()
        x = min(global_pos.x() + 12, geo.right() - self.width())
        y = min(global_pos.y() + 12, geo.bottom() - self.height())
        self.move(max(geo.left(), x), max(geo.top(), y))
        self.show()
