import re

from PyQt6.QtCore import Qt, QSize, QEvent, pyqtSignal, QStringListModel
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QLineEdit, QPushButton, QCompleter, QMessageBox, QToolTip,
    QSizePolicy, QScrollArea, QFrame,
)

from app_logic.midi.ScoreData import ScoreData
from resources.program_map import program_to_name
from ui.info.MistakeWidget import _svg_icon


# note-name <-> MIDI helpers, kept consistent with how MistakeWidget /
# Note.get_note_name() / MidiAxis display note names: sharps only, octave =
# midi // 12 - 1, so C4 == MIDI 60.
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_PITCH_CLASS = {name: i for i, name in enumerate(NOTE_NAMES)}
_NAME_RE = re.compile(r"^\s*([A-Ga-g])(#?)(-?\d+)\s*$")


def midi_to_name(m: int) -> str:
    """MIDI number -> note name, e.g. 60 -> 'C4'."""
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"


def name_to_midi(name: str) -> int | None:
    """Note name -> MIDI number, e.g. 'C4' -> 60. Returns None if unparseable."""
    match = _NAME_RE.match(name)
    if not match:
        return None
    letter, sharp, octave = match.group(1).upper(), match.group(2), int(match.group(3))
    pc = _PITCH_CLASS.get(letter + sharp)
    if pc is None:
        return None
    return (octave + 1) * 12 + pc


class SettingsWidget(QWidget):
    """
    Left-column panel (below the RecordingTree) holding the per-recording /
    per-score settings, stacked in a vertical scroll area so it can grow without
    capping the column:

      - "Instrument": a dropdown of every instrument detected in the score (one
        per MIDI channel, minus the metronome). Applying it makes that channel
        the active instrument (the analysis target).
      - "Range": low/high note inputs that auto-complete to the discretized note
        bins (MIDI 0..127, named like 'G3' / 'C#5'). Defaults are filled in from
        the active instrument's note range. Applying it sets the Config's
        fmin/fmax (so pitch detection is re-run).
      - "Tuning": the A4 reference pitch (Hz) the pitch detector tunes to.
      - "Transpose": a note-name input that transposes the WHOLE score so its
        first note lands on the entered pitch (relative to the first note). Emits
        the target MIDI; app.py shifts the score's MIDI / music21 / NoteData.

    The widget only collects/validates input and emits signals; app.py owns the
    Recording / ScoreData and performs the actual reset + re-analysis.
    """
    instrument_applied = pyqtSignal(int)   # emits the active-instrument channel
    range_applied = pyqtSignal(int, int)   # emits (lowest_midi, highest_midi)
    tuning_applied = pyqtSignal(float)     # emits the A4 reference tuning in Hz
    transpose_applied = pyqtSignal(int)    # emits the target MIDI for the first note
    full_score_toggled = pyqtSignal(bool)  # True = show full score, False = active instrument only

    # the discretized note bins the range / transpose inputs auto-complete to
    _NOTE_BIN_NAMES = [midi_to_name(m) for m in range(128)]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.score_data: ScoreData | None = None
        self._check_icon = _svg_icon("check.svg")
        self._first_note_midi: int | None = None
        self._TRANSPOSE_HELP = "Transposes the score relative to the pitch of the first note."

        # the whole panel scrolls vertically: a fixed-frame scroll area wraps an
        # inner content widget that all the rows live in (init_ui fills _layout).
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        content = QWidget()
        self._layout = QVBoxLayout(content)
        self._layout.setContentsMargins(6, 6, 6, 6)
        self._scroll.setWidget(content)
        outer.addWidget(self._scroll)

        self.init_ui()

        # keep the column narrow; height is left free so the vertical splitter
        # can size it and the inner scroll area handles any overflow.
        self.setMinimumWidth(180)
        self.setMaximumWidth(320)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

    # --- UI ---
    def init_ui(self):
        # a single shared model backs the range + transpose inputs' completion
        self._note_bin_model = QStringListModel(self._NOTE_BIN_NAMES, self)

        # --- ACTIVE INSTRUMENT SELECT ---
        row = QHBoxLayout()
        row.addWidget(QLabel("Instrument:"))
        self.instrument_combo = QComboBox()
        row.addWidget(self.instrument_combo, 1)

        self.instrument_apply = self._make_apply_button("Apply instrument")
        self.instrument_apply.clicked.connect(self._on_instrument_apply)
        row.addWidget(self.instrument_apply)
        self._layout.addLayout(row)

        # --- RANGE UI ---
        row = QHBoxLayout()
        row.addWidget(QLabel("Range:"))
        self.low_input = self._make_note_input("G3")
        row.addWidget(self.low_input, 1)
        row.addWidget(QLabel("—"))
        self.high_input = self._make_note_input("E7")
        row.addWidget(self.high_input, 1)

        self.range_apply = self._make_apply_button("Apply frequency range")
        self.range_apply.clicked.connect(self._on_range_apply)
        row.addWidget(self.range_apply)
        self._layout.addLayout(row)

        # --- TUNING UI ---
        # the A4 reference pitch (Hz) the pitch detector tunes to; applying it
        # updates the Config and re-runs pitch detection.
        row = QHBoxLayout()
        row.addWidget(QLabel("Tuning:"))
        self.tuning_input = QLineEdit()
        self.tuning_input.setPlaceholderText("440")
        self.tuning_input.setText("440")
        row.addWidget(self.tuning_input, 1)
        row.addWidget(QLabel("Hz"))

        self.tuning_apply = self._make_apply_button("Apply tuning")
        self.tuning_apply.clicked.connect(self._on_tuning_apply)
        row.addWidget(self.tuning_apply)
        self._layout.addLayout(row)

        # --- TRANSPOSE UI ---
        # transposes the WHOLE score so its first note lands on the entered pitch.
        row = QHBoxLayout()
        help_icon = _svg_icon("circle-help.svg", px=32)
        self.transpose_help = QLabel()
        self.transpose_help.setPixmap(help_icon.pixmap(QSize(14, 14)))
        self.transpose_help.setToolTip(self._TRANSPOSE_HELP)
        self.transpose_help.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        # hover tooltips can silently fail on macOS, so the icon is also clickable:
        # a click forces the same text into a small popup (mirrors ToleranceWidget).
        self.transpose_help.setCursor(Qt.CursorShape.PointingHandCursor)
        self.transpose_help.installEventFilter(self)
        row.addWidget(self.transpose_help)

        row.addWidget(QLabel("Transpose:"))
        self.transpose_input = self._make_note_input("C4")
        self.transpose_input.returnPressed.connect(self._on_transpose_apply)
        row.addWidget(self.transpose_input, 1)

        self.transpose_apply = self._make_apply_button("Apply transpose")
        self.transpose_apply.clicked.connect(self._on_transpose_apply)
        row.addWidget(self.transpose_apply)
        self._layout.addLayout(row)

        # keep the rows hugging the top; any extra column height is empty space.
        self._layout.addStretch(1)

    def _make_apply_button(self, tooltip: str) -> QPushButton:
        button = QPushButton()
        button.setIcon(self._check_icon)
        button.setIconSize(QSize(16, 16))
        button.setFixedSize(QSize(28, 28))
        button.setToolTip(tooltip)
        return button

    def _make_note_input(self, placeholder: str) -> QLineEdit:
        box = QLineEdit()
        box.setPlaceholderText(placeholder)
        completer = QCompleter(self._note_bin_model, box)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        box.setCompleter(completer)
        return box

    def eventFilter(self, obj, event):
        """Clicking the transpose help icon shows its text in a small popup on top
        of the icon — a reliable fallback for when hover tooltips don't fire."""
        if obj is self.transpose_help and event.type() == QEvent.Type.MouseButtonPress:
            QToolTip.showText(
                self.transpose_help.mapToGlobal(self.transpose_help.rect().topLeft()),
                self._TRANSPOSE_HELP,
                self.transpose_help,
            )
            return True
        return super().eventFilter(obj, event)

    # --- PUBLIC API ---
    def load_score(self, score_data: ScoreData):
        """Populate the instrument dropdown from a freshly loaded score, then
        sync the dropdown + range defaults to the current active instrument."""
        self.score_data = score_data
        self._populate_instruments()
        self.set_active_instrument(score_data.active_instrument)

    def set_active_instrument(self, channel: int):
        """Reflect `channel` in the dropdown and refresh the range defaults to
        that instrument's note range. Does not emit (no Apply)."""
        idx = self.instrument_combo.findData(channel)
        if idx >= 0:
            self.instrument_combo.setCurrentIndex(idx)
        if self.score_data is not None:
            self.populate_range_from_score(self.score_data, self.current_channel())

    def current_channel(self) -> int | None:
        """The channel currently selected in the dropdown, or None if empty."""
        data = self.instrument_combo.currentData()
        return int(data) if data is not None else None

    def populate_range_from_score(self, score_data: ScoreData, channel: int | None):
        """Prefill the low/high note inputs from the score's note range for the
        given instrument channel (defaults populated 'based on the score')."""
        if channel is None:
            return
        note_data = score_data.note_datas.get(channel)
        if note_data is None or not note_data.times:
            return
        midis = [
            n.midi_num[0] for n in note_data.data.values()
            if n.midi_num and n.midi_num[0] != -1
        ]
        if not midis:
            return
        low, high = int(round(min(midis))), int(round(max(midis)))
        self.low_input.setText(midi_to_name(low))
        self.high_input.setText(midi_to_name(high))

    def set_first_note(self, midi: int | None):
        """Reflect the score's current first-note pitch in the transpose input (no
        emit). Disables Apply when there's no note to anchor the transposition to."""
        self._first_note_midi = midi
        if midi is None:
            self.transpose_input.clear()
            self.transpose_input.setEnabled(False)
            self.transpose_apply.setEnabled(False)
            return
        self.transpose_input.setEnabled(True)
        self.transpose_apply.setEnabled(True)
        self.transpose_input.setText(midi_to_name(int(midi)))

    # --- INTERNAL ---
    def _populate_instruments(self):
        self.instrument_combo.blockSignals(True)
        self.instrument_combo.clear()
        if self.score_data is not None:
            for channel, program in self.score_data.instruments.items():
                if channel == self.score_data.metronome_channel:
                    continue  # don't expose the metronome click track
                self.instrument_combo.addItem(program_to_name(program), channel)
        self.instrument_combo.blockSignals(False)

    def _on_instrument_apply(self):
        channel = self.current_channel()
        if channel is None:
            return
        self.instrument_applied.emit(channel)

    def _on_range_apply(self):
        low = name_to_midi(self.low_input.text())
        high = name_to_midi(self.high_input.text())
        if low is None or high is None:
            QMessageBox.warning(
                self, "Invalid note",
                "Enter valid note names like 'G3' or 'C#5' for both ends of the range.",
            )
            return
        if low > high:
            low, high = high, low  # be forgiving if the inputs are swapped
        self.range_applied.emit(low, high)

    def set_tuning(self, tuning: float):
        """Reflect the active recording's current tuning (Hz) in the input,
        without emitting (no Apply). Mirrors the range defaults sync."""
        self.tuning_input.setText(f"{tuning:g}")

    def _on_tuning_apply(self):
        try:
            tuning = float(self.tuning_input.text().strip())
        except ValueError:
            tuning = None
        if tuning is None or tuning <= 0:
            QMessageBox.warning(
                self, "Invalid tuning",
                "Enter a positive number of Hz for the tuning (e.g. 440).",
            )
            return
        self.tuning_applied.emit(tuning)

    def _on_transpose_apply(self):
        if self._first_note_midi is None:
            return
        target = name_to_midi(self.transpose_input.text())
        if target is None:
            QMessageBox.warning(
                self, "Invalid note",
                "Enter a valid note name like 'C4' or 'F#3' to transpose the "
                "first note to.",
            )
            return
        self.transpose_applied.emit(target)
