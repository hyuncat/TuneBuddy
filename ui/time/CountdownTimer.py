from typing import TYPE_CHECKING

import mido
from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from ui.info.StatusBar import StatusBar
from app_logic.midi.MidiData import WOODBLOCK_PROGRAM, DOWNBEAT_NOTE, BEAT_NOTE

if TYPE_CHECKING:
    from app_logic.midi.MidiSynth import MidiSynth


class CountdownTimer(QObject):
    """A metronome count-in before recording.

    Plays one (or more) measures of woodblock clicks via the MidiSynth, then
    emits `finished` after the final beat so the caller can start recording.
    The clicks line up with the metronome the score already plays during
    recording, so there's no audible seam.
    """
    finished = pyqtSignal()

    # fallback when no beats are supplied (e.g. no score) -> 4 beats @ 120bpm
    DEFAULT_BEATS = [(0.0, True), (0.5, False), (1.0, False), (1.5, False)]

    def __init__(self, status_bar: StatusBar, midi_synth: "MidiSynth" = None):
        super().__init__()
        self.status_bar = status_bar  # reference to parent status bar to update msgs
        self.midi_synth = midi_synth

        # count-in state
        self.beats: list[tuple[float, bool]] = []
        self.i: int = 0  # index of the next beat to play
        self.interval_ms: int = 500
        self.channel: int = 0  # metronome channel, set in start()
        self._last_note: int | None = None  # last struck click, for note_off

        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._on_timeout)

    def start(self, beats: list[tuple[float, bool]] = None, channel: int = 0):
        """Start the count-in.

        Args:
            beats: list of (offset_sec, is_downbeat) tuples (see
                ScoreData.count_in_beats). Falls back to a 4-beat 4/4 measure.
            channel: MIDI channel to play the woodblock clicks on (the score's
                metronome channel, so the program is already loaded for it).
        """
        self.beats = list(beats) if beats else list(self.DEFAULT_BEATS)
        self.channel = channel if channel is not None else 0
        self.i = 0
        self._last_note = None

        # equal spacing within a measure; trailing beat reuses the same interval
        if len(self.beats) >= 2:
            self.interval_ms = int(round((self.beats[1][0] - self.beats[0][0]) * 1000))
        else:
            self.interval_ms = 500

        # make sure the woodblock program is loaded on the metronome channel
        if self.midi_synth is not None:
            self.midi_synth.handle_midi(mido.Message(
                'program_change', program=WOODBLOCK_PROGRAM,
                channel=self.channel, time=0))

        self._on_timeout()  # strike the first beat immediately, then schedule

    def cancel(self):
        """Abort an in-progress count-in (e.g. user un-arms recording)."""
        self.timer.stop()
        self._silence()
        self.i = 0
        self.status_bar.update_status("")

    def _on_timeout(self):
        if self.i < len(self.beats):
            _, is_downbeat = self.beats[self.i]
            self._click(is_downbeat)
            remaining = len(self.beats) - self.i
            self.status_bar.update_status(f"Count-in: {remaining}")
            self.i += 1
            self.timer.start(self.interval_ms)
        else:
            # one interval has elapsed since the last beat -> begin recording
            self.timer.stop()
            self._silence()
            self.finished.emit()
            self.status_bar.update_status("Recording...")

    def _click(self, is_downbeat: bool):
        """Strike a woodblock click, silencing the previous one first."""
        if self.midi_synth is None:
            return
        self._silence()
        note = DOWNBEAT_NOTE if is_downbeat else BEAT_NOTE
        self.midi_synth.handle_midi(mido.Message(
            'note_on', channel=self.channel, note=note, velocity=100, time=0))
        self._last_note = note

    def _silence(self):
        """Turn off the last struck click note, if any."""
        if self.midi_synth is not None and self._last_note is not None:
            self.midi_synth.handle_midi(mido.Message(
                'note_off', channel=self.channel, note=self._last_note, time=0))
        self._last_note = None
