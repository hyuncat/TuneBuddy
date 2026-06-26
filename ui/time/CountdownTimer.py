import time
from typing import TYPE_CHECKING

import mido
from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal

from ui.info.StatusBar import StatusBar
from app_logic.midi.MidiData import WOODBLOCK_PROGRAM, DOWNBEAT_NOTE, BEAT_NOTE

if TYPE_CHECKING:
    from app_logic.midi.MidiSynth import MidiSynth


class CountdownTimer(QObject):
    """A metronome count-in that SCROLLS the views before recording.

    Unlike a plain "play N clicks then go", this drives the GuitarHero/score
    timeline through the whole count-in so the player can watch the notes
    approach the head instead of being surprised when recording snaps into
    motion. The geometry (with `head` = where the cursor sits when you hit
    record, `spb` = seconds per beat, `n` = beats in one count-in measure):

        * the plot starts one measure early, at `head - n*spb`, and scrolls in
          real time (1:1 with playback), `progress(plot_time)` driving the views;
        * the first `n-1` beats are struck audibly (4/4 -> "3", "2", "1");
        * the last beat's click is OMITTED; instead `finished(record_time)` fires
          and the status flips to "Go!". `record_time` is supplied by the caller
          (Perform starts a beat before the head for runway; Practice starts at
          the head), so the handoff to the recording clock has no visual jump.

    The clicks line up with the score's own metronome during recording, so
    there's no audible seam.
    """
    finished = pyqtSignal(float)   # emits record_time (sec): start capture here
    progress = pyqtSignal(float)   # emits plot time (sec) to scroll the views

    FPS = 30  # plot-scroll refresh rate during the count-in

    # fallback when no beats are supplied (e.g. no score) -> 4 beats @ 120bpm
    DEFAULT_BEATS = [(0.0, True), (0.5, False), (1.0, False), (1.5, False)]

    def __init__(self, status_bar: StatusBar, midi_synth: "MidiSynth" = None):
        super().__init__()
        self.status_bar = status_bar  # reference to parent status bar to update msgs
        self.midi_synth = midi_synth

        # count-in geometry (set in start())
        self.beats: list[tuple[float, bool]] = []
        self.channel: int = 0
        self._spb: float = 0.5            # seconds per beat
        self._n: int = 4                  # beats in the count-in measure
        self._plot_start: float = 0.0     # plot time where the count-in begins
        self._record_time: float = 0.0    # plot time where recording starts
        self._duration: float = 0.0       # count-in length in real seconds
        self._num_counting: int = 3       # audible counting clicks before record
        self._metronome_on: bool = False  # play the click ON the record beat?

        # runtime state
        self._active: bool = False
        self._t0: float = 0.0             # monotonic anchor for the count-in
        self._next_tick: int = 0          # index of the next audible click
        self._last_note: int | None = None

        self.timer = QTimer()
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.setInterval(max(1, int(round(1000 / self.FPS))))
        self.timer.timeout.connect(self._on_frame)

    def start(self, beats: list[tuple[float, bool]] = None, channel: int = 0,
              head_time: float = 0.0, record_time: float | None = None,
              metronome_on: bool = False):
        """Start the scrolling count-in.

        Args:
            beats: list of (offset_sec, is_downbeat) tuples (see
                ScoreData.count_in_beats). Falls back to a 4-beat 4/4 measure.
            channel: MIDI channel to play the woodblock clicks on (the score's
                metronome channel, so the program is already loaded for it).
            head_time: the head — where the cursor sits when record is pressed
                (first score note / clip start). The plot scrolls in from one
                measure before this.
            record_time: plot time at which recording should begin (emitted via
                `finished`). Defaults to one beat before the head.
            metronome_on: when the recording point falls ON a count-in beat (the
                Perform runway records a beat before the head, so it does), play
                that beat's click too iff this is set — matching the metronome
                toggle. Practice records at the head (no beat there), so all of its
                beats are counted and this has no effect.
        """
        self.beats = list(beats) if beats else list(self.DEFAULT_BEATS)
        self.channel = channel if channel is not None else 0
        self._metronome_on = metronome_on
        self._n = len(self.beats)
        if self._n >= 2:
            self._spb = self.beats[1][0] - self.beats[0][0]
        else:
            self._spb = 0.5

        if record_time is None:
            record_time = head_time - self._spb
        # record_time may be NEGATIVE (a Perform runway recorded before t=0); the
        # capture buffers handle that via a time origin, so don't clamp it here.
        self._record_time = record_time
        # the plot always scrolls in from one full measure before the head; the
        # count-in's audible part runs up to the recording point. So Perform
        # (record a beat before the head) counts the first n-1 beats and the n-th
        # falls ON the record point (its click is the metronome-gated one), while
        # Practice (record at the head) counts all n beats — "4, 3, 2, 1, Go".
        self._plot_start = head_time - self._n * self._spb
        self._duration = max(0.0, self._record_time - self._plot_start)
        self._num_counting = int(round(self._duration / self._spb)) if self._spb else 0
        self._next_tick = 0
        self._last_note = None
        self._active = True

        # make sure the woodblock program is loaded on the metronome channel
        if self.midi_synth is not None:
            self.midi_synth.handle_midi(mido.Message(
                'program_change', program=WOODBLOCK_PROGRAM,
                channel=self.channel, time=0))

        self.progress.emit(self._plot_start)  # jump the views to the lead-in start
        self._t0 = time.monotonic()
        self._on_frame()        # strike the first beat immediately, then schedule
        if self._active:        # (a zero-length count-in may already have finished)
            self.timer.start()

    def cancel(self):
        """Abort an in-progress count-in (e.g. user un-arms recording)."""
        self.timer.stop()
        self._silence()
        self._active = False
        self._next_tick = 0
        self.status_bar.update_status("")

    def _on_frame(self):
        """Driven at FPS: scroll the plot, strike any beats now due, and hand off
        to recording once the count-in measure has elapsed."""
        if not self._active:
            return
        elapsed = min(time.monotonic() - self._t0, self._duration)
        self.progress.emit(self._plot_start + elapsed)

        # strike every still-pending counting click whose beat time has arrived
        # (the beats strictly before the record point). The beat ON the record
        # point, if any, is handled in _finish (metronome-gated).
        while (self._next_tick < self._num_counting
               and elapsed >= self._next_tick * self._spb - 1e-6):
            _, is_downbeat = self.beats[self._next_tick]
            self._click(is_downbeat)
            remaining = self._num_counting - self._next_tick  # Perform 3,2,1 / Practice 4,3,2,1
            self.status_bar.update_status(f"Count-in: {remaining}")
            self._next_tick += 1

        if elapsed >= self._duration - 1e-9:
            self._finish()

    def _finish(self):
        """Count-in elapsed: stop, say 'Go!', and tell the caller to record. If the
        recording point lands on a count-in beat (Perform) and the metronome is on,
        click that beat too so the player gets the downbeat-lead-in tick."""
        self.timer.stop()
        if self._metronome_on and self._num_counting < self._n:
            _, is_downbeat = self.beats[self._num_counting]
            self._click(is_downbeat)  # the metronome-gated click ON the record beat
        else:
            self._silence()
        self._active = False
        self.status_bar.update_status("Go!")
        self.finished.emit(self._record_time)

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
