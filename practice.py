# code for practice mode
from __future__ import annotations
import os
import time
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QStatusBar, QPushButton, QLabel, QTreeWidget, QTreeWidgetItem, QSplitter,
    QInputDialog, QMenu, QMessageBox, QStackedLayout
)
from PyQt6.QtCore import Qt, QSize, QTimer, QPoint, pyqtSignal, QObject
from PyQt6.QtGui import QIcon

from app_logic.midi.ScoreData import ScoreData
from app_logic.user.ds.Recording import Recording

from app_logic.user.AudioPlayer import AudioPlayer
from app_logic.user.AudioRecorder import AudioRecorder
from app_logic.midi.ScoreData import ScoreData
from app_logic.midi.MidiSynth import MidiSynth
from app_logic.midi.MidiPlayer import MidiPlayer

# adjust this import to wherever your GuitarHero widget lives
from ui.ScoreViewer import ScoreViewer
from ui.GuitarHero import GuitarHero
from ui.info.Toolbar import Toolbar
from ui.info.StatusBar import StatusBar
from ui.time.CountdownTimer import CountdownTimer
from ui.time.WallClock import WallClock
from ui.time.Slider import Slider


class PracticeAttune(QMainWindow):
    """
    Standalone practice-mode window.

    For now this is just a thin shell around a GuitarHero widget.
    Later it can own its own playback controls, countdown, scoring,
    user recording logic, note-hit detection, etc.
    """

    def __init__(self, score_data: ScoreData, midi_synth: MidiSynth=None, parent=None):
        super().__init__(parent)
        self.score_data = score_data
        self.recording = Recording(score_data=self.score_data)
        self.wall_clock = WallClock(hz=10)

        self.is_playing = False
        self.is_recording = False
        self.is_counting_in = False

        # While RECORDING, practice mode is driven by the PitchDetector's emitted
        # pitch times (not the wall clock): the audio->pitch buffer only advances
        # its read time when the user's pitch matches the score, so the last
        # emitted time IS the playhead. `practice_time` is that time; repaints are
        # throttled (the detector emits hundreds of frames/sec) to ~30 fps.
        self.practice_time = 0.0
        self._RENDER_INTERVAL = 1.0 / 30.0
        self._last_render = 0.0

        # score viewer renders only the active instrument's part (matches the
        # main app default); never the full score in practice mode.
        self.viewer_show_full = False

        self.audio_recorder = AudioRecorder(self.recording)
        self.midi_synth = midi_synth if midi_synth is not None else MidiSynth("resources/MuseScore_General.sf3")
        self.midi_player = MidiPlayer(self.midi_synth, self.wall_clock)
        self.midi_player.load_score(self.score_data)

        self.init_ui()
        self.init_slider_layout()
        self.init_signals()

    def init_ui(self):
        """Create the basic window layout."""
        self.setWindowTitle("Practice Mode")
        self.resize(1200, 700)

        central = QWidget(self)
        self.setCentralWidget(central)

        self._layout = QVBoxLayout(central)
        self._layout.setContentsMargins(12, 12, 12, 12)
        self._layout.setSpacing(8)

        ABSOLUTE_PROJECT_ROOT = Path(__file__).resolve().parent

        # score viewer requires a loading screen until Verovio's JS is ready
        self.score_viewer_container = QWidget()
        stack = QStackedLayout(self.score_viewer_container)
        self.score_viewer = ScoreViewer(project_root=ABSOLUTE_PROJECT_ROOT)
        loading = QLabel("Loading...")
        loading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        stack.addWidget(loading)
        stack.addWidget(self.score_viewer)
        stack.setCurrentWidget(loading)  # show loading screen until viewer is ready
        self.score_viewer.load_finished.connect(lambda ok: stack.setCurrentIndex(1) if ok else 0)

        self.guitar_hero = GuitarHero(self.recording)
        self.guitar_hero.load_score(self.score_data)

        # score viewer stacked ON TOP of the guitar hero, in a vertical splitter
        # so both are adjustable in height (mirrors the main app's center column).
        self.center_splitter = QSplitter(Qt.Orientation.Vertical)
        self.center_splitter.addWidget(self.score_viewer_container)
        self.center_splitter.addWidget(self.guitar_hero)
        self.center_splitter.setStretchFactor(0, 1)  # score viewer grows
        self.center_splitter.setStretchFactor(1, 1)  # guitar hero grows
        self.center_splitter.setSizes([180, 520])     # initial heights (resizable)
        self._layout.addWidget(self.center_splitter)

        # --- UTILITIES --- 
        self.status_bar = StatusBar() # with default recording name
        self.status_bar.update_status("Ready...")
        self.setStatusBar(self.status_bar)
        self.countdown_timer = CountdownTimer(self.status_bar, midi_synth=self.midi_synth)

    def init_slider_layout(self):
        """
        Initialize the layout containing the play/pause, 
        record button and the slider.
        """
        self.slider_layout = QHBoxLayout()

        # get the play/pause button icons
        app_directory = os.path.dirname(__file__) 
        play_filepath = os.path.join(app_directory, 'resources', 'icons', 'play.png')
        pause_filepath = os.path.join(app_directory, 'resources', 'icons', 'pause.png')
        record_filepath = os.path.join(app_directory, 'resources', 'icons', 'record.png')

        self.play_icon = QIcon(play_filepath)
        self.pause_icon = QIcon(pause_filepath)
        self.record_icon = QIcon(record_filepath)

        # play button
        self.play_button = QPushButton()
        self.play_button.setIcon(self.play_icon)
        self.play_button.setFixedSize(QSize(26, 26))
        self.play_button.clicked.connect(self.toggle_playback)
        self.slider_layout.addWidget(self.play_button)

        # record button
        self.record_button = QPushButton()
        self.record_button.setIcon(self.record_icon)
        self.record_button.setFixedSize(QSize(26, 26))
        self.record_button.clicked.connect(self.toggle_recording)
        self.slider_layout.addWidget(self.record_button)

        # time label (current/total)
        self.time_label = QLabel("00:00.0 / 00:00.0")
        self.time_label.setMinimumWidth(100)
        self.slider_layout.addWidget(self.time_label)

        # the slider
        self.slider = Slider(self.wall_clock)
        self.slider_layout.addWidget(self.slider)
        self._layout.addLayout(self.slider_layout)

    def init_signals(self):
        # timekeeping signals
        self.wall_clock.time_changed.connect(self.time_changed)
        self.slider.slider_changed.connect(self.slider_changed)
        self.slider.slider_end.connect(self.slider_end)
        self.countdown_timer.finished.connect(self._start_recording)
        self.score_viewer.load_finished.connect(self.on_score_viewer_loaded)
        # the plot is the master view: keep the slider following it (during
        # recording the plot is driven by emitted pitch times, not the clock)
        self.guitar_hero.plot_moved.connect(self.slider.handle_timer_update)
        self.recording.pitch_detector.pitch_detected.connect(self.pitch_detected)

    # --- PLAYBACK / RECORDING TOGGLES ---
    def toggle_playback(self):
        if not self.is_playing:
            self._start_playback()

        elif self.is_playing:
            self._stop_playback()

    def _start_playback(self):
        t = self.slider.get_time()
        self.is_playing = True
        # plain playback is wall-clock driven (recording is pitch driven); the
        # wall clock's stall is unused now, so nothing to reset here.
        self.wall_clock.start(t)
        self.midi_player.play(start_time=t)
        # update UI
        self.play_button.setIcon(self.pause_icon)

    def _stop_playback(self):
        self.is_playing = False
        self.wall_clock.pause()
        self.midi_player.stop()
        # update UI
        self.play_button.setIcon(self.play_icon)

    def toggle_recording(self):
        if self.is_counting_in:
            # clicking again during the count-in cancels it (un-arm recording)
            self.countdown_timer.cancel()
            self.is_counting_in = False
            self.record_button.setIcon(self.record_icon)
            return
        if not self.is_recording:
            # play a one-measure metronome count-in; _start_recording on finish
            self.is_counting_in = True
            self.record_button.setIcon(self.pause_icon)
            self.countdown_timer.start(
                beats=self.score_data.count_in_beats(),
                channel=self.score_data.metronome_channel,
            )
        else:
            self._stop_recording()

    def _start_recording(self):
        """Called when the countdown timer finishes, to start recording.

        Recording is NOT driven by the wall clock — the slider/cursor follow the
        PitchDetector's emitted pitch times instead (see pitch_detected), which
        only advance on a correct pitch. So we pause the wall clock here (a tick
        would otherwise jump the slider before any pitch is validated) and let
        the buffer start unblocked at the slider's current time.
        """
        # update UI
        self.record_button.setIcon(self.pause_icon)
        self.is_counting_in = False
        self.midi_player.stop() # stop things we don't want
        # the wall clock must NOT advance the slider during recording
        self.wall_clock.pause()
        self.recording.pitch_detector.block = False
        # seed the pitch-driven playhead at the current slider position
        t = self.slider.get_time()
        self.practice_time = t
        self._last_render = 0.0
        self.is_recording = True
        self.audio_recorder.run(start_time=t)
        self.recording.pitch_detector.run(start_time=t)


    def _stop_recording(self):
        """Called when user clicks the record button while already recording, 
        to stop the recording and playback."""
        # update UI
        self.record_button.setIcon(self.record_icon)
        # stuff
        self.is_recording = False
        self.wall_clock.pause()
        self.audio_recorder.stop()
        self.recording.pitch_detector.stop()
    
    def time_changed(self, t: float):
        """Wall clock tick. Only drives the views during plain PLAYBACK — while
        recording the wall clock is paused and the views follow the emitted pitch
        times instead (see pitch_detected)."""
        if not self.is_playing:
            return
        self.score_data.update_time(t)
        self.score_viewer.set_playback_time(self._score_viewer_time(t))
        self.guitar_hero.move_plot(t)

    def pitch_detected(self, t: float):
        """Master driver while recording. `t` is the time of the just-emitted
        pitch frame; because the audio->pitch buffer only advances its read time
        on a correct pitch (we block it otherwise), `t` only moves forward when
        the user matches the score. So we use `t` directly as the playhead:
        (1) decide whether this frame matched and block/unblock the buffer for the
        NEXT frame, then (2) move every view to `t` (throttled).
        """
        if not self.is_recording:
            return
        # block the buffer (freeze the next emitted time) until the user lands the
        # right note; unblock to let `t` advance. This is the whole mechanism by
        # which the playhead only moves forward on a correct pitch.
        self.recording.pitch_detector.block = not self._pitch_matches(t)
        # `t` is the time of the last emitted pitch -> drive the whole UI from it
        self.practice_time = t
        self._render_practice(t)

    def _pitch_matches(self, t: float) -> bool:
        """Whether the detected pitch at time `t` should let the playhead advance.

        Advance (True) when there's no note to hold for — a gap, before the first
        note / after the last, or a rest (midi -1) — otherwise the playhead would
        deadlock on a spot with no note. For a real note, advance only when a
        clean, finite pitch lands within a semitone of the target; silence, an
        unvoiced/too-noisy frame, a NaN/inf candidate, or a wrong pitch all hold.
        """
        note_data = self.score_data.note_datas.get(self.score_data.active_instrument)
        target = note_data.read_current_note(t) if note_data else None
        m = target.midi_num[0] if target is not None else None

        if m is None or m == -1:
            return True

        p = self.recording.pitch_data.read_pitch(t)
        unv_thresh = self.recording.pitch_data.UNVOICED_THRESHOLD
        if p is None or not p.candidates or p.unvoiced_prob >= unv_thresh:
            self.status_bar.update_status(f"Waiting for note: {m:.1f}…")
            return False

        u = p.candidates[0][0]
        # NaN/inf guard: abs(nan - m) <= 1 is False anyway, but be explicit so a
        # garbage candidate can never be read as "on pitch".
        if not np.isfinite(u):
            self.status_bar.update_status(f"Waiting for note: {m:.1f}…")
            return False

        on_pitch = abs(u - m) <= 1
        state = "On" if on_pitch else "Off"
        self.status_bar.update_status(f"{state}! Detected note: {u:.1f}, Target note: {m:.1f}")
        return on_pitch

    def _render_practice(self, t: float):
        """Throttled repaint of the practice playhead at time `t`. The detector
        emits hundreds of frames/sec; repainting (and calling the Verovio JS
        cursor) on every one would swamp the UI, so cap redraws to ~30 fps. `t`
        itself is always current — only the redraw is throttled."""
        now = time.monotonic()
        if now - self._last_render < self._RENDER_INTERVAL:
            return
        self._last_render = now
        self.score_data.update_time(t)
        self.update_time_label(t)
        self.score_viewer.set_playback_time(self._score_viewer_time(t))
        # move_plot moves the timeline + redraws user pitch dots, and emits
        # plot_moved -> the slider follows it
        self.guitar_hero.move_plot(t)

    def slider_changed(self, t: float):
        """Called when the slider moves. Only acts when the user is scrubbing
        (neither playing nor recording) — during recording the slider is moved
        programmatically to follow the pitch playhead, so we must ignore those
        echoes here to avoid double-rendering / feedback."""
        if self.is_playing or self.is_recording:
            return
        # else, move the score viewer cursor and guitar hero plot
        self.update_time_label(t)
        self.score_data.update_time(t)
        self.score_viewer.set_playback_time(self._score_viewer_time(t))
        self.guitar_hero.move_plot(t)

    def slider_end(self, t: float):
        self._stop_recording()
        self._stop_playback()

    def update_time_label(self, t: float):
        """Update the 'current / total' time label from time `t` (sec)."""
        def fmt(seconds: float) -> str:
            mins = int(seconds // 60)
            secs = seconds % 60
            return f"{mins:02}:{secs:04.1f}"
        self.time_label.setText(f"{fmt(t)} / {fmt(self.slider.get_total_time())}")

    # --- SCORE VIEWER ---
    def _score_viewer_time(self, t: float) -> float:
        """Map a wall-clock time `t` (current tempo) back into the *original*
        score-tempo timeframe that Verovio's timemap uses, so the cursor stays
        aligned after a tempo change. Mirrors Attune._score_viewer_time."""
        bpm_og = self.score_data.bpm_og or self.score_data.bpm
        if not bpm_og:
            return t
        return t * self.score_data.bpm / bpm_og

    def refresh_score_viewer(self):
        """(Re)render the score viewer to match the active instrument's part.
        The single expensive Verovio layout step — never called per tick."""
        if self.score_data is None or self.score_data.score is None:
            return
        channel = None if self.viewer_show_full else self.score_data.active_instrument
        self.score_viewer.load_score(self.score_data, channel=channel)

    def on_score_viewer_loaded(self, ok: bool = True):
        """Practice score viewer finished loading its JS API; render whatever
        score is currently loaded (no-op if none yet)."""
        self.refresh_score_viewer()

    def load_score(self, score_data: ScoreData):
        """Load a score into the practice mode, initializing the recording and guitar hero with the new score data."""
        self.score_data = score_data
        self.recording = Recording(score_data=self.score_data)
        self.guitar_hero.load_score(self.score_data)
        self.guitar_hero.load_user(self.recording)
        self.midi_player.load_score(self.score_data)
        self.audio_recorder.load_recording(self.recording)
        self.recording.pitch_detector.pitch_detected.connect(self.pitch_detected)
        self.slider.update_range(score_data=self.score_data)
        # render the score into the viewer (no-op if its JS API isn't ready yet;
        # on_score_viewer_loaded re-renders once it is).
        self.refresh_score_viewer()

    def closeEvent(self, event):
        """Hook for cleanup later."""
        # future: stop timers / playback / recording threads here
        super().closeEvent(event)