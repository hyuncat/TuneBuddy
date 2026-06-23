# code for performance / analysis mode
from __future__ import annotations
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QSplitter, QStackedLayout, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal

from app_logic.midi.ScoreData import ScoreData
from app_logic.midi.MidiPlayer import MidiPlayer
from app_logic.user.ds.Recording import Recording
from app_logic.user.AudioPlayer import AudioPlayer
from app_logic.user.AudioRecorder import AudioRecorder
from app_logic.Alignment import Alignment
from app_logic.NoteData import NoteData

from ui.ScoreViewer import ScoreViewer
from ui.GuitarHero import GuitarHero


class PerformPanel(QWidget):
    """
    The "Perform" tab — record a take against the score, then **Analyze** it to
    detect and visualize mistakes. The counterpart to ``PracticePanel``: it owns
    the same view stack (a ScoreViewer over a GuitarHero) plus its own audio
    player/recorder and the whole offline analysis pipeline.

    It operates on the host's *active* ``Recording`` (handed in via
    :meth:`set_active_recording`, since recordings are created/selected in the
    RecordingTree, which stays app-side) and *shares* the bottom transport
    (play / record / slider / status bar / MIDI) injected via
    :meth:`attach_transport`. The host (``Attune``) routes the transport's button
    clicks and the shared clock/slider ticks to this panel whenever the Perform
    tab is active. The MistakeWidget (Perform-only side UI) is wired here too.
    """

    viewer_ready = pyqtSignal()   # the ScoreViewer's JS API finished loading
    analyzed = pyqtSignal()       # an Analyze pass just resized/aligned the score

    def __init__(self, score_data: ScoreData, parent=None):
        super().__init__(parent)
        self.score_data = score_data
        self.recording: Recording | None = None   # the active take (set by host)

        self.is_playing = False
        self.is_recording = False
        # mirrors the toolbar "User" checkbox: play the user's audio under the
        # score MIDI during plain playback.
        self.user_playback_enabled = True
        # set when Analyze is pressed while offline pitch detection is still
        # running; _on_detection_finished runs the deferred analyze once the
        # smoothed pitch track is ready (analyzing raw pitches gives garbage).
        self._pending_analyze = False
        # pitch_detectors we've already wired signals for (one per recording), so
        # _wire_detector never double-connects the same detector.
        self._wired_detectors: set = set()
        # score viewer: render only the active instrument's part (default) or the
        # full score (toggled via the instrument panel).
        self.viewer_show_full = False

        self.audio_player = AudioPlayer(None)
        self.audio_recorder = AudioRecorder(self.recording)

        # shared transport + Perform-only side UI, injected via attach_transport()
        self.wall_clock = None
        self.slider = None
        self.status_bar = None
        self.midi_player = None
        self.mistake_widget = None

        self.init_ui()
        self.init_signals()

    def init_ui(self):
        """Create the panel layout (just the views — the transport is shared)."""
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
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

        # score viewer stacked ON TOP of the guitar hero, in a vertical splitter
        # so both are adjustable in height.
        self.center_splitter = QSplitter(Qt.Orientation.Vertical)
        self.center_splitter.addWidget(self.score_viewer_container)
        self.center_splitter.addWidget(self.guitar_hero)
        self.center_splitter.setStretchFactor(0, 1)  # score viewer grows
        self.center_splitter.setStretchFactor(1, 1)  # guitar hero grows
        # start the score viewer compact so its single white page roughly fills
        # the box (still user-resizable via the handle below it).
        self.center_splitter.setSizes([180, 520])    # initial heights (resizable)
        self._layout.addWidget(self.center_splitter)

    def init_signals(self):
        self.score_viewer.load_finished.connect(self.on_score_viewer_loaded)

    def attach_transport(self, wall_clock, slider, status_bar, midi_synth, mistake_widget):
        """Inject the shared transport (owned by the host) plus the Perform-only
        MistakeWidget. The panel drives the transport during playback/recording;
        the host routes the matching button clicks / clock+slider ticks back. The
        MIDI player is the panel's OWN (sharing only the synth + clock) so it
        plays this tab's score independently of the Practice tab's."""
        self.wall_clock = wall_clock
        self.slider = slider
        self.status_bar = status_bar
        self.midi_player = MidiPlayer(midi_synth, wall_clock)
        self.mistake_widget = mistake_widget
        # keep the shared slider following the plot as it moves
        self.guitar_hero.plot_moved.connect(self.slider.handle_timer_update)
        # mistake list <-> guitar hero highlight/override coupling
        self.mistake_widget.selected.connect(self.on_mistake_selected)
        self.mistake_widget.cleared.connect(self.guitar_hero.clear_highlight)
        self.mistake_widget.override_toggled.connect(self.on_mistake_override_toggled)

    # --- GUARDS ---
    def _has_recording(self, warn=False) -> bool:
        """True if there's an active recording, else optionally warn."""
        if self.recording is None:
            if warn:
                QMessageBox.warning(self, "No recording selected", "Please select a recording first.")
            return False
        return True

    def _recording_is_empty(self, warn=False) -> bool:
        """True if the active recording has no audio (optionally warn)."""
        if not self._has_recording(warn=False):
            return False
        if self.recording.audio_data.end_index <= 0:
            if warn:
                QMessageBox.warning(self, "No audio available", "Please record or upload audio first.")
            return True
        return False

    def _has_analysis(self, warn=False) -> bool:
        """True if the active recording has been analyzed (optionally warn)."""
        if not self._has_recording():
            return False
        if not self.recording.has_analysis():
            if warn:
                QMessageBox.warning(self, "No analysis available", "Please analyze the recording first.")
            return False
        return True

    # --- HOST-DRIVEN STATE ---
    def set_active_recording(self, rec: Recording):
        """Make `rec` the active take. Wires its pitch detector once, then loads
        its analysis/audio into the views, audio engines and mistake list."""
        self.recording = rec
        self._wire_detector(rec)
        self.guitar_hero.load_user(rec)
        self.audio_player.load_audio(rec.audio_data)
        self.audio_recorder.load_recording(rec)
        self.mistake_widget.load_mistakes(rec.alignment.mistakes)

    def _wire_detector(self, rec: Recording):
        """Each recording owns its own pitch detector; connect each only once."""
        if rec is None or rec.pitch_detector in self._wired_detectors:
            return
        rec.pitch_detector.status_changed.connect(self.status_bar.update_status)
        rec.pitch_detector.detection_finished.connect(self._on_detection_finished)
        self._wired_detectors.add(rec.pitch_detector)

    def set_user_audio_enabled(self, enabled: bool):
        """Mirror the toolbar 'User' checkbox; stop playback now if turning off."""
        self.user_playback_enabled = enabled
        if not enabled:
            self.audio_player.stop()

    def load_score(self, score_data: ScoreData):
        """Point the views + the panel's MIDI player at a freshly loaded score
        (the host re-creates the active recording separately and calls
        set_active_recording)."""
        self.score_data = score_data
        if self.midi_player is not None:
            self.midi_player.load_score(self.score_data)
        self.guitar_hero.load_score(self.score_data)
        self.refresh_score_viewer()

    def cleanup(self):
        """Reset all analysis-derived state so no stale artifacts survive an input
        change (new score/audio, re-run detection, re-analyze): clears the active
        recording's notes/alignment/overrides, the mistake list, and the
        GuitarHero overlays."""
        rec = self.recording
        if rec is not None:
            rec.note_data = NoteData()
            rec.alignment = Alignment(config=rec.config)
            rec.overridden_mistake_indices = set()
            self.guitar_hero.load_user(rec)
        if self.mistake_widget is not None:
            self.mistake_widget.clear()

    def set_active_instrument(self, channel: int):
        """Make `channel` the active instrument: wipe analysis-derived data, re-init
        the algorithms from the (unchanged) Config, and re-render the views."""
        if not self._has_recording():
            return
        self.score_data.active_instrument = channel
        self.recording.active_instrument = channel
        self.cleanup()
        self.recording.update_config(self.recording.config)
        self.refresh_score_viewer()

    def on_full_score_toggled(self, show_full: bool):
        """Toggle the score viewer between the full score and the active part."""
        self.viewer_show_full = show_full
        self.refresh_score_viewer()

    # --- AUDIO / DETECTION ---
    def refresh_audio(self):
        """A new take's raw audio is loaded: make it playable and (re-)run offline
        pitch detection on it in the background."""
        if not self._has_recording():
            return
        self.audio_player.load_audio(self.recording.audio_data)
        self.detect_pitches()

    def detect_pitches(self):
        """Clear stale analysis, then (re-)run offline pitch detection on the
        active recording's audio in the background. When detection finishes,
        _on_detection_finished loads the fresh pitch data into view."""
        rec = self.recording
        if rec is None or rec.audio_data.end_index <= 0:
            return
        self.cleanup()
        self._wire_detector(rec)  # just in case
        rec.pitch_detector.detect_pitches_async()

    def detect_notes(self):
        """Run note detection on the current pitch data, then refresh the view."""
        if self.recording is None:
            return
        self.recording.detect_notes()
        self.guitar_hero.load_user(self.recording)

    def _detection_in_flight(self) -> bool:
        """True while offline pitch detection+smoothing is still running."""
        rec = self.recording
        if rec is None:
            return False
        thread = getattr(rec.pitch_detector, "offline_thread", None)
        return bool(thread and thread.is_alive())

    def _on_detection_finished(self):
        """Offline pitch detection finished (queued onto the main thread): clear
        the status, load the now-ready pitch data, and run a deferred Analyze."""
        self.status_bar.update_status("")
        self.guitar_hero.load_user(self.recording)
        if self._pending_analyze:
            self._pending_analyze = False
            self.analyze()

    # --- PLAYBACK / RECORDING (called by the host when this tab is active) ---
    def toggle_playback(self) -> bool:
        """Toggle plain playback. Returns the new is_playing state so the host can
        update the shared play button icon."""
        if self.is_playing:
            self.stop_playback()
        else:
            self.start_playback()
        return self.is_playing

    def start_playback(self):
        t = self.slider.get_time()
        self.is_playing = True
        self.wall_clock.start(t)
        self.midi_player.play(start_time=t)
        if self.user_playback_enabled:
            self.audio_player.play(start_time=t)
        self.status_bar.update_status("Playing...")

    def stop_playback(self):
        if not self.is_playing:
            return
        self.is_playing = False
        self.wall_clock.pause()
        self.midi_player.stop()
        self.audio_player.stop()
        self.status_bar.update_status("")

    def start_recording(self):
        """Start the take (host calls this after the shared count-in)."""
        t = self.slider.get_time()
        self.is_recording = True
        self.audio_player.stop()
        self.wall_clock.start(t)
        self.audio_recorder.run(start_time=t)
        self.recording.pitch_detector.run(start_time=t)
        self.midi_player.play(start_time=t)  # play whatever audio the user enabled

    def stop_recording(self):
        if not self.is_recording:
            return
        self.is_recording = False
        self.wall_clock.pause()
        self.audio_recorder.stop()
        self.midi_player.stop()
        self.recording.pitch_detector.stop()
        self.status_bar.update_status("")

    # --- VIEW DRIVING (called by the host's shared clock/slider dispatch) ---
    def _move_views(self, t: float):
        self.score_data.update_time(t)
        self.score_viewer.set_playback_time(self._score_viewer_time(t))
        self.guitar_hero.move_plot(t)

    def on_clock_tick(self, t: float):
        """Shared wall-clock tick: move the views during plain PLAYBACK only."""
        if not self.is_playing:
            return
        self._move_views(t)

    def on_slider_changed(self, t: float):
        """Shared slider moved: move the views unless we're the one playing
        (during recording the clock is running and the slider follows it, so the
        views should track it then too — hence we don't guard on is_recording)."""
        if self.is_playing:
            return
        self._move_views(t)

    def render_at(self, t: float):
        """Render the views at `t` (host uses this when the Perform tab becomes
        active, to line the cursor up with the shared slider's position)."""
        self._move_views(t)

    # --- ANALYSIS PIPELINE ---
    def _find_best_w2(self):
        """Find the note-detection frame size that minimizes mistakes, then set the
        config to it."""
        W2_SIZES = [33, 31, 29, 27, 25, 23, 21, 19, 17]

        min_mistake, best_w2 = float('inf'), None
        for w2 in W2_SIZES:
            self.recording.config.w2 = w2
            self.recording.config.h2 = w2 - 2
            self.recording.update_config(self.recording.config)
            self.recording.detect_notes()
            self.recording.detect_mistakes()

            if len(self.recording.alignment.mistakes) < min_mistake:
                min_mistake = len(self.recording.alignment.mistakes)
                best_w2 = w2

        print(f"best ND frame-size: {best_w2}, min mistakes: {min_mistake}")

        self.recording.config.w2 = best_w2
        self.recording.config.h2 = best_w2 - 2
        self.recording.update_config(self.recording.config)

    def mistake_correction_loop(self):
        """Run mistake correction until it stops improving."""
        if not self.recording or not self.recording.alignment:
            print("No active recording or alignment to correct mistakes on.")
            return
        n_mistakes = len(self.recording.alignment.mistakes)
        print(f"initial mistakes: {n_mistakes}")
        while True:
            # keep the current (best-so-far) state in case this pass regresses
            prev_nd = self.recording.note_data
            prev_alignment = self.recording.alignment

            self.recording.correct_mistakes()
            new_n_mistakes = len(self.recording.alignment.mistakes)
            print(f" > mistakes after correction: {new_n_mistakes}")

            if new_n_mistakes >= n_mistakes:
                # correction stopped helping -> revert to the better previous state
                self.recording.note_data = prev_nd
                self.recording.alignment = prev_alignment
                print("no improvement after correction, breaking loop")
                break
            n_mistakes = new_n_mistakes  # made progress -> raise the baseline

    def analyze(self):
        if not self._has_recording(warn=True) or self._recording_is_empty(warn=True):
            return

        rec = self.recording
        # don't analyze raw/partial pitches: if offline detection+smoothing is
        # still running in the background, defer until it finishes (the smoothed
        # track has the octave errors / noise cleaned up). _on_detection_finished
        # will re-call analyze() once the pitch_data is ready.
        if self._detection_in_flight():
            self._pending_analyze = True
            self.status_bar.update_status("Detecting pitches… will analyze when ready")
            return
        print("analyzing... ")
        self.cleanup()  # clear stale notes/alignment/mistakes before recomputing

        # detect notes (at the best ND frame size), stretch the score to match
        # the take's length (also rescales p.distances), then string-edit align
        self._find_best_w2()
        self.detect_notes()
        # flag high-slope transition frames now (after onset refinement pulls them
        # back into note spans) so update_alignment_distances leaves them grey
        # instead of letting slides bias a note's coloring / flag false mistakes.
        rec.detect_transitions()
        # re-median note pitches over non-transition frames so onset-refinement
        # slide frames don't bias a note sharp/flat (kills false "too sharp"
        # substitutions). Must run before detect_mistakes() / the alignment.
        rec.recompute_note_pitches()
        # drop phantom notes that are almost entirely slide (the wide ND window can
        # detect a "note" inside a long transition). Also before detect_mistakes().
        rec.prune_transition_notes()
        length = rec.get_length(raw=True)
        rec.resize(new_length=length)
        rec.detect_mistakes()
        self.mistake_correction_loop()

        # color the user pitches by the final alignment (insertions red, others by
        # distance to their *aligned* score note) instead of the live per-frame
        # distance. Runs after the correction loop so it reflects the final pairs.
        rec.update_alignment_distances()

        # reload every view with the fresh analysis (note/alignment may have been
        # overwritten by the correction loop)
        self.guitar_hero.load_alignment(rec.alignment)
        self.guitar_hero.load_user(rec)
        self.guitar_hero.update_view_items()
        self.slider.update_range(score_data=self.score_data, recording=rec)
        self.mistake_widget.load_mistakes(rec.alignment.mistakes)

        # the resize stretched the score to match the take => its BPM/length
        # changed; let the host reflect that in the tempo display.
        self.analyzed.emit()

    def reanalyze_if_analyzed(self):
        """Re-run Analyze only if the take has already been analyzed (used after a
        tolerance change, which only affects the string-edit step)."""
        if self._has_analysis(warn=False):
            self.analyze()

    # --- MISTAKE LIST <-> GUITAR HERO ---
    def on_mistake_selected(self, idx: int):
        """A mistake was clicked: highlight its note in the guitar-hero plot."""
        if self.recording is None:
            return
        mistakes = self.recording.alignment.mistakes
        if 0 <= idx < len(mistakes):
            self.guitar_hero.highlight_mistake(mistakes[idx])

    def on_mistake_override_toggled(self, idx: int):
        if self.recording is None:
            return
        self.recording.toggle_mistake_override(idx)
        mistake = self.recording.alignment.mistakes[idx]
        self.mistake_widget.refresh_override(idx)
        self.guitar_hero.update_highlight_override(mistake.is_overridden())
        self.guitar_hero.update_view_items()

    # --- SCORE VIEWER ---
    def _score_viewer_time(self, t: float) -> float:
        """Map a wall-clock time `t` (current tempo) back into the *original*
        score-tempo timeframe Verovio's timemap uses, so the cursor stays aligned
        after a tempo change."""
        bpm_og = self.score_data.bpm_og or self.score_data.bpm
        if not bpm_og:
            return t
        return t * self.score_data.bpm / bpm_og

    def refresh_score_viewer(self):
        """(Re)render the score viewer to match the current instrument/full-score
        state. The single expensive Verovio layout step — never called per tick."""
        if self.score_data is None or self.score_data.score is None:
            return
        channel = None if self.viewer_show_full else self.score_data.active_instrument
        self.score_viewer.load_score(self.score_data, channel=channel)

    def on_tempo_changed(self):
        """Mirror a tempo change: refresh the guitar-hero view to the new tempo.
        (ScoreData is shared, so change_tempo already ran; the Verovio render is
        kept and remapped via _score_viewer_time. The shared slider is re-ranged
        by the host.)"""
        self.guitar_hero.update_view_items()

    def on_score_viewer_loaded(self, ok: bool = True):
        """The ScoreViewer's JS API is ready: render whatever score is loaded, and
        let the host know (it loads the demo score on first ready)."""
        self.refresh_score_viewer()
        self.viewer_ready.emit()
