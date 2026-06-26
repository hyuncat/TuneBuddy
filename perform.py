# code for performance / analysis mode
from __future__ import annotations
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QSplitter, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal

from app_logic.midi.ScoreData import ScoreData
from app_logic.midi.MidiPlayer import MidiPlayer
from app_logic.user.ds.Recording import Recording
from app_logic.user.AudioPlayer import AudioPlayer
from app_logic.user.AudioRecorder import AudioRecorder
from app_logic.Alignment import Alignment
from app_logic.NoteData import NoteData
from app_logic.user.ds.PitchData import PitchData

from ui.ScoreViewer import ScoreViewer
from ui.GuitarHero import GuitarHero
from ui.time.ScoreTimeMap import ScoreTimeMap


class PerformTab(QWidget):

    viewer_ready = pyqtSignal()   # the ScoreViewer's JS API finished loading
    analyzed = pyqtSignal()       # an Analyze pass just resized/aligned the score
    clip_changed = pyqtSignal(object)  # (i0,i1) note-index clip or None; host mirrors it onto the other tab

    def __init__(self, score_data: ScoreData, parent=None):
        super().__init__(parent)
        # important data structures
        self.score_data = score_data
        self.recording: Recording | None = None   # the active take (set by host)

        # audio engines
        self.audio_player = AudioPlayer(None)
        self.audio_recorder = AudioRecorder(self.recording)

        # playback variables
        self.is_playing = False
        self.is_recording = False
        self.user_playback_enabled = True

        # pitch_detectors we've already wired signals for (one per recording), so
        # _wire_detector never double-connects the same detector.
        self._wired_detectors: set = set()
        # score viewer: render only the active instrument's part (default) or the
        # full score. The host (app.py) owns the toggle and pushes it in via
        # set_show_full; this is the panel's render-time cache of it.
        self.viewer_show_full = False

        # barline-anchored app<->Verovio time correspondence: keeps the score
        # cursor on the MIDI/NoteData timeline (the source of truth) instead of
        # Verovio's independently-drifting timemap. Rebuilt on every re-render.
        self._time_map = ScoreTimeMap()

        # injected via attach_timekeeping()
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

        # the score viewer owns its own "Loading..." placeholder until Verovio's
        # JS API is ready (see ScoreViewer).
        self.score_viewer = ScoreViewer(project_root=ABSOLUTE_PROJECT_ROOT)

        self.guitar_hero = GuitarHero(self.recording)

        # score viewer stacked ON TOP of the guitar hero, in a vertical splitter
        # so both are adjustable in height.
        self.center_splitter = QSplitter(Qt.Orientation.Vertical)
        self.center_splitter.addWidget(self.score_viewer)
        self.center_splitter.addWidget(self.guitar_hero)
        self.center_splitter.setStretchFactor(0, 1)  # score viewer grows
        self.center_splitter.setStretchFactor(1, 1)  # guitar hero grows
        # start the score viewer compact so its single white page roughly fills
        # the box (still user-resizable via the handle below it).
        self.center_splitter.setSizes([180, 520])    # initial heights (resizable)
        self._layout.addWidget(self.center_splitter)

    def init_signals(self):
        self.score_viewer.load_finished.connect(self.on_score_viewer_loaded)

    def attach_timekeeping(self, wall_clock, slider, status_bar, midi_synth, mistake_widget):
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

    # --- HOST-DRIVEN STATE ---
    def set_active_recording(self, rec: Recording):
        """Make `rec` the active take. Wires its pitch detector once, then loads
        its analysis/audio into the views, audio engines and mistake list."""
        self.recording = rec
        if rec.score_data is not self.score_data:
            self.load_score(rec.score_data)
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
        """Reset all state - unalive, then cleanup recording and mistake_widget."""
        # just in case!
        self.stop_recording()
        self.stop_playback()
        rec = self.recording
        if rec is not None:
            rec.cleanup()
            self.guitar_hero.load_user(rec)
        if self.mistake_widget is not None:
            self.mistake_widget.clear()

    def _clear_analysis(self):
        """Clear stale analysis (notes/alignment/mistakes/overrides) and refresh
        the views, but KEEP the recording's audio + pitch data. Used before a
        re-detection: unlike cleanup() (which calls Recording.cleanup() and so
        wipes audio_data back to a 60s zero buffer), this preserves the take we
        just loaded so detection actually runs on the real waveform."""
        self.stop_recording()
        self.stop_playback()
        rec = self.recording
        if rec is not None:
            rec.reset_analysis()
            self.guitar_hero.load_user(rec)
        if self.mistake_widget is not None:
            self.mistake_widget.clear()

    def set_active_instrument(self, channel: int):
        """Make `channel` the active instrument: wipe analysis-derived data, re-init
        the algorithms from the (unchanged) Config, and re-render the views."""
        if not self._has_recording():
            return
        self.score_data = self.recording.score_data
        self.score_data.active_instrument = channel
        self.recording.active_instrument = channel
        self._clear_analysis()
        self.recording.update_config(self.recording.config)
        self.refresh_score_viewer()

    def set_show_full(self, show_full: bool):
        """Host-driven (app.py owns the toggle): show the full score (True) or just
        the active instrument's part (False), then re-render the viewer."""
        self.viewer_show_full = show_full
        self.refresh_score_viewer()

    def transpose(self, semitones: int):
        """Transpose this tab's score by `semitones` half steps, then re-render
        the piano-roll + sheet-music views (playback reads the shifted MIDI
        live). Pitch-only: timing and the clip are untouched."""
        if self.score_data is None or self.score_data.score is None:
            return
        self.score_data.transpose(semitones)
        self.guitar_hero.update_view_items()
        self.refresh_score_viewer()

    # --- AUDIO / DETECTION ---
    def refresh_audio(self):
        """A new take's raw audio is loaded: make it playable and (re-)run offline
        pitch detection on it in the background."""
        if not self._has_recording():
            return
        self.audio_player.load_audio(self.recording.audio_data)
        if self.recording.has_pitch_data():
            self.guitar_hero.load_user(self.recording)
            self.mistake_widget.load_mistakes(self.recording.alignment.mistakes)
            self._refresh_guitar_hero_now()
            return
        self.detect_pitches()

    def detect_pitches(self):
        """Clear stale analysis, then (re-)run offline pitch detection on the
        active recording's audio in the background. When detection finishes,
        _on_detection_finished loads the fresh pitch data into view."""
        rec = self.recording
        if rec is None or rec.audio_data.end_index <= 0:
            return
        self._clear_analysis()  # clear stale analysis but KEEP the loaded audio
        rec.pitch_data = PitchData(config=rec.config)
        self.guitar_hero.load_user(rec)
        rec.save_cache()
        self._wire_detector(rec)  # just in case
        rec.pitch_detector.detect_pitches_async()

    def _detection_in_flight(self) -> bool:
        """True while offline pitch detection+smoothing is still running."""
        rec = self.recording
        if rec is None:
            return False
        thread = getattr(rec.pitch_detector, "offline_thread", None)
        return bool(thread and thread.is_alive())

    def _on_detection_finished(self):
        """Offline pitch detection finished (queued onto the main thread): clear
        the status and load the now-ready pitch data into view."""
        sender = self.sender()
        if (self.recording is None
                or (sender is not None and sender is not self.recording.pitch_detector)):
            return
        self.status_bar.update_status("")
        self.guitar_hero.load_user(self.recording)
        self._refresh_guitar_hero_now()
        self.recording.save_cache()

    def _refresh_guitar_hero_now(self):
        """Rebuild visible GuitarHero items and ask Qt to repaint immediately."""
        self.guitar_hero.update_view_items()
        self.guitar_hero.plot.viewport().update()
        self.guitar_hero.update()

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
        # a clipped take/playback always begins at the clip start (bounds[0])
        self.slider.sync_clip_window(self.score_data)
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

    def start_recording(self, start_time: float | None = None):
        """Start recording; called after app.py's count-in.

        `start_time` is where the count-in left the playhead — one beat BEFORE the
        head (the runway), which can be NEGATIVE when recording from the very start.
        We capture that lead-in anyway: a negative time origin keeps the audio/pitch
        buffers 0-indexed, and the clock floor lets the cursor show the runway. The
        take is realigned to the score's clip start at analysis (Recording.resize)."""
        # a clipped take always begins at the clip start (bounds[0])
        self.slider.sync_clip_window(self.score_data)
        t = self.slider.get_time() if start_time is None else start_time
        # record a one-beat runway even into negative app-time: origin (= the
        # earliest, possibly-negative start) keeps the buffers 0-indexed.
        origin = min(0.0, t)
        self.recording.audio_data.t_origin = origin
        self.recording.pitch_data.t_origin = origin
        self.wall_clock.set_floor(origin)
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
        self.wall_clock.set_floor(0.0)  # drop the runway floor for plain playback
        self.audio_recorder.stop()
        self.midi_player.stop()
        self.recording.pitch_detector.stop()
        self.status_bar.update_status("")

    # --- VIEW DRIVING (called by the host's shared clock/slider dispatch) ---
    def move_views(self, t: float):
        self.score_data.update_time(t)
        self.score_viewer.set_playback_time(self._score_viewer_time(t))
        self.guitar_hero.move_plot(t)

    def render_at(self, t: float):
        """Public alias used by the host (e.g. on tab switch) to line this tab's
        views up with a given time."""
        self.move_views(t)

    def on_clock_tick(self, t: float):
        """Shared wall-clock tick: drive the views during playback AND recording.
        Recording drives off the clock (not the slider) so the cursor can show the
        negative pre-head runway — the slider clamps to 0 and would otherwise pin
        the plot there."""
        if not (self.is_playing or self.is_recording):
            return
        self.move_views(t)

    def on_slider_changed(self, t: float):
        """Shared slider moved: move the views only when we're idle (scrubbing).
        During playback/recording the wall clock owns the cursor (see
        on_clock_tick), so ignore the slider's clamped echoes here."""
        if self.is_playing or self.is_recording:
            return
        self.move_views(t)

    # --- ANALYSIS PIPELINE ---
    def analyze(self):
        if not self._has_recording(warn=True) or self._recording_is_empty(warn=True):
            return

        rec = self.recording
        # don't analyze raw/partial pitches: if offline detection+smoothing is
        # still running in the background, the smoothed track (octave errors /
        # noise cleaned up) isn't ready yet. Warn the user to retry once it's
        # finished rather than silently queuing the analysis.
        if self._detection_in_flight():
            QMessageBox.warning(
                self, "Still detecting pitches",
                "Pitch detection is still running. Please wait for it to finish, "
                "then try analyzing again.",
            )
            return
        print("analyzing... ")
        rec.reset_analysis()  # clear stale notes/alignment/mistakes before recomputing

        # Detect notes at the best ND frame size, stretch the score to match the
        # take, then build the final alignment against those resized score notes.
        # Alignment stores score Note objects directly; if we align before the
        # resize, overlays keep pointing at the old note timings.
        rec.note_detector.find_best_w2()
        rec.detect_notes()
        # re-median note pitches over non-transition frames
        rec.detect_transitions()
        rec.recompute_note_pitches()
        rec.prune_transition_notes()

        # resize the score data to the new length
        length = rec.get_length(raw=False)
        rec.resize(new_length=length)

        rec.detect_mistakes()
        rec.mistake_checker.mistake_correction_loop()
        rec.update_alignment_distances() # color the user pitches by the final alignment

        # reload every view with the fresh analysis (note/alignment may have been
        # overwritten by the correction loop)
        self.guitar_hero.load_alignment(rec.alignment)
        self.guitar_hero.load_user(rec)
        self.guitar_hero.update_view_items()
        self.slider.update_range(score_data=self.score_data, recording=rec)
        # a clip-resize re-anchors the clip (and the take) at t=0 — line the view up
        # at the clip start and redraw the dim bands at the new absolute positions.
        b = self.score_data.clip_bounds()
        if b is not None:
            self.slider.set_time(b[0])
        self.guitar_hero.update_clip_overlay()
        self.mistake_widget.load_mistakes(rec.alignment.mistakes)
        rec.save_cache()

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
        """Triggered after a mistake is clicked in MistakeWidget.
        Calls GuitarHero to highlight the respective note(s).
        
        Args:
            idx (int): The index of the selected mistake in the MistakeWidget.
        """
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
        self.recording.save_cache()

    # --- SCORE VIEWER ---
    def _score_viewer_time(self, t: float) -> float:
        """Map a wall-clock time `t` (current tempo) into the Verovio cursor's
        timeframe. First undo any tempo change (-> original-tempo app time), then
        run that through the barline-anchored map so the cursor lands on whatever
        note is actually SOUNDING (the MIDI/NoteData timeline), not on Verovio's
        independently-drifting timemap. Falls back to the plain scalar until the
        map's anchors have been pulled."""
        bpm_og = self.score_data.bpm_og or self.score_data.bpm
        if not bpm_og:
            return t
        # undo the transpose offset (a clip-resize shifts the notes so the clip
        # starts at t=0) THEN the tempo change -> original-tempo app time, the
        # frame measure_onsets_og / the barline map are anchored in.
        og_t = (t - self.score_data.transpose_offset) * self.score_data.bpm / bpm_og
        return self._time_map.to_viewer(og_t)

    def refresh_score_viewer(self):
        """Re-render the Verovio score viewer.
        Reflects any active_instrument/full-score state changes."""
        if self.score_data is None or self.score_data.score is None:
            return
        channel = None if self.viewer_show_full else self.score_data.active_instrument
        self.score_viewer.load_score(self.score_data, channel=channel)
        # rebuild the barline time map for the freshly laid-out score (async pull
        # of Verovio's measure onsets), then re-assert the clip grey-out so it
        # survives the re-layout (and clears itself when the score isn't clipped).
        self._rebuild_time_map(channel)
        self._refresh_clip_focus()

    def _rebuild_time_map(self, channel):
        """Re-anchor the app<->Verovio time map to the freshly rendered score, so
        the cursor tracks the MIDI/NoteData timeline (see ScoreTimeMap). Pairs the
        score's own measure onsets with Verovio's measure timemap, pulled async."""
        app_onsets = self.score_data.measure_onsets_og(channel)

        def _store(vero_onsets):
            if not vero_onsets or not app_onsets:
                self._time_map.clear()  # fall back to the plain bpm/bpm_og scalar
            else:
                self._time_map.set_anchors(app_onsets, vero_onsets)
            # the clip grey-out is placed through the map; the first
            # _refresh_clip_focus ran before the anchors landed, so re-assert it.
            self._refresh_clip_focus()

        self.score_viewer.get_measure_timemap(_store)

    def on_score_viewer_loaded(self, ok: bool = True):
        """The ScoreViewer's JS API is ready: render whatever score is loaded, and
        let the host know (it loads the demo score on first ready)."""
        self.refresh_score_viewer()
        self.viewer_ready.emit()

    # --- CLIP (measure-range focus; stored on ScoreData as note indices) ---
    def apply_clip(self):
        """Clip menu 'Clip': clip to the measures selected in the score viewer
        (async pull; _on_clip_selection applies it)."""
        self.score_viewer.get_clip_selection(self._on_clip_selection)

    def _on_clip_selection(self, sel: dict | None):
        """Turn a pulled measure selection into a note-index clip. `sel` holds
        inclusive measure INDICES (startIdx/endIdx); ScoreData resolves them to
        the exact notes in those measures off its own MIDI timeline, so the clip
        can't drift even where Verovio's rendered timeline runs ahead."""
        if not sel:
            return  # nothing selected -> leave the current clip as-is
        clip = self.score_data.note_index_range_for_measures(
            sel["startIdx"], sel["endIdx"])
        if clip is None:
            return
        self.set_clip(clip, seek=True)
        self.clip_changed.emit(clip)  # mirror onto the other tab (global clip)

    def reset_clip(self):
        """Clip menu 'Reset': drop the clip (mirrored onto the other tab)."""
        self.set_clip(None)
        self.clip_changed.emit(None)

    def set_clip(self, clip, seek: bool = False):
        """Apply `clip` ((i0, i1) note indices, or None) to THIS tab's score and
        refresh its slider window / grey-out / views. Used both to clip/reset here
        and by the host to mirror the clip onto the inactive tab."""
        if clip is None:
            self.score_data.clear_clip()
        else:
            self.score_data.set_clip(*clip)
        self.score_viewer.clear_clip_selection()
        self.slider.update_range(score_data=self.score_data, recording=self.recording)
        if seek:
            b = self.score_data.clip_bounds()
            if b is not None:
                self.slider.set_time(b[0])  # jump the cursor to the clip start
        self._refresh_clip_focus()
        self.move_views(self.slider.get_time())
        if self.recording is not None:
            self.recording.save_cache()

    def sync_clip(self, clip):
        """Mirror a clip made in the OTHER tab onto this score (the clip is global).
        Updates this tab's grey-out + guitar-hero but NOT the shared slider — the
        active tab and the tab-switch handler own the slider window."""
        if clip is None:
            self.score_data.clear_clip()
        else:
            self.score_data.set_clip(*clip)
        self._refresh_clip_focus()
        self.guitar_hero.update_view_items()
        if self.recording is not None:
            self.recording.save_cache()

    def _refresh_clip_focus(self):
        """(Re)assert (or clear) the score-viewer grey-out from the clip. Keyed on
        the clip's measure indices (derived from its notes) so it greys exactly
        the clipped measures regardless of Verovio's timeline drift."""
        mr = self.score_data.clip_measure_range()
        if mr is not None:
            self.score_viewer.set_clip_range(mr[0], mr[1])
        else:
            self.score_viewer.clear_clip_range()

    # --- ERROR HANDLING ---
    # the following warn the user of erroneous inputs on warn=True
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
