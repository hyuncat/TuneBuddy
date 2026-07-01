from PyQt6.QtCore import pyqtSignal, QTimer, Qt
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSlider
from math import ceil, floor

from ui.time.WallClock import WallClock

class Slider(QWidget):

    slider_changed = pyqtSignal(float) # emits current time in seconds
    slider_end = pyqtSignal(bool)

    def __init__(self, wall_clock: WallClock):
        super().__init__()
        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

        # slider <==> timer resolution variables
        self.wall_clock = wall_clock
        self.wall_clock.time_changed.connect(self.handle_timer_update)

        # init our slider!!
        self.DEFAULT_LENGTH_SEC = 30
        self.TICKS_PER_SEC = self.wall_clock.hz # 10 ticks per sec
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.midi_length_ticks = int(self.DEFAULT_LENGTH_SEC*self.TICKS_PER_SEC)
        self.midi_length_sec = self.DEFAULT_LENGTH_SEC
        self.slider.setRange(0, self.midi_length_ticks)

        # Clip window: when a clip is active the slider keeps its FULL range (the
        # whole piece stays visible) but the cursor is constrained to this
        # [b0, b1] sub-range so it can't escape the clip. Derived from the active
        # tab's score_data.get_bounds(respect_clip=True) on every update_range
        # (None = no constraint).
        # `_clamping` guards the setValue -> valueChanged -> slider_moved re-entry.
        self.clip_window: tuple[float, float] | None = None
        self._clamping = False

        # slider emissions
        # self.slider.sliderMoved.connect(self.slider_moved)
        self.slider.valueChanged.connect(self.slider_moved)

        self._layout.addWidget(self.slider)
        
    # --- SIGNAL RELATED ---
    def slider_moved(self, value: int) -> None:
        """is called whenever the slider moves. emits the slider_changed signal
        corresponding to what time in the plot it now is at.

        Args:
            value (int): the current tick value of the slider
        """
        if self._clamping:
            return  # re-entrant call from our own setValue (clip snap-back)

        t = value / self.TICKS_PER_SEC # convert to seconds
        # constrain to the clip window: a scrub past either edge snaps back in.
        if self.clip_window is not None:
            b0, b1 = self.clip_window
            t = min(max(t, b0), b1)
        tick = int(round(t * self.TICKS_PER_SEC))
        if tick != value:
            self._clamping = True
            self.slider.setValue(tick)
            self._clamping = False

        self.current_tick = tick
        self.slider_changed.emit(t)
        # print(f"Slider moved to {t} sec")

        if tick >= self._max_tick(): # emit signal when reached end of (clipped) range
            self.slider_end.emit(True)

    def _max_tick(self) -> int:
        """The furthest tick the cursor may reach: the clip window's end when a
        clip is active, otherwise the full slider length."""
        if self.clip_window is not None:
            b1_tick = int(round(self.clip_window[1] * self.TICKS_PER_SEC))
            return min(b1_tick, self.midi_length_ticks)
        return self.midi_length_ticks

    # --- RANGE HANDLING ---
    def update_range(self, score_data=None, recording=None):
        """Update the slider range based on max(MIDI.length, audio.length).

        The left edge sits one slider click (1 / hz) before the earlier of the
        score's first note (S) and the user's first played note (U), so neither
        gets clipped at the very edge of the comparison. This slider is the master
        timeline driving the ScoreViewer cursor and GuitarHero, so the lead-in
        lands there. resize_score keeps the take fixed and moves the score onto
        it, so the take's first voiced note (U) sits at its own recorded app-time;
        a Perform runway recorded before it stays left of this edge."""
        m0, m1 = 0, 0
        user_audio_bounds = None
        if score_data:
            b = score_data.get_bounds(respect_clip=False)
            if b:
                m0, m1 = b
        if recording and recording.audio_data:
            user_audio_bounds = recording.audio_bounds()

        # one slider click before min(S, U), clamped to 0 (can't seek negative)
        one_click = 1.0 / self.TICKS_PER_SEC
        first_notes = [m0]
        u0 = self._user_first_note_start(recording)
        if u0 is not None:
            first_notes.append(u0)
        x0 = max(0.0, min(first_notes) - one_click)

        end_times = [m1]
        if user_audio_bounds is not None:
            end_times.append(user_audio_bounds[1])
        x1 = max(end_times)
        self._update_range(x0, x1)

        # (Re)derive the clip window from the ACTIVE tab's bounds. Because the
        # host re-calls update_range on every tab switch, a tab with no clip
        # (bounds == full score) clears the window here — so a Perform clip never
        # carries over to the shared slider on the Practice tab.
        self.clip_window = self._derive_clip_window(score_data)

    def sync_clip_window(self, score_data) -> None:
        """Re-derive the clip window from the current bounds (so it can't go stale)
        and snap the cursor to the clip start if it's currently outside the clip.
        Called right before playback/recording starts so a clipped take always
        begins at the clip's start time `b0`."""
        self.clip_window = self._derive_clip_window(score_data)
        if self.clip_window is not None:
            b0, b1 = self.clip_window
            raw = self.slider.value() / self.TICKS_PER_SEC
            if raw < b0 - 1e-9 or raw > b1 + 1e-9:
                self.set_time(b0)

    @staticmethod
    def _derive_clip_window(score_data) -> tuple[float, float] | None:
        """The clip's [b0, b1] time window (derived from the score's note indices),
        or None when unclipped. Single source of truth:
        ScoreData.get_bounds(respect_clip=True)."""
        if score_data is None or not score_data.is_clipped():
            return None
        return score_data.get_bounds(respect_clip=True)

    @staticmethod
    def _user_first_note_start(recording) -> float | None:
        """Start time of the recording's first voiced note (the same note resize
        aligns the score to), or None if there isn't one yet (no recording / no
        notes detected)."""
        if recording is None:
            return None
        bounds = recording.note_data.get_bounds(clean=True)
        return bounds[0] if bounds is not None else None

    def _update_range(self, start_time: float, end_time: float):
        """update the slider range to have [sec] amount of space"""
        start_ticks = int(floor(start_time * self.TICKS_PER_SEC))
        end_ticks = int(ceil(end_time * self.TICKS_PER_SEC))
        self.midi_length_ticks = end_ticks
        self.midi_length_sec = end_time
        self.slider.setRange(start_ticks, end_ticks)
    
    # --- TIMER RELATED ---
    def handle_timer_update(self, t: float) -> None:
        """called whenever timer is updated (every 100ms)"""
        tick = int(t * self.wall_clock.hz)
        self.current_tick = tick

        # ensure current tick never exceeds the (possibly clipped) maximum; on a
        # clip, hitting b1 fires slider_end so playback stops at the clip end.
        if self.current_tick > self._max_tick():
            self.current_tick = self._max_tick()
            self.slider_end.emit(True)
            # self.wall_clock.stop() # this may not always be good

        self.slider.setValue(self.current_tick)

    # utils
    def set_time(self, t: float) -> None:
        """Move the slider (and cursor) to `t` seconds, clamped to the clip
        window if one is active. Used to jump the cursor to the clip start."""
        if self.clip_window is not None:
            b0, b1 = self.clip_window
            t = min(max(t, b0), b1)
        self.slider.setValue(int(round(t * self.TICKS_PER_SEC)))

    def get_time(self):
        """get current time of slider in seconds (clamped to the clip window)"""
        t = self.slider.value() / self.wall_clock.hz
        if self.clip_window is not None:
            b0, b1 = self.clip_window
            t = min(max(t, b0), b1)
        return t
    
    def get_total_time(self):
        """get total time of slider in seconds"""
        return self.slider.maximum() / self.TICKS_PER_SEC
