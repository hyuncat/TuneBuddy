import time

from PyQt6.QtCore import QObject, Qt, pyqtSignal, QTimer

class WallClock(QObject):

    time_changed = pyqtSignal(float) # emits current time in seconds

    def __init__(self, hz=10):
        super().__init__()
        self.hz = hz # updates per second
        self.interval = max(1, int(round(1.0 / hz * 1000))) # 100 ms at 10 Hz

        self.timer = QTimer()
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.setInterval(self.interval) # called every 100 ms at 10 Hz

        self.timer.timeout.connect(self._timeout)
        self.current_tick = 0

        # Keep media time anchored to monotonic wall time. `hz` controls how often
        # the UI is notified, not how quickly time advances.
        self._running = False
        self.stall = False
        self._media_anchor = 0.0
        self._wall_anchor = time.monotonic()
        self._stalling = False


    def start(self, t: float=0.0):
        """Start the wall clock"""
        self.seek(t)
        self._running = True
        self.timer.start()

    def seek(self, t: float):
        """Seek the current_tick to the given time in seconds
        Args:
            t (float): time in seconds
        """
        self._media_anchor = float(t)
        self._wall_anchor = time.monotonic()
        self._stalling = False
        self._set_tick(t)

    def _timeout(self):
        """Called every interval by the QTimer."""
        now = time.monotonic()
        if self.stall:
            if not self._stalling:
                self._media_anchor = self._time_at(now)
                self._set_tick(self._media_anchor)
                self._stalling = True
            self._wall_anchor = now
        else:
            if self._stalling:
                self._wall_anchor = now
                self._stalling = False
            self._set_tick(self._time_at(now))
        self.time_changed.emit(self.now())

    def pause(self):
        """Pause the wall clock"""
        if self._running:
            self._media_anchor = self._time_at(time.monotonic())
            self._set_tick(self._media_anchor)
            self._running = False
        self.timer.stop()

    def stop(self):
        """Stop the wall clock and reset the tick counter to 0"""
        self.pause()
        self.current_tick = 0

    def toggle(self, t: float=0.0):
        """Toggle the wall clock between running and paused states"""
        if self._running:
            self.pause()
        else:
            self.start(t)

    def is_running(self) -> bool:
        """Check if the wall clock is running"""
        return self._running

    def now(self) -> float:
        """Get the current time in seconds"""
        return round(self.current_tick / self.hz, 5)

    def _time_at(self, wall_time: float) -> float:
        if not self._running:
            return self._media_anchor
        return self._media_anchor + max(0.0, wall_time - self._wall_anchor)

    def _set_tick(self, t: float):
        self.current_tick = max(0, int(round(t * self.hz)))
