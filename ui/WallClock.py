# wall_clock.py
from PyQt6.QtCore import QObject, pyqtSignal, QTimer
import time

class WallClock(QObject):

    time_changed = pyqtSignal(float) # emits current time in seconds

    def __init__(self, hz=10):
        super().__init__()
        self.hz = hz # updates per second
        self.interval = int(1.0 / hz * 1000) # 100 ms

        self.timer = QTimer()
        self.timer.setInterval(self.interval) # called every 100 ms

        self.timer.timeout.connect(self._timeout)
        self.current_tick = 0

        # keep track of a continuous time
        self._running = False
        self._base_media = 0.0
        self.base_wall = time.perf_counter()


    def start(self, t: float=0.0):
        """Start the wall clock"""
        self.seek(t)                     # sets base_media and base_wall
        self._running = True
        self.timer.start()

    def seek(self, t: float):
        """Seek the current_tick to the given time in seconds
        Args:
            t (float): time in seconds
        """
        self._base_media = float(t)
        self._base_wall = time.perf_counter()

    def _timeout(self):
        """Called every interval (eg, 100ms) by the QTimer"""
        self.time_changed.emit(self.now())

    def pause(self):
        """Pause the wall clock"""
        if self._running:
            # freeze media time
            self._base_media = self.now()
            self._running = False
        self.timer.stop()

    def stop(self):
        """Stop the wall clock and reset the tick counter to 0"""
        self.pause()
        self._base_media = 0.0

    def is_running(self) -> bool:
        """Check if the wall clock is running"""
        return self._running

    def now(self) -> float:
        """Get the current time in seconds"""
        if not self._running:
            return self._base_media
        return self._base_media + (time.perf_counter() - self._base_wall)