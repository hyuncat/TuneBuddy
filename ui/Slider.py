from PyQt6.QtCore import pyqtSignal, QTimer, Qt
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSlider
from math import ceil

class Slider(QWidget):

    slider_changed = pyqtSignal(int)
    slider_end = pyqtSignal(bool)
    
    def __init__(self):
        super().__init__()
        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

        # slider <==> timer resolution variables
        self.TICKS_PER_SEC = 10
        self.N_TICKS_PER_CALLBACK = 1

        # current time variables
        self.current_tick = 0
        self.timer = QTimer()
        self.timer.setInterval(100) # called every 100 ms
        self.timer.timeout.connect(self.handle_timer_update)

        # init our slider!!
        self.DEFAULT_LENGTH_SEC = 30
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.midi_length_ticks = int(self.DEFAULT_LENGTH_SEC*self.TICKS_PER_SEC)
        self.slider.setRange(0, self.midi_length_ticks)

        # slider emissions
        self.slider.sliderMoved.connect(self.slider_moved)
        self.slider.valueChanged.connect(self.slider_moved)

        self._layout.addWidget(self.slider)
        

    def update_slider_range(self, sec: float):
        """update the slider range to have [sec] amount of space"""
        ticks = int(ceil(sec * self.TICKS_PER_SEC))
        self.slider.setRange(0, ticks)

    def get_time(self):
        """get current time of slider in seconds"""
        return self.slider.value() / self.TICKS_PER_SEC
    
    def slider_moved(self, value: int) -> None:
        """is called whenever the slider moves"""
        self.current_tick = value
        self.slider_changed.emit(value)
        
        if value >= self.midi_length_ticks:
            self.slider_end.emit(True)

    def handle_timer_update(self) -> None:
        """called whenever timer is updated (every 100ms)"""
        self.current_tick += 1
        # ensure current tick never exceeds maximum
        if self.current_tick > self.slider.maximum():
            self.current_tick = self.slider.maximum()
            self.timer.stop()
        self.slider.setValue(self.current_tick)
    
    def start_timer(self) -> None:
        """starts the timer, leading to the slider_changed signal to be emitted"""
        interval = int(1000 / self.TICKS_PER_SEC) # should be like 100ms lol
        self.timer.start(interval)

    def stop_timer(self) -> None:
        """stop the timer, causing slider_changed signal to stop emitting"""
        self.timer.stop()
