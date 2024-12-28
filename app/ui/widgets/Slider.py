from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtWidgets import QWidget, QSlider, QVBoxLayout

from app.core.midi.MidiData import MidiData

class Slider(QWidget):
    """
    The Slider object is the central hub for all time-based operations 
    regarding the recording / RT-pitch correction process.

    The slider holds references to MidiData to determine x-limits

    The slider has the following functionalities:
        - toggle_play: Toggles between play and pause
            - play: Begins slowly moving the slider to the right
            - pause: Pauses the slider's movement
    
    In the 'playing' state, the slider moves based on QTimer:
        - Interval: 100 ms (Regular interval at which 'timeout' event is called)
        - Timeout: And at each timeout, we move a corresponding interval (1 tick) 
                   to the right.
    """
    slider_changed = pyqtSignal(int)  # Signal to emit when the slider value changes

    def __init__(self, MidiData: MidiData):
        super().__init__()
        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

        # Slider is always initialized specific to a MidiData instance
        self.load_midi(MidiData)

        # Slider variables
        self.TICKS_PER_SEC = 10
        self.TIMER_INTERVAL = int((1/self.TICKS_PER_SEC) * 1000)  # Update +1 tick every 100 ms
        self.current_tick: int = 0 # Current tick position in the slider (10 ticks / sec)

        self.init_slider_ui()
        self.init_timer_ui()

    def load_midi(self, MidiData: MidiData) -> None:
        """Load a MidiData instance into the Slider."""
        self._MidiData = MidiData

    def init_slider_ui(self) -> None:
        """Initialize the UI for the Slider widget."""
        self.slider = QSlider(Qt.Orientation.Horizontal)

        # Initialize the slider with a little more leeway than MIDI length
        midi_length_ticks = int((self._MidiData.get_length()+0.5) * self.TICKS_PER_SEC)
        self.slider.setRange(0, midi_length_ticks)

        # Emit signal when the slider value changes
        self.slider.sliderMoved.connect(self.slider_moved)
        self.slider.valueChanged.connect(self.slider_moved)
        self._layout.addWidget(self.slider)

    def init_timer_ui(self) -> None:
        """Initialize the QTimer for the Slider widget."""
        self.timer = QTimer()
        self.timer.timeout.connect(self.handle_timer_update)

    def slider_moved(self, value: int) -> None:
        """Handle when the user moves the slider, or when value changes from autoplay."""
        self.current_tick = value
        self.slider_changed.emit(value)

    def handle_timer_update(self) -> None:
        """Handle the timer update event."""
        self.current_tick += 1
        # Ensure the current tick does not exceed the maximum
        if self.current_tick > self.slider.maximum():
            self.current_tick = self.slider.maximum()
            self.timer.stop()
        self.slider.setValue(self.current_tick)

    def toggle_play(self) -> None:
        """Toggle between playing and pausing the slider movement."""
        if self.timer.isActive():
            self.timer.stop()
        else:
            self.timer.start(self.TIMER_INTERVAL)

    def start_timer(self) -> None:
        """Start the timer."""
        self.timer.start(self.TIMER_INTERVAL)
    
    def stop_timer(self) -> None:
        """Stop the timer."""
        self.timer.stop()

    def get_current_time(self) -> float:
        """Get the current time in seconds from the slider."""
        return self.current_tick / self.TICKS_PER_SEC

    def update_slider_max(self, new_length: int) -> None:
        """Update the slider's maximum value (based on AudioData length)."""
        self.slider.setRange(0, new_length)