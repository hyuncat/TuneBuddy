from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QFileDialog, QToolBar

class Toolbar(QToolBar):
    """
    Supports the following actions:
        1, uploading midi
        2. uploading audio
        3. adjusting recording settings
    """
    midi_uploaded = pyqtSignal(str)
    audio_uploaded = pyqtSignal(str)

    def __init__(self):
        super().__init__() # init the QToolBar it inherits from
        self.setOrientation(Qt.Orientation.Horizontal)

        self.buttons = {}

        self.buttons["Upload MIDI"] = QAction("Upload MIDI", self)
        self.buttons["Upload MIDI"].setStatusTip("Upload a MIDI or XML file")
        self.buttons["Upload MIDI"].triggered.connect(self.upload_midi)

        self.buttons["Upload audio"] = QAction("Upload audio", self)
        self.buttons["Upload audio"].setStatusTip("Upload an audio file")
        self.buttons["Upload audio"].triggered.connect(self.upload_audio)

        for b in self.buttons.values():
            self.addAction(b)
    
    def upload_midi(self):
        print("uploading midi")
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select MIDI or XML File", "", "All Files (*)"

        )
        if file_path:
            print(f"Selected MIDI/XML file: {file_path}")
            self.midi_uploaded.emit(file_path)

    def upload_audio(self):
        print("uploading audio")
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "", "All Files (*)"
        )
        if file_path:
            print(f"Selected audio file: {file_path}")
            self.audio_uploaded.emit(file_path)

