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
    show_settings = pyqtSignal(bool)

    def __init__(self):
        super().__init__() # init the QToolBar it inherits from
        self.setOrientation(Qt.Orientation.Horizontal)

        self.buttons = {}

        self.buttons["Upload score"] = QAction("Upload Score", self)
        self.buttons["Upload score"].setStatusTip("Upload a .mid or .musicxml file")
        self.buttons["Upload score"].triggered.connect(self.upload_midi)

        self.buttons["Upload audio"] = QAction("Upload Audio", self)
        self.buttons["Upload audio"].setStatusTip("Upload an audio file")
        self.buttons["Upload audio"].triggered.connect(self.upload_audio)

        self.buttons["Settings"] = QAction("Settings", self)
        self.buttons["Settings"].setStatusTip("Set recording / playback settings")
        self.buttons["Settings"].triggered.connect(self.trigger_settings)

        for b in self.buttons.values():
            self.addAction(b)
    
    def upload_midi(self):
        """Open a file dialog to upload a MIDI or musicXML file.
        Emit the midi_uploaded signal with the file path.
        """
        print("Uploading score...")
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select .mid or .musicxml File", "", 
            "All Files (*)"
        )
        if file_path:
            print(f"Selected MIDI/musicXML file: {file_path}")
            self.midi_uploaded.emit(file_path)

    def upload_audio(self):
        """Open a file dialog to upload an audio file.
        Emit the audio_uploaded signal with the file path.
        """
        print("Uploading audio...")
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "", 
            "All Files (*)"
        )
        if file_path:
            print(f"Selected audio file: {file_path}")
            self.audio_uploaded.emit(file_path)
    
    def trigger_settings(self):
        """Open the settings dialog."""
        self.show_settings.emit(True)