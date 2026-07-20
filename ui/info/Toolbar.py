from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction, QFontMetrics
from PyQt6.QtWidgets import (
    QCheckBox, QFileDialog, QHBoxLayout, QLabel, QMenu, QSizePolicy,
    QSpinBox, QToolBar, QToolButton, QWidget, QWidgetAction
)
from ui.info.ToggleSwitch import ToggleSwitch
from app_logic.midi.ScoreData import ScoreData

from resources.program_map import program_to_name


class Toolbar(QToolBar):
    """
    Supports the following actions:
        1, uploading score (midi or musicxml)
        2. uploading audio
        3. adjusting recording settings
    """
    score_uploaded = pyqtSignal(str)
    audio_uploaded = pyqtSignal(str)
    folder_uploaded = pyqtSignal(str)
    save_recording_requested = pyqtSignal()
    show_settings = pyqtSignal(bool)
    select_measures_requested = pyqtSignal() # arm measure selection in the score viewer
    clip_reset = pyqtSignal()                # restore the full score
    user_audio_toggled = pyqtSignal(bool) # value = user audio on/off
    tempo_changed = pyqtSignal(int) # value = new tempo in BPM

    def __init__(self, score_data: ScoreData):
        super().__init__() # init the QToolBar it inherits from
        self.setOrientation(Qt.Orientation.Horizontal)
        self.instrument_checkboxes: dict[int, QCheckBox] = {}

        # important references
        self.score_data = score_data
        self.init_ui()

    def init_upload_widget(self):
        """Make a button widget that has a dropdown menu
        for uploading either a score or a recording file."""
        # --- UPLOAD BUTTON + MENU ---
        upload_button = QToolButton(self)
        upload_button.setText("Upload")

        upload_menu = QMenu(self)
        upload_score_action = QAction("Score", self)
        upload_score_action.setStatusTip("Upload a .mid or .musicxml file")
        upload_score_action.triggered.connect(self.upload_score)

        upload_recording_action = QAction("Recording", self)
        upload_recording_action.setStatusTip("Upload an audio file")
        upload_recording_action.triggered.connect(self.upload_audio)

        upload_folder_action = QAction("Folder", self)
        upload_folder_action.setStatusTip("Upload an Attune folder")
        upload_folder_action.triggered.connect(self.upload_folder)

        upload_menu.addAction(upload_score_action)
        upload_menu.addAction(upload_recording_action)
        upload_menu.addAction(upload_folder_action)

        upload_button.setMenu(upload_menu)
        upload_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        upload_button.setStyleSheet("""
            QToolButton::menu-indicator {
                image: none;
                width: 0px;
            }
        """)
        self.addWidget(upload_button)

    def init_clip_widget(self):
        """A 'Clip' button whose dropdown holds two items:
          - 'Select measures': arm measure picking in the score viewer (click a
            start + end measure to highlight the range, then right-click to clip).
          - 'Reset': restores the full score.
        """
        self.clip_button = QToolButton(self)
        self.clip_button.setText("Clip")

        self.clip_menu = QMenu(self)
        self.clip_menu.setToolTipsVisible(True)
        self.select_measures_action = QAction("Select measures", self)
        tip = ("Select the first and last measures of your desired range, "
               "then right-click to clip.")
        self.select_measures_action.setToolTip(tip)
        self.select_measures_action.setStatusTip(tip)
        self.select_measures_action.triggered.connect(self.select_measures_requested.emit)
        self.clip_menu.addAction(self.select_measures_action)

        self.reset_clip_action = QAction("Reset", self)
        self.reset_clip_action.setStatusTip("Restore the full score (undo the clip)")
        self.reset_clip_action.triggered.connect(self.clip_reset.emit)
        self.clip_menu.addAction(self.reset_clip_action)

        self.clip_button.setMenu(self.clip_menu)
        self.clip_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.clip_button.setStyleSheet("""
            QToolButton::menu-indicator {
                image: none;
                width: 0px;
            }
        """)
        self.addWidget(self.clip_button)

    def init_instrument_select(self):
        # --- instrument multi-select dropdown ---
        self.instrument_menu = QMenu(self)
        # create + style the button that will trigger the dropdown
        self.instrument_button = QToolButton(self)
        self.instrument_button.setText("Playback")
        self.instrument_button.setMenu(self.instrument_menu)
        self.instrument_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.instrument_button.setStyleSheet("""
            QToolButton::menu-indicator {
                image: none;
                width: 2px;
            }
        """)
        self.populate_instrument_menu() # populate with current score data
        self.addWidget(self.instrument_button)

    def init_tempo_spinbox(self):
        """Make a spinbox for adjusting tempo, with a label and proper formatting."""
        # --- tempo spinbox ---
        self.tempo_spinbox = QSpinBox()
        self.tempo_spinbox.setRange(20, 400)
        self.tempo_spinbox.setValue(120)
        self.tempo_spinbox.setSuffix(" BPM")

        def get_spinbox_width(spinbox: QSpinBox) -> int:
            """helper to get width based on '400 BPM' being the longest text"""
            fm = QFontMetrics(spinbox.font())
            sample_text = "400 BPM"
            text_width = fm.horizontalAdvance(sample_text)
            return text_width + 40 # padding for frame + arrows
        
        w = get_spinbox_width(self.tempo_spinbox)
        self.tempo_spinbox.setMinimumWidth(w)
        self.tempo_spinbox.valueChanged.connect(self.tempo_changed.emit)
        self.addWidget(self.tempo_spinbox)

    def set_tempo(self, bpm: float):
        """Reflect `bpm` in the spinbox WITHOUT emitting tempo_changed (used to
        display the active tab's score tempo, e.g. the new length after Analyze)."""
        self.tempo_spinbox.blockSignals(True)
        self.tempo_spinbox.setValue(int(round(bpm)))
        self.tempo_spinbox.blockSignals(False)


    def init_ui(self):
        # --- UPLOAD BUTTON + MENU ---
        self.init_upload_widget()

        # other buttons (edit later)
        settings = QAction("Settings", self)
        settings.setStatusTip("Set recording / playback settings")
        settings.triggered.connect(self.trigger_settings)
        self.addAction(settings)

        save_recording = QAction("Save", self)
        save_recording.setStatusTip("Save the active recording to the score folder")
        save_recording.triggered.connect(self.save_recording_requested.emit)
        self.addAction(save_recording)

        self.init_clip_widget()

        self.addSeparator()
        self.init_instrument_select()

        # --- RIGHT SIDE: TIMEKEEPING ---
        # stretch so right-side controls don’t crowd left actions
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.addWidget(spacer)

        self.init_tempo_spinbox()

        # --- metronome toggle ---
        self.metronome_label = QLabel("Metronome")
        self.addWidget(self.metronome_label)
        self.metronome_switch = ToggleSwitch("Metronome")
        self.metronome_switch.toggled.connect(self.metronome_toggled)
        self.metronome_switch.setChecked(True) # default to metronome on
        self.addWidget(self.metronome_switch)

    def init_signals(self):
        """Initialize signal connections for the toolbar.
            - score / audio uploads
            - settings / clip dialog triggers
            - instrument select changes
            - tempo changes
            - metronome toggling
        """
        pass
        # self.tempo_spinbox.valueChanged.connect(self.tempo_changed.emit)
        # self.metronome_switch.toggled.connect(self.metronome_toggled.emit)
    
    def upload_score(self):
        """Open a file dialog to upload a MIDI or musicXML file.
        Emit the score_uploaded signal with the file path.
        """
        print("Uploading score...")
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select .mid or .musicxml File", "", 
            "All Files (*)"
        )
        if file_path:
            print(f"Selected MIDI/musicXML file: {file_path}")
            self.score_uploaded.emit(file_path)

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

    def upload_folder(self):
        """Open a folder dialog and emit the selected Attune folder path."""
        print("Uploading folder...")
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Attune Folder", ""
        )
        if folder_path:
            print(f"Selected folder: {folder_path}")
            self.folder_uploaded.emit(folder_path)
    
    def trigger_settings(self):
        """Open the settings dialog."""
        self.show_settings.emit(True)

    def populate_instrument_menu(self):
        """Dynamically populate the instrument multi-select dropdown
        based on the instruments present in the loaded score."""
        # clear existing menu items
        self.instrument_menu.clear()
        self.instrument_checkboxes: dict[int, QCheckBox] = {}

        # ---> add a 'user' button
        user_checkbox = QCheckBox("User")
        user_checkbox.setChecked(True) # default to user audio playback on
        user_checkbox.toggled.connect(self.user_audio_toggled.emit)
        # format container
        user_container = QWidget()
        user_layout = QHBoxLayout(user_container)
        user_layout.setContentsMargins(2, 2, 10, 2)
        user_layout.addWidget(user_checkbox)
        user_layout.addStretch()

        user_action = QWidgetAction(self)
        user_action.setDefaultWidget(user_container)
        self.instrument_menu.addAction(user_action)

        self.instrument_menu.addSeparator() # divider

        if len(self.score_data.instruments) == 0:
            # print("oops no instruments")
            return # no score loaded, or score has no instruments

        # a freshly loaded score resets playing_instruments to ALL channels
        # (metronome included); re-apply the switch so the toolbar toggle —
        # not the load default — decides whether the metronome plays. (Guarded
        # because this also runs during init, before the switch is built.)
        if hasattr(self, "metronome_switch"):
            self.metronome_toggled(self.metronome_switch.isChecked())

        for channel, program in self.score_data.instruments.items():
            instr_name = f"{program_to_name(program)}"
            if channel == self.score_data.metronome_channel:
                continue # don't show metronome sound
            checkbox = QCheckBox(instr_name)
            checkbox.setChecked(True) # default to all instruments selected
            checkbox.toggled.connect(self.on_instrument_selection_changed)
            self.instrument_checkboxes[channel] = checkbox

            # format its container nicely
            container = QWidget()
            layout = QHBoxLayout(container)
            layout.setContentsMargins(2, 2, 10, 2)  # left, top, right, bottom
            layout.addWidget(checkbox)
            layout.addStretch()

            action = QWidgetAction(self)
            action.setDefaultWidget(container)
            self.instrument_menu.addAction(action)

    def on_instrument_selection_changed(self, checked: bool):
        """Handle changes in instrument selection."""
        selected_channels = {
            ch for ch, cb in self.instrument_checkboxes.items()
            if cb.isChecked()
        }
        # the metronome isn't one of our checkboxes; preserve its current state
        # (it's toggled independently by the metronome switch).
        if self.score_data.metronome_channel in self.score_data.playing_instruments:
            selected_channels.add(self.score_data.metronome_channel)
        self.score_data.playing_instruments = selected_channels
        print(f"playing instruments: {self.score_data.playing_instruments}")

    def metronome_toggled(self, checked: bool):
        """Handle metronome toggling. The switch is the source of truth for
        whether the metronome plays — re-applied on every score load (see
        populate_instrument_menu) so a freshly imported score respects the
        toggle instead of defaulting on."""
        channel = self.score_data.metronome_channel
        if channel is None:
            return  # no metronome track on this score (no free MIDI channel)
        if checked:
            self.score_data.playing_instruments.add(channel)
        else:
            self.score_data.playing_instruments.discard(channel)
        print(f"Metronome toggled: {'ON' if checked else 'OFF'}")
