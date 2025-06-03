import numpy as np

from app_logic.user.ds.AudioData import AudioData
from app_logic.user.ds.PitchData import PitchData
from app_logic.user.ds.Buffer import Buffer

class UserData:
    def __init__(self, pitch_detector):
        """the user data"""
        # essential data variables
        self.audio_data = AudioData()
        self.pitch_data = PitchData(pitch_detector)
        self.note_data = None

        # queue data structures for real time pitch + note detection + correction
        self.a2p_queue = Buffer() #audio-to-pitches
        self.p2n_queue = Buffer() #pitches-to-notes
        self.n2c_queue = None #notes-to-corrections

        # algorithms
        self.pitch_detector = pitch_detector

    def load_audio(self, audio_filepath: str):
        """load in a pre-recorded audio file from a filepath
        also computes pitches on the entire file"""
        self.audio_data.load_data(audio_filepath)
        self.pitch_data = self.pitch_detector.detect_pitches(self.audio_data.data)

    def write_data(self, indata: np.ndarray, start_time: float):
        """write indata to the audio_data at the given start_time
        and append to our queue for pitch processing
        """
        self.audio_data.write_data(indata, start_time)
        self.a2p_queue.push(indata)