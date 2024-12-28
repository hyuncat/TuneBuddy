import numpy as np
import threading
import essentia.standard as es
import pretty_midi

from app.core.midi.MidiData import MidiData
from app.config import AppConfig


class AudioData:
    def __init__(self, MidiData: MidiData=None, audio_filepath: str=None, sample_rate=AppConfig.SAMPLE_RATE):
        """
        Initialize the RecordingData with an array of zeros of length equal to the MIDI file length.
        @param:
            - midi_length: int, length of the MIDI file in samples
        """
        self._MidiData = MidiData
        self.sample_rate = sample_rate

        # Initialize the audio data array with all zeros, with capacity 
        # based on MIDI file length and app's SAMPLE_RATE
        if self._MidiData is not None:
            self.capacity = int(self._MidiData.get_length() * self.sample_rate)
            self.data = np.zeros(self.capacity, dtype=np.float32)
        
        else: # If no MIDI file is provided, use a default length of 60 seconds
            DEFAULT_LENGTH = 60
            self.capacity = int(DEFAULT_LENGTH * self.sample_rate)
            self.data = np.zeros(self.capacity, dtype=np.float32)
            
        if audio_filepath is not None:
            self.load_data(audio_filepath) # (also sets capacity)

        # To ensure thread-safe access to the buffer
        # as AudioRecorder and AudioPlayer will be accessing it
        self.lock = threading.Lock()

    def load_data(self, audio_filepath: str):
        """
        Load audio data into the recording data array.
        Args:
            audio_filepath (str): A correct file path pointing to audio data to load
        """
        # app_directory = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        # audio_file_path = os.path.join(app_directory, 'resources', 'audio', audio_filepath)
        loader = es.MonoLoader(filename=audio_filepath, sampleRate=self.sample_rate)
        audio_data = loader()

        # Set the loaded audio from file as the new self.audio_data and update capacity
        self.data = audio_data
        self.capacity = len(audio_data)

    def load_midi_file(self, midi_filepath: str, soundfont_filepath: str):
        """
        Convert the given MIDI file to np.array of audio signals with a given soundfont;
        uses the AppConfig sample rate.
        Args:
            midi_filepath (str): MIDI file to convert
            soundfont_filepath (str): Soundfont file
        """
        midi_obj = pretty_midi.PrettyMIDI(midi_filepath)
        midi_audio = midi_obj.fluidsynth(fs=self.sample_rate, sf2_path=soundfont_filepath)

        self.data = midi_audio
        self.capacity = len(midi_audio)

    def write_data(self, buffer: np.ndarray, start_time: float=0):
        """
        Add a new audio chunk to the recording data, growing the self.data array if necessary.
        Args:
            buffer (np.ndarray): Temporary buffer of new audio data to be added
            start_time (float), time in seconds to start adding the new chunk
        """
        start_index = int(start_time * self.sample_rate)
        end_index = start_index + len(buffer)

        if end_index > self.capacity:
            # Double the capacity
            self.capacity *= 2
            self.data = np.resize(self.data, self.capacity)

        # Write the new audio data to the data array
        with self.lock:
            self.data[start_index:end_index] = buffer

    def read_data(self, start_time: float=0, end_time: float=0) -> np.ndarray:
        """
        Read audio data from the recording data array.
        Args:
            start_time (float): time in seconds to start reading from
            end_time (float): time in seconds to stop reading
        Returns:
            data (np.ndarray): audio data array from start_time to end_time
        """
        start_index = int(start_time * self.sample_rate)
        end_index = int(end_time * self.sample_rate)

        with self.lock:
            return self.data[start_index:end_index]

    def get_length(self) -> int:
        """
        Get the length of the audio data in seconds
        Returns:
            length (float): length of the audio data in seconds
        """
        return len(self.data) / self.sample_rate