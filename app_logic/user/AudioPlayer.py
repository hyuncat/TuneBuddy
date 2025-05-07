import sounddevice as sd
import logging
import threading
from app_logic.user.ds.AudioData import AudioData

class AudioPlayer:
    """a basic audio player, which can load an audio_data, play it from a given 
    start_time, pause playback, then resume where it started

    (+) runs on a thread for pyqt parallelism
    """
    def __init__(self, audio_data: AudioData=None):
        self.audio_data = audio_data

        # threading variables
        self.playback_thread: threading.Thread = None
        self.thread_stop_event = threading.Event()

        # other playback variables
        self.current_time = 0
        self.is_playing = False


    def load_audio(self, audio: str | AudioData):
        if isinstance(audio, str):
            self.audio_data = AudioData(audio)
        elif isinstance(audio, AudioData):
            self.audio_data = audio
        else:
            logging.error("audio was unable to be loaded")
            return

    def play(self, start_time: float=0):
        """play the audio starting from a specified time
        handles threading logic and calls internal function _play()
        
        Args:
            start_time: time (sec) to start playing from
        """
        if self.audio_data is None:
            logging.error("no audio data loaded. exiting")
            return

        # if playback thread already exists
        # eg, seeking while it's still playing
        if self.playback_thread is not None and self.playback_thread.is_alive():
            self.thread_stop_event.set()
            self.playback_thread.join()

        # clear stop event (eg, no longer triggers event during playback)
        self.thread_stop_event.clear()
        self.playback_thread = threading.Thread(target=self._play, args=(start_time,))
        self.playback_thread.start()

    def _play(self, start_time: float=0):
        audio_array = self.audio_data.read_data(start_time, self.audio_data.get_length())

        # basic error checking
        if len(audio_array) == 0:
            logging.error("no audio data available for playback")
            return
        
        sd.play(audio_array, self.audio_data.sr)
        sd.wait()

    def pause(self):
        sd.stop()
        if self.playback_thread is not None and self.playback_thread.is_alive():
            self.playback_thread.join()