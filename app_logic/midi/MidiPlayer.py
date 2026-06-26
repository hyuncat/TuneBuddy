import threading
import time
from typing import Union
# import logging
from PyQt6.QtCore import QObject

from app_logic.midi.ScoreData import ScoreData
from app_logic.midi.MidiSynth import MidiSynth
from ui.time import WallClock

class MidiPlayer(QObject):
    """object to play a score object using a midi synthesizer"""

    def __init__(self, midi_synth: MidiSynth, wall_clock: WallClock):
        super().__init__()
        self.midi_synth = midi_synth
        self.score_data: ScoreData = None
        self.wall_clock = wall_clock

        # threading variables
        self.playback_thread: threading.Thread = None
        self.thread_stop_event = threading.Event()

        # other playback variables
        self.current_time = 0
        self.playback_speed = 1.0

    def load_score(self, score: Union[str, ScoreData]):
        """
        Load a score from a file path (str) or a ScoreData object
        
        Args:
            score: Union[str, ScoreData], path to score file or ScoreData object
        """
        if isinstance(score, str):
            self.score_data = ScoreData(score)
        elif isinstance(score, ScoreData):
            self.score_data = score
        else:
            raise ValueError("score source must be either a file path (str) or a ScoreData object")

    def play(self, start_time: float=0):
        """play a MIDI file using the synthesizer from a particular time
        handles threading logic and calls internal function _play()

        Args:
            start_time: time (sec) to start playing from
        """
        # exit if no score data loaded
        if self.score_data is None:
            print("Ignoring MIDI playback, no score data loaded.")
            return
        
        # if playback thread already exists
        # eg, seeking while it's still playing

        # STOP existing playback thread
        if self.playback_thread is not None and self.playback_thread.is_alive():
            self.thread_stop_event.set()
            self.midi_synth.pause()
            self.playback_thread.join()

        # PLAY from new start_time
        # clear stop event (eg, no longer triggers event during playback)
        self.thread_stop_event.clear()
        # self.wall_clock.start(t=start_time)
        self.playback_thread = threading.Thread(target=self._play, args=(start_time,))
        self.playback_thread.start()

    def _play(self, start_time: float=0):
        """
        internal function called by self.playback_thread to handle playback from any
        time in the score. uses binary search for faster indexing based on start time
        """
        # even if start_time =/= 0, ensure all programs are initialized
        midi_data = self.score_data.midi_data
        for _, program_change_messages in midi_data.programs.items():
            for msg in program_change_messages:
                self.midi_synth.handle_midi(msg)
        
        msg_times = list(midi_data.messages.keys())
        # find index to start from based on start_time
        _, start_idx = MidiPlayer.binary_search(msg_times, start_time)

        # Pace playback off a high-resolution monotonic clock, NOT WallClock.now().
        # WallClock ticks at 10 Hz, so now() is quantized to 0.1 s; that snaps every
        # MIDI message onto a 0.1 s grid. With 16th notes at 80 BPM (0.1875 s apart)
        # the grids beat against each other and every ~8th note's onset lands exactly
        # on a tick, cutting the previous note a tick short. The WallClock still drives
        # the UI cursor; only audio timing needs sub-tick precision.
        speed = self.playback_speed or 1.0
        # anchor perf_counter so that "now" reads start_time in score-time
        play_anchor = time.perf_counter() - start_time / speed

        for i in range(start_idx, len(msg_times)):
            # stop handling : if stop event is set, break look and exit playback
            if self.thread_stop_event.is_set():
                # self.midi_synth.pause()
                break

            # iterate through the messages and play them
            current_time = msg_times[i]
            messages = midi_data.messages[current_time]

            for msg in messages:
                if msg.is_meta:
                    self.midi_synth.handle_midi(msg)
                    continue
                channel = getattr(msg, "channel", None)
                enabled = channel in self.score_data.playing_instruments
                # only sound enabled channels, but ALWAYS let note_off through so
                # toggling a channel (e.g. the metronome) off mid-playback never
                # leaves a hanging note. A note_off on a silent note is a no-op.
                if enabled or msg.type == "note_off":
                    self.midi_synth.handle_midi(msg)
                
            # compute time to sleep until next msg
            if i+1 <= len(msg_times)-1:
                next_time = msg_times[i+1]

                while not self.thread_stop_event.is_set():
                    now = (time.perf_counter() - play_anchor) * speed
                    if now >= next_time:
                        break
                    time.sleep(0.001)

            elif i == len(msg_times)-1:
                self.stop()

    def stop(self):
        """stop playback, setting thread_stop_event"""
        if self.playback_thread is not None and self.playback_thread.is_alive():
            self.thread_stop_event.set()
            # Wait for thread to finish (will soon, since stop_event is set)
            # self.playback_thread.join()
        
        # self.wall_clock.pause()
        self.midi_synth.pause()

    @staticmethod
    def binary_search(arr, target):
        """generic binary search implementation for time searching
        also return the index in the arr where the target was found
        """
        med = 0
        res = arr[med]
        low = 0
        high = len(arr) - 1

        while low <= high:
            mid = int((high-low)/2 + low)
            
            # RESULT UPDATING
            # update res if mid is closer to target
            if abs(arr[mid]-target) < abs(res-target):
                res = arr[mid]
            # return max if tie
            elif abs(arr[mid]-target) == abs(res-target):
                res = max(res, arr[mid])

            # BOUND EDITING
            # return the mid if it's the target
            if arr[mid] == target:
                return arr[mid], mid
            elif arr[mid] < target:
                low = mid + 1
            else:
                high = mid - 1
            
        return res, mid