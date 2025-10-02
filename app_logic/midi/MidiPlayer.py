import threading
import time
from typing import Union
import logging
from PyQt6.QtCore import QObject

from app_logic.midi.MidiData import MidiData
from app_logic.midi.MidiSynth import MidiSynth
from ui import WallClock

class MidiPlayer(QObject):
    """object to play a midi_data"""

    def __init__(self, midi_synth: MidiSynth, wall_clock: WallClock):
        super().__init__()
        self.midi_synth = midi_synth
        self.midi_data: MidiData = None
        self.wall_clock = wall_clock

        # threading variables
        self.playback_thread: threading.Thread = None
        self.thread_stop_event = threading.Event()

        # other playback variables
        self.current_time = 0
        self.playback_speed = 1.0

        # channel stuff
        self.all_channels = []
        self.active_channels = []

    def load_midi(self, midi: Union[str, MidiData]):
        """
        Load a MIDI file from a file path (str) or a MidiData object 
        and sets the internal midi_data + all_channels/active_channels
        
        Args:
            midi: Union[str, MidiData], path to MIDI file or MidiData object
        """
        if isinstance(midi, str):
            self.midi_data = MidiData(midi)
        elif isinstance(midi, MidiData):
            self.midi_data = midi
        else:
            raise ValueError("midi source must be either a file path (str) or a MidiData object")

        # set all/current channels
        self.all_channels = self.midi_data.channels
        self.active_channels = self.all_channels

    def set_channels(self, channels: list):
        """
        Set the channels to play from the MIDI file.
        Args:
            channels: list, channels to play
        """
        # you wouldn't think we'd need this but just in case lmao
        valid_channels = [c for c in channels if c in self.all_channels]
        self.active_channels = valid_channels

    def play(self, start_time: float=0):
        """play a MIDI file using the synthesizer from a particular time
        handles threading logic and calls internal function _play()

        Args:
            start_time: time (sec) to start playing from
        """
        # exit if no midi data loaded
        if self.midi_data is None:
            print("Ignoring MIDI playback, no MIDI data loaded.")
            return
        
        # if playback thread already exists
        # eg, seeking while it's still playing
        if self.playback_thread is not None and self.playback_thread.is_alive():
            self.thread_stop_event.set()
            self.midi_synth.pause()
            self.playback_thread.join()

        # clear stop event (eg, no longer triggers event during playback)
        self.wall_clock.start(t=start_time)
        self.thread_stop_event.clear()
        self.playback_thread = threading.Thread(target=self._play, args=(start_time,))
        self.playback_thread.start()

    def _play(self, start_time: float=0):
        """
        internal function called by self.playback_thread to handle playback from any
        time in the MIDI file
        """
        # print('made it here')
        # even if start_time =/= 0, ensure all programs are initialized
        for _, program_change_messages in self.midi_data.programs.items():
            self.midi_synth.handle_midi(program_change_messages[0])
        
        msg_times = list(self.midi_data.messages.keys())
        # find index to start from based on start_time
        _, start_idx = MidiPlayer.binary_search(msg_times, start_time)


        for i in range(start_idx, len(msg_times)):
            # stop handling : if stop event is set, break look and exit playback
            if self.thread_stop_event.is_set():
                # self.midi_synth.pause()
                break

            # iterate through the messages and play them
            current_time = msg_times[i]
            messages = self.midi_data.messages[current_time]

            for msg in messages:
                # only handle messages in active_channels
                if hasattr(msg, "channel") and msg.channel in self.active_channels:
                    self.midi_synth.handle_midi(msg)
                elif msg.is_meta:
                    self.midi_synth.handle_midi(msg)
                
            # compute time to sleep until next msg
            if i+1 <= len(msg_times)-1:
                next_time = msg_times[i+1]

                while not self.thread_stop_event.is_set() and self.wall_clock.now() < next_time:
                    time.sleep(0.001)

                # sleep_time = next_time - current_time
                # time.sleep(sleep_time)
                
            elif i == len(msg_times)-1:
                self.pause()

    def pause(self):
        """pause playback, setting thread_stop_event"""
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