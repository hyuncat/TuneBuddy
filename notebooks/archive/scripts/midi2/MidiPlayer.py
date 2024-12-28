import threading
import time
import logging

from .MidiSynth import MidiSynth
from .MidiData import MidiData

class MidiPlayer:
    """
    Class for real-time playback of MIDI files using a MidiSynth instance.
    Stores MidiData for a particular loaded MIDI file and keeps track of playback time, 
    currently playing channels.

    @methods:
    - load_midi(midi_file_path: str): Load the player with a MIDI file from a file path
    - play(start_time=0): Play a MIDI file from a particular time
    - pause(): Pause playback

    @attr:
    MIDI data structures:
        - MidiSynth: MidiSynth instance for handling playback
        - midi_data: MidiData instance with the following data structures:
            - message_dict: dict, {elapsed_time: [msg1, msg2, ...]}
            - program_dict: dict, {channel: program_change (Message)}
            - pitch_df: DataFrame with columns for 
                -> start_time, channel, pitch, velocity, duration

    Threading variables:
        - playback_thread: threading.Thread, thread for playback
        - thread_stop_event: threading.Event, event to stop playback when is_set()

    Other playback variables:
        - current_time: float, current playback time in seconds
        - is_playing: bool, whether the player is currently playing
        - playback_speed: float, playback speed multiplier
        - all_channels: list, all channels in the MIDI file
        - current_channels: list, channels currently playing
    """
    def __init__(self, MidiSynth: MidiSynth):
        # MIDI data structures
        self.MidiSynth = MidiSynth
        self.midi_data = None

        # Threading variables
        self.playback_thread = None
        self.thread_stop_event = threading.Event()

        # Other playback variables
        self.current_time = 0
        self.is_playing = False
        self.playback_speed = 1.0
        self.all_channels = []
        self.current_channels = []

    def load_midi(self, midi_file_path: str):
        """
        Load a MIDI file from a file path (str) and stores data
        as instance variables.
        @param:
            - midi_file_path: str, path to MIDI file
        @side_effects:
            - Sets midi_data
            - Sets all_channels, current_channels
        """
        self.midi_data = MidiData(midi_file_path)

        # Set all/current channels
        self.all_channels = self.midi_data.get_channels()
        self.current_channels = self.all_channels

    def set_channels(self, channels: list):
        """
        Set the channels to play from the MIDI file.
        @param:
            - channels: list, channels to play
        """
        # Make sure channels exist within all_channels
        valid_channels = [c for c in channels if c in self.all_channels]
        self.current_channels = valid_channels

    def play(self, start_time=0):
        """
        Plays a MIDI file using the synthesizer from a particular time.
        Handles threading logic and calls internal function _play().
        @param:
            - start_from (float): Time in seconds to start playing from
        """
        # If no MIDI data loaded, exit
        if self.midi_data.message_dict is None:
            logging.error("No MIDI data loaded. Exiting.")
            return
        
        # If playback_thread already exists, stop playback, then re-join thread
        if self.playback_thread is not None and self.playback_thread.is_alive():
            self.thread_stop_event.set()
            self.synthesizer.pause()
            self.playback_thread.join()
        
        # Clear the stop event (no longer triggers stop during playback)
        self.thread_stop_event.clear() 
        self.playback_thread = threading.Thread(target=self._play, args=(start_time,))
        self.playback_thread.start()

    def _play(self, start_time=0):
        """
        Internal function called by self.playback_thread to handle playback from 
        any time in the MIDI file.

        Converts midi_data.message_dict into a list of all message times
        Finds the corresponding elapsed_time_index closest to start_time
        Then plays back the messages and sleeps the specified duration until next message.

        self.playback_speed is used to speed up or slow down playback.
        (Where 2 -> twice as fast, 0.5 -> half as fast)

        @param:
            - start_time (float): Time in seconds to start playing from
        """
        
        # Even if start_time =/= 0, ensure all programs are initialized
        for _, program_change_msg in self.midi_data.program_dict.items():
            self.MidiSynth.handle_midi(program_change_msg)
        
        all_message_times = list(self.midi_data.message_dict.keys())

        # Find the index to start from based on the start_from time
        start_index = None
        for index, elapsed_time in enumerate(all_message_times):
            if elapsed_time >= start_time:
                start_index = index
                break
        if start_index is None:
            print("Start time is beyond the last message. Exiting.")
            return
        
        # Play messages starting from start_index in message_dict
        for index in range(start_index, len(all_message_times)):
            # If stop_event is set, break out of loop and exit playback
            if self.thread_stop_event.is_set():
                break

            # Get the current time and messages at start_index
            current_time = all_message_times[index]
            current_messages = self.midi_data.message_dict[current_time]

            # Handle each message in current_messages
            for message in current_messages:
                # Only play messages on current channels
                if message.channel in self.current_channels:
                    self.MidiSynth.handle_midi(message)

            # Calculate the time to sleep before the next message
            if index+1 < len(all_message_times): # Until the last message
                next_time = all_message_times[index + 1]
                sleep_time = (next_time - current_time) / self.playback_speed
                time.sleep(sleep_time)


    def pause(self):
        """
        Pauses playback by setting the thread_stop_event and 
        pausing the synthesizer.
        """
        if self.playback_thread is not None and self.playback_thread.is_alive():
            self.thread_stop_event.set()
            # Wait for thread to finish (will soon, since stop_event is set)
            self.playback_thread.join() 
        
        self.MidiSynth.pause()