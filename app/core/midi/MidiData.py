import mido
import pandas as pd
import logging
import tempfile

class MidiLoader:
    def __init__(self):
        """
        MidiLoader is a class for parsing MIDI files into playable & readable dfs.
        
        Creates the following
        - message_dict: dict, {elapsed_time: [msg1, msg2, ...]}
        - program_dict: dict, {channel: program_change (Message)}
        - pitch_df: DataFrame with columns for 
            -> start_time, channel, pitch, velocity, duration
        """
        pass

    @staticmethod
    def parse_midi(midi_file_path: str, tempo_factor=1) -> tuple[dict, dict, pd.DataFrame]:
        """
        Parse a MIDI file into a message_dict, program_dict, and pitch_df.
        """
        midi_data = mido.MidiFile(midi_file_path) # Creates array of Message objects

        message_dict = {}
        program_dict = {}

        elapsed_time = 0
        for msg in midi_data:
            if msg.is_meta:
                continue # Skip meta messages like tempo change, key signature, etc.
            
            time_since_last_msg = msg.time * tempo_factor
            elapsed_time += time_since_last_msg

            if elapsed_time not in message_dict:
                message_dict[elapsed_time] = []

            message_dict[elapsed_time].append(msg)

            # Add 'program_change' messages to program_dict
            if msg.type == 'program_change':
                program_dict[msg.channel] = msg
        
        pitch_df = MidiLoader.create_pitchdf(message_dict)
        return message_dict, program_dict, pitch_df


    @staticmethod
    def create_pitchdf(message_dict: dict) -> pd.DataFrame:
        """
        Internal function to create a more interpretable DataFrame of pitches, 
        velocity, and duration from a message_dict object
        @param:
            - message_dict: dict, keys are elapsed time (in sec) and values are 
                            lists of messages all occurring at that time
                -> dict, {elapsed_time: [msg1, msg2, ...]}
        @return:
            - pitch_df: Dataframe with columns for 
                -> start time, pitch, MIDI number, velocity, duration
        """
        note_start_times = {}  # Dictionary to keep track of note start times
        rows = []  # List to store note details including calculated duration

        note_idx = 0
        for elapsed_time, messages in message_dict.items():
            for msg in messages: # Iterate through all messages at a given time

                # Check if the message is a note-related message before accessing the note attribute
                if msg.type in ['note_on', 'note_off']:
                    key = (msg.channel, msg.note)  # Unique key for each note

                    if msg.type == 'note_on' and msg.velocity > 0:
                        # Record the start time of the note
                        velocity = msg.velocity
                        note_start_times[key] = (elapsed_time, velocity)

                    # Stop note and compute duration
                    elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                        # Calculate duration and prepare the row for DataFrame
                        if key in note_start_times:
                            start_time, velocity = note_start_times[key]
                            duration = elapsed_time - start_time
                            row = [int(note_idx), start_time, msg.channel, msg.note, velocity, duration]
                            note_idx += 1
                            rows.append(row)
                            del note_start_times[key]  # Remove the note from start times

        # Create DataFrame from the rows
        pitch_df = pd.DataFrame(rows, columns=['note_idx', 'start', 'channel', 'pitch', 'velocity', 'duration'])
        
        # Create frequency col (https://www.music.mcgill.ca/~gary/307/week1/node28.html)
        pitch_df['frequency'] = 440 * (2 ** ((pitch_df['pitch'] - 69) / 12))
        return pitch_df


class MidiData:
    """
    MidiData stores all data and methods related to a single midifile.
    
    Stores the following data structures:
        - message_dict: dict, {elapsed_time: [msg1, msg2, ...]}
        - program_dict: dict, {channel: program_change (Message)}
        - pitch_df: DataFrame with columns for 
            -> start_time, channel, pitch, velocity, duration
    
    And provides the following methods:
        - get_length() -> float: returns the length of the MIDI file in seconds
        - get_channels() -> list: returns the list of channels in the MIDI file
    """

    def __init__(self, midi_file_path: str):
        """
        Initialize the MidiData object by parsing the given MIDI file.
        """
        self.midi_file_path = midi_file_path
        self.tempo_factor = 1
        self.message_dict, self.program_dict, self.pitch_df = MidiLoader.parse_midi(midi_file_path)

    def get_length(self) -> float:
        """Get the length of the MIDI file in seconds."""
        if self.pitch_df is None:
            logging.error("No MIDI data loaded!")
            return 0
        
        return self.pitch_df['start'].max() + self.pitch_df['duration'].max()
    
    def get_channels(self) -> list:
        """Get the list of channels in the MIDI file."""
        return list(self.program_dict.keys())

    def change_tempo(self, tempo_factor: float=None, target_length: float=None) -> None:
        """
        Multiplies all note lengths in message_dict by the tempo factor.
        Really, just recalls the initialization function with a specific tempo factor
        based on how fast we want it.

        Used to help DTW find a better alignment.

        Args:
            tempo_factor: Float by which to multiply duration of all notes in midi
            target_length: How long we want our midi to be. (Used in the app.)
        """

        if tempo_factor is None and target_length is None:
            logging.error("Either tempo_factor or target_length must be provided.")
            return

        # Calculate the current length of the MIDI in seconds
        current_length = self.get_length()
        
        # Determine the scaling factor
        if target_length is not None:
            tempo_factor = target_length / current_length
            logging.info(f"Setting tempo to achieve target length: {target_length}s with a factor of {tempo_factor}")
        else:
            logging.info(f"Applying tempo factor: {tempo_factor}")

        # Update message_dict: scale all elapsed times
        self.tempo_factor = tempo_factor
        self.message_dict, self.program_dict, self.pitch_df = MidiLoader.parse_midi(self.midi_file_path, tempo_factor)

        print(f"Tempo change applied. New MIDI length is {self.get_length()} seconds.")

    def save_to_file(self) -> None:
        """
        Saves the current MIDI data to a file, "tempo_warped.mid"
        """
        # with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as temp_file:
        # temp_file_path = temp_file.name
        midi_file = self._to_midi_file()
        midi_file.save('tempo_warped.mid')

    def _to_midi_file(self, ticks_per_beat=480):
        """
        Converts a list of mido.Message objects into a MIDI file.
        
        Args:
            messages (list): List of mido.Message objects.
            output_path (str): Path to save the MIDI file.
            ticks_per_beat (int): Ticks per beat for the MIDI file (default is 480).
        """
        # Create a new MIDI file and add a track
        midi_file = mido.MidiFile(ticks_per_beat=ticks_per_beat)
        track = mido.MidiTrack()
        midi_file.tracks.append(track)

        # Track the previous elapsed time to calculate delta times
        previous_time = 0

        for elapsed_time, messages in sorted(self.message_dict.items()):
            # Calculate the delta time in seconds
            delta_time_seconds = elapsed_time - previous_time
            previous_time = elapsed_time  # Update previous time

            # Convert delta time from seconds to ticks based on original tempo
            for msg in messages:
                if msg.time > 0:
                    # Convert msg.time (in seconds) to ticks
                    msg.time = int(delta_time_seconds * ticks_per_beat * self.tempo_factor)
                track.append(msg)
                # Set delta time to 0 for subsequent messages at the same time
                delta_time_seconds = 0

        return midi_file
        