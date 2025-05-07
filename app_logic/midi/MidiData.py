import mido
import pandas as pd
import logging

class MidiData:
    def __init__(self, midi_filepath: str, tuning: float=440.0):

        self.midi_filepath: str = midi_filepath
        self.tuning = tuning
        self.tempo = 0

        self.message_dict, self.program_dict, self.meta_dict = self.parse_messages(midi_filepath)
        self.note_df = self.make_note_df(self.message_dict)

    def parse_messages(self, midi_filepath: str):
        """load the midi file, parsing the message list and creating
        something that MidiSynth and MidiPlayer can work with"""
        midi_data = mido.MidiFile(midi_filepath) # a list of message objects

        meta_dict = {}
        message_dict = {}
        program_dict = {}

        elapsed_time = 0
        for msg in midi_data:

            # update time elapsed
            time_since_last_msg = msg.time
            elapsed_time += time_since_last_msg

            if msg.is_meta:
                if msg.type == "set_tempo":
                    self.tempo = round(mido.tempo2bpm(msg.tempo))

                meta_dict.setdefault(elapsed_time, []).append(msg)
                continue # don't add meta messages to the message_dict

            # append the msg with the elapsed time into the message_dict
            message_dict.setdefault(elapsed_time, []).append(msg)

            # do same for program_dict
            if msg.type == "program_change":
                program_dict.setdefault(msg.channel, []).append(msg)

        return message_dict, program_dict, meta_dict
    
    def make_note_df(self, message_dict: dict[float, list[mido.Message]]):
        """create a dataframe of notes from the messages
        allows easy comparison to user notes later on"""

        note_onsets = {} # record note start times
        note_df_rows = [] # store list of rows

        note_idx = 0
        for elapsed_time, messages in message_dict.items():
            for msg in messages:
                # skip non-note related stuff
                if msg.type not in ['note_on', 'note_off'] or msg.is_meta:
                    continue
                
                key = (msg.channel, msg.note) # create unique key per note

                # velocity >0 because sometimes midi files are weird lol
                if msg.type == 'note_on' and msg.velocity > 0:
                    note_onsets[key] = (elapsed_time, msg.velocity)

                elif msg.type=='note_off' or (msg.type=='note_on' and msg.velocity==0):
                    if key not in note_onsets:
                        continue
                    # end the note we recorded in note_onsets and append to the df
                    start_time, velocity = note_onsets[key]
                    duration = elapsed_time - start_time
                    note_df_row = [int(note_idx), start_time, msg.channel, msg.note, velocity, duration]
                    note_df_rows.append(note_df_row)

                    # cleanup our iteration variables
                    del note_onsets[key]
                    note_idx += 1

        # create final df from the accumulated rows
        note_df = pd.DataFrame(
            note_df_rows,
            columns=['note_idx', 'start', 'channel', 'midi_num', 'velocity', 'duration']
        )
        note_df['frequency'] = self.tuning * (2 ** ((note_df['midi_num']-69) /12))
        return note_df
    
    def update_tuning(self, tuning: float):
        self.tuning = tuning
        self.note_df['frequency'] = self.tuning * (2 ** ((self.note_df['midi_num']-69) /12))

    def get_channels(self):
        return list(self.program_dict.keys())
    
    def get_tempo(self):
        return self.tempo
    
    def get_programs(self):
        programs = []
        for msgs in self.program_dict.values():
            for m in msgs:
                programs.append(m.program)
        return programs
        
    
    def get_length(self) -> float:
        """get length of MIDI file in seconds"""
        if self.note_df is None:
            logging.error("No MIDI data loaded!")
            return 0
        return self.note_df['start'].iloc[-1] + self.note_df['duration'].iloc[-1]


if __name__ == "__main__":
    # test it out
    MIDI_FILEPATH = 'data/fugue_midi.mid'
    midi_data = MidiData(MIDI_FILEPATH)
    print(f"midi programs: {midi_data.get_programs()}")