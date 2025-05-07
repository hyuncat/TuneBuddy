import fluidsynth
import logging
import mido

class MidiSynth:
    def __init__(self, soundfont_path: str):
        """
        MidiSynth uses pyfluidsynth to control MIDI playback.
        Keeps track of now notes to ensure all note playback stops
        when paused.
        """
        print("Loading MidiSynth...")
        self.synth = fluidsynth.Synth()

        # coreaudio is for MacOS
        # other options: 'alsa', 'dsound', etc., based on OS
        self.synth.start(driver='coreaudio')
        self.soundfont_id = self.synth.sfload(soundfont_path)

        print("Synth + soundfont loaded.")
        
        # track 'now' playing notes: {channel: [midi_number, ...]}
        self.now = {}
    
    def handle_midi(self, msg: mido.Message):
        """
        Handle note_on, note_off, and program_change MIDI messages.
        """
        if msg.type == 'note_on':
            self.synth.noteon(msg.channel, msg.note, msg.velocity)
            # also add channel+notes to currently playing notes
            if msg.channel not in self.now:
                self.now[msg.channel] = []
            self.now[msg.channel].append(msg.note)
        
        elif msg.type == 'note_off':
            self.synth.noteoff(msg.channel, msg.note)
            if msg.channel not in self.now:
                logging.warning("Trying to turn off a message on a channel not previously defined")
        
            # also remove note from currently playing notes
            new_now = [n for n in self.now[msg.channel] if n != msg.note]
            self.now[msg.channel] = new_now
        
        elif msg.type == 'program_change':
            # worry about 'banks' later
            self.synth.program_change(msg.channel, msg.program)
        
        elif msg.type == "control_change":
            # handle control changes for things like volume, pan, etc.
            # for now, just log the message
            logging.info(f"Control change: {msg}")

        else:
            logging.warning(f"Unhandled message: {msg}")

    def pause(self):
        """Stop all currently playing notes
        
        Iterates through all possible channels and pitches just in case
        there's something not accounted for
        """
        TOTAL_CHANNELS = 16
        PITCH_RANGE = 128
        for channel in range(TOTAL_CHANNELS):
            for midi_note_number in range(PITCH_RANGE):
                self.synth.noteoff(channel, midi_note_number)