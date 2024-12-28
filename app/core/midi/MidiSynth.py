from mido import Message
import fluidsynth
import logging

class MidiSynth:
    def __init__(self, soundfont_path: str):
        """
        MidiSynth uses pyfluidsynth to control MIDI playback.
        Keeps track of currently_playing notes to ensure all note playback stops
        when paused.
        """
        print("Loading MidiSynth...")
        self.synth = fluidsynth.Synth()

        # Coreaudio is for MacOS
        # Other options: 'alsa', 'dsound', etc., based on OS
        self.synth.start(driver='coreaudio')
        self.soundfont_id = self.synth.sfload(soundfont_path)

        print("Synth + soundfont loaded.")
        
        # Track playing notes: {channel: [midi_number, ...]}
        self.currently_playing = {}

    def handle_midi(self, message: Message):
        """
        Handle note_on, note_off, and program_change MIDI messages.
        """
        if message.type == 'note_on':
            self.synth.noteon(message.channel, message.note, message.velocity)
            # Also add channel+notes to currently playing notes
            if message.channel not in self.currently_playing:
                self.currently_playing[message.channel] = []
            self.currently_playing[message.channel].append(message.note)
        
        elif message.type == 'note_off':
            self.synth.noteoff(message.channel, message.note)
            # Also remove note from currently playing notes
            if message.channel in self.currently_playing: # (Should always be true)
                old_currently_playing = self.currently_playing[message.channel]
                new_currently_playing = [old_note for old_note in old_currently_playing if old_note != message.note]
                self.currently_playing[message.channel] = new_currently_playing
        
        elif message.type == 'program_change':
            # Worry about 'banks' later
            self.synth.program_change(message.channel, message.program)
        
        elif message.type == "control_change":
            # Handle control changes for things like volume, pan, etc.
            # For now, just log the message
            logging.info(f"Control Change: {message}")

        else:
            logging.warning(f"Unhandled message: {message}")
    
    def pause(self):
        """Stop all currently playing notes."""
        TOTAL_CHANNELS = 16
        PITCH_RANGE = 128
        for channel in range(TOTAL_CHANNELS):
            for midi_note_number in range(PITCH_RANGE):
                self.synth.noteoff(channel, midi_note_number)
         
