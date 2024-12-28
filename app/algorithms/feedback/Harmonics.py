
import pandas as pd
import numpy as np

def is_harmonic(f1, f2, harmonic_range=0.75):
    """
    Check if two frequencies are harmonically related.
    """
    MIN_VIOLIN_FREQ = 196
    MAX_VIOLIN_FREQ = 5000

    # Find all harmonics of the note within violin frequency range
    above_harmonics = [f1 * i for i in range(1, 10) if f1 * i <= MAX_VIOLIN_FREQ]
    below_harmonics = [f1 / i for i in range(2, 10) if f1 / i >= MIN_VIOLIN_FREQ]
    harmonics = above_harmonics + below_harmonics # All harmonics to search for

    freq_to_midi = lambda freq: 12*np.log(freq/220)/np.log(2) + 57
    harmonic_midi_pitches = [freq_to_midi(harmonic) for harmonic in harmonics]

    

def group_harmonics(note_df: pd.DataFrame, harmonic_range=0.75):
    MIN_VIOLIN_FREQ = 196
    MAX_VIOLIN_FREQ = 5000

    harmonic_groups = []
    visited = set()
    print("Grouping harmonics...")

    for i, note_row in note_df.iterrows():

        if i in visited: # Ignore notes that have already been grouped
            continue

        frequency = note_row['frequency']

        # Find all harmonics of the note within violin frequency range
        above_harmonics = [frequency * i for i in range(1, 10) if frequency * i <= MAX_VIOLIN_FREQ]
        below_harmonics = [frequency / i for i in range(2, 10) if frequency / i >= MIN_VIOLIN_FREQ]
        harmonics = above_harmonics + below_harmonics # All harmonics to search for

        # Convert harmonics to MIDI pitches
        # https://www.music.mcgill.ca/~gary/307/week1/node28.html
        freq_to_midi = lambda freq: 12*np.log(freq/220)/np.log(2) + 57
        harmonic_midi_pitches = [freq_to_midi(harmonic) for harmonic in harmonics]

        
        harmonic_group = []
        visited.add(i)
        for j, next_note_row in note_df.iterrows():
            if j in visited: # Again ignore notes that have already been grouped
                continue

            next_midi_pitch = next_note_row['midi_pitch']

            # Check if next_midi_pitch is within a ±0.5 range of any harmonic_midi_pitches
            if any(abs(next_midi_pitch - harmonic_midi_pitch) <= harmonic_range for harmonic_midi_pitch in harmonic_midi_pitches):
                # Group the notes together
                harmonic_group.append(next_note_row)
                visited.add(j)

            else:
                break # stop searching for consecutive harmonics

        harmonic_groups.append(harmonic_group)

    return harmonic_groups