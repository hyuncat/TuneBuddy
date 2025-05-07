## Synchrony
An opinionated practice tool for musicians. Upload the MIDI score of what you want to play, record yourself playing any pitched monophonic instrument (violin, voice, flute, etc.), then analyze how you did!


### The app
Pictured
- The notes corresponding to the uploaded MIDI file (in grey).
- A multicolor line, where the x-axis is time, y-axis is the pitch, and the color represents the volume of the user's audio.

<img width="800" alt="App screenshot" src="https://github.com/user-attachments/assets/56b612a8-f952-422a-b1b5-cc9533375cba" />

### User flow

1. The user uploads the piece they are trying to play
2. The user records their audio and sees the corrections in real time
3. We analyze the user's mistakes

<img width="300" alt="User flow" src="https://github.com/user-attachments/assets/c25e311c-0a5d-4183-a281-ff21531d402e" />

### Code logic flow

We store the **user's data** in `UserData`, which contains the user's audio, pitches, and notes. Is acted upon by audio recorder, player, pitch detector, and note detector.

**MIDI data** is stored in `MidiData`, containing notes and is acted upon by the MIDI player / synth classes.

<img width="800" alt="Logic flow" src="https://github.com/user-attachments/assets/eab396e0-a74f-49a0-ae6a-459fcc5b33f4" />

