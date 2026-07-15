// Builds a minimal Standard MIDI File (format 0, single track) from note
// data, for playback via js-synthesizer's built-in SMF player
// (addSMFDataToPlayer/playPlayer/seekPlayer/setPlayerTempo/
// retrievePlayerCurrentTick). Chosen over manually scheduling
// midiNoteOn/midiNoteOff via setTimeout: the built-in player gives
// tick-accurate scheduling, live tempo changes, and seeking for free,
// without hand-rolling a scheduler subject to setTimeout jitter.
//
// The desktop app's own MidiPlayer (app_logic/midi/MidiPlayer.py) instead
// schedules events by hand against time.perf_counter(), bypassing
// fluidsynth's player entirely - but that's driven by needs specific to its
// architecture (live per-message channel muting, a shared WallClock across
// two tabs). This web port achieves the same user-facing behavior (mute a
// channel, hear it stop) by rebuilding and reloading the SMF when mute
// state changes instead - see playback.svelte.js's rebuild-on-mute-change.

const TICKS_PER_QUARTER = 480;

function encodeVarLen(value) {
  const bytes = [value & 0x7f];
  value = value >>> 7;
  while (value > 0) {
    bytes.unshift((value & 0x7f) | 0x80);
    value = value >>> 7;
  }
  return bytes;
}

export function secondsToTicks(seconds, bpm) {
  return Math.round(seconds * (bpm / 60) * TICKS_PER_QUARTER);
}

export function ticksToSeconds(ticks, bpm) {
  return (ticks / TICKS_PER_QUARTER / bpm) * 60;
}

// channelsData: array of { channel, program, notes }, where notes are
// noteFromArray()-shaped ({ startTime, endTime, midiNum, velocity }).
// bpm: the tempo used to convert note_data's second-based times to ticks -
// always the score's ORIGINAL bpm (not any live setPlayerTempo scaling),
// since ticks are musical-time, not wall-clock time.
// minDurationSeconds: floors the track's length (via the end-of-track meta
// event's position) even if channelsData has few/no events - otherwise
// muting every channel collapses the SMF to almost nothing, and the player
// races through it instead of silently running for the song's real length.
export function buildSMF(channelsData, bpm, minDurationSeconds = 0) {
  const events = [];

  const microsPerQuarter = Math.round(60000000 / bpm);
  events.push({
    tick: 0,
    order: -1,
    bytes: [
      0xff,
      0x51,
      0x03,
      (microsPerQuarter >> 16) & 0xff,
      (microsPerQuarter >> 8) & 0xff,
      microsPerQuarter & 0xff,
    ],
  });

  for (const { channel, program, notes } of channelsData) {
    if (program != null) {
      events.push({ tick: 0, order: 0, bytes: [0xc0 | (channel & 0x0f), program & 0x7f] });
    }
    for (const note of notes) {
      const startTick = secondsToTicks(note.startTime, bpm);
      const endTick = Math.max(startTick + 1, secondsToTicks(note.endTime, bpm));
      const velocity = Math.max(1, Math.min(127, Math.round(note.velocity ?? 90)));
      for (const midi of note.midiNum) {
        if (midi == null || midi < 0) continue;
        const key = Math.max(0, Math.min(127, Math.round(midi)));
        // note-offs sort before note-ons at the same tick (order 1 vs 2) so
        // a repeated pitch's release doesn't clip the next attack.
        events.push({ tick: endTick, order: 1, bytes: [0x80 | (channel & 0x0f), key, 0] });
        events.push({ tick: startTick, order: 2, bytes: [0x90 | (channel & 0x0f), key, velocity] });
      }
    }
  }

  events.sort((a, b) => a.tick - b.tick || a.order - b.order);

  const trackBytes = [];
  let lastTick = 0;
  for (const ev of events) {
    trackBytes.push(...encodeVarLen(ev.tick - lastTick));
    trackBytes.push(...ev.bytes);
    lastTick = ev.tick;
  }
  const minTick = secondsToTicks(minDurationSeconds, bpm);
  trackBytes.push(...encodeVarLen(Math.max(0, minTick - lastTick)), 0xff, 0x2f, 0x00); // end of track

  const header = [
    0x4d, 0x54, 0x68, 0x64, // "MThd"
    0x00, 0x00, 0x00, 0x06, // header length = 6
    0x00, 0x00, // format 0
    0x00, 0x01, // 1 track
    (TICKS_PER_QUARTER >> 8) & 0xff, TICKS_PER_QUARTER & 0xff,
  ];
  const trackHeader = [
    0x4d, 0x54, 0x72, 0x6b, // "MTrk"
    (trackBytes.length >>> 24) & 0xff,
    (trackBytes.length >>> 16) & 0xff,
    (trackBytes.length >>> 8) & 0xff,
    trackBytes.length & 0xff,
  ];

  return new Uint8Array([...header, ...trackHeader, ...trackBytes]);
}

export { TICKS_PER_QUARTER };
