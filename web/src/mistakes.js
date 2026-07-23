// Client-side mistake classification over an already-fixed set of alignment
// pairs. These are direct ports of MistakeDetector.py's CLASSIFICATION logic
// only - not the alignment DP itself, which stays server-side via /realign
// (pitch tolerance is baked into the DP's cost matrix and can change which
// notes pair with which, not just whether a pairing counts as a mistake; see
// project notes on why). Both functions here are pure threshold checks over
// pairs that are already decided, so they're safe to run locally.

// Converts one of /analyze's or /notedata's serialized note arrays
// ([id, start, end, [midi...], velocity, instrument, baseStart, baseEnd])
// into a plain object - mirrors JsonHandler._note_from_payload. Exported:
// the results UI needs this same conversion for notes that aren't part of
// any mistake (a clean match), not just the ones classify* returns.
export function noteFromArray(arr) {
  if (arr == null) return null;
  return {
    id: arr[0],
    startTime: arr[1],
    endTime: arr[2],
    midiNum: arr[3],
    velocity: arr[4],
    instrument: arr[5],
    baseStartTime: arr[6] ?? arr[1],
    baseEndTime: arr[7] ?? arr[2],
  };
}

// Closest pitch distance between a user note and a score note. scoreNote's
// midiNum can hold multiple pitches (a chord) - the user matches whichever
// chord member is nearest. Mirrors MistakeDetector.get_distance().
function pitchDistance(userNote, scoreNote) {
  let min = Infinity;
  for (const u of userNote.midiNum) {
    for (const m of scoreNote.midiNum) {
      const d = Math.abs(u - m);
      if (d < min) min = d;
    }
  }
  return min;
}

// Classifies each pair as a deletion (score note never played), insertion
// (extra user note), or substitution (paired but too far in pitch); a clean
// match produces no entry. Mirrors the classification half of
// MistakeDetector._align()'s traceback - note the >= boundary (a distance
// exactly AT tolerance counts as a mistake), matching the Python original.
export function classifyPitchMistakes(pairs, userNotes, scoreNotes, pitchTolerance) {
  const mistakes = [];
  pairs.forEach(([userIdx, scoreIdx], pairIndex) => {
    const userNote = userIdx != null ? noteFromArray(userNotes[userIdx]) : null;
    const scoreNote = scoreIdx != null ? noteFromArray(scoreNotes[scoreIdx]) : null;

    if (userIdx === null) {
      mistakes.push({ type: "deletion", pairIndex, userIdx, scoreIdx, userNote, scoreNote });
      return;
    }
    if (scoreIdx === null) {
      mistakes.push({ type: "insertion", pairIndex, userIdx, scoreIdx, userNote, scoreNote });
      return;
    }
    const distance = pitchDistance(userNote, scoreNote);
    if (Math.abs(distance) >= pitchTolerance) {
      mistakes.push({ type: "substitution", pairIndex, userIdx, scoreIdx, userNote, scoreNote, distance });
    }
  });
  return mistakes;
}

// Derives early/late/short/long timing mistakes from the same fixed pairs -
// a pure threshold check, no alignment involved. Mirrors
// MistakeDetector.detect_timing_mistakes() exactly, including its strict >
// boundary (distinct from the pitch check's >= above - preserved as-is from
// the Python original, not an inconsistency I introduced). A single pair can
// produce up to two mistakes (onset AND duration), independently.
export function classifyTimingMistakes(pairs, userNotes, scoreNotes, timingTolerance) {
  const timingTol = Math.max(0, timingTolerance);
  const mistakes = [];
  pairs.forEach(([userIdx, scoreIdx], pairIndex) => {
    if (userIdx === null || scoreIdx === null) return;
    const userNote = noteFromArray(userNotes[userIdx]);
    const scoreNote = noteFromArray(scoreNotes[scoreIdx]);

    const onsetOff = userNote.startTime - scoreNote.startTime;
    const userDur = Math.max(1e-9, userNote.endTime - userNote.startTime);
    const scoreDur = Math.max(1e-9, scoreNote.endTime - scoreNote.startTime);
    const durOff = userDur - scoreDur;

    if (Math.abs(onsetOff) > timingTol) {
      mistakes.push({
        type: onsetOff > 0 ? "late" : "early",
        pairIndex,
        userIdx,
        scoreIdx,
        userNote,
        scoreNote,
        info: `${onsetOff >= 0 ? "+" : ""}${onsetOff.toFixed(2)}s`,
      });
    }
    if (Math.abs(durOff) > timingTol) {
      mistakes.push({
        type: durOff > 0 ? "long" : "short",
        pairIndex,
        userIdx,
        scoreIdx,
        userNote,
        scoreNote,
        info: `${durOff >= 0 ? "+" : ""}${durOff.toFixed(2)}s`,
      });
    }
  });
  return mistakes;
}

// A mistake's own category, independent of whichever tab (pitch/timing) is
// currently active in ResultsView - insertion/deletion/substitution are
// pitch mistakes, early/late/long/short are timing ones. Needed anywhere a
// mistake is looked at outside that active-tab context (e.g. the ScoreViewer
// annotation payload combines both lists at once - see annotations.js).
export function mistakeCategory(type) {
  return type === "early" || type === "late" || type === "long" || type === "short"
    ? "timing"
    : "pitch";
}

// Mirrors Config.get_note_name(): converts a MIDI number to a letter name
// like "C4"/"F#5". Rests/invalid pitches (null or negative) render as "—".
const SHARP_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
export function noteName(midiNum) {
  if (midiNum == null || midiNum < 0) return "—";
  const n = Math.round(midiNum);
  const pitchClass = ((n % 12) + 12) % 12; // JS % can go negative; Python's can't
  const octave = Math.floor(n / 12) - 1;
  return `${SHARP_NOTE_NAMES[pitchClass]}${octave}`;
}

// Inverse of noteName(): parses a letter name ("G3", "F#5", or "Bb3" - flats
// accepted for input even though noteName() only ever emits sharps) back to
// a MIDI number. Returns null for anything unparseable, so callers can
// distinguish "invalid input" from a valid MIDI 0.
const LETTER_SEMITONES = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };
export function noteNameToMidi(name) {
  const match = /^([A-Ga-g])([#b]?)(-?\d+)$/.exec(String(name ?? "").trim());
  if (!match) return null;
  const [, letter, accidental, octaveStr] = match;
  let semitone = LETTER_SEMITONES[letter.toUpperCase()];
  if (accidental === "#") semitone += 1;
  else if (accidental === "b") semitone -= 1;
  return (parseInt(octaveStr, 10) + 1) * 12 + semitone;
}

// Config.fmin/fmax are in Hz, not MIDI - this is the same A440-equal-temperament
// formula Config/PitchDetector use, parameterized by the current tuning
// reference (ToleranceWidget-adjacent SettingsWidget's Tuning field) instead
// of hardcoding 440.
export function midiToHz(midiNum, tuning = 440) {
  return tuning * Math.pow(2, (midiNum - 69) / 12);
}
