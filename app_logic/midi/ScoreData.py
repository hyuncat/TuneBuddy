from pathlib import Path
from music21 import converter, tempo, meter
import tempfile

from app_logic.midi.MidiData import MidiData
from app_logic.NoteData import NoteData, Note

class ScoreData:
    def __init__(self, filepath: str | Path=None): 
        # --- ESSENTIAL DATA ---
        # for their respective midiplayer / verovio uses
        self.midi_data: MidiData = None
        self.score = None # the music21 score object
        # note data for string editing and GuitarHero visualization
        self.note_datas: dict[int, NoteData] = {}

        # --- META ---
        # score metadata
        self.length = 0.0 # sec
        self.bpm, self.bpm_og = 120, 120
        self.title: str = "" # display title written to verovio

        # instrument selection
        self.instruments: dict[int, int] = {} # {channel: program_number}
        self.active_instrument: int = 0 # channel number (TODO: decouple this)
        self.displayed_instruments: set[int] = set() # channels to display
        self.playing_instruments: set[int] = set() # channels to play
        self.metronome_channel: int = None
        # self.metronome_on: bool = True
        # --- CLIPPING (non-destructive measure-range focus) ---
        # The clip is stored as a (first, last) pair of NOTE INDICES into the
        # active instrument's NoteData — NOT as times. Indices are stable across
        # tempo changes / resize (which rebuild the NoteData but preserve count +
        # order) and are tab-independent (both tabs parse the same file), so the
        # same clip is shared globally and never goes stale. None = no clip.
        # Everything else (the [b0, b1] time window, the clipped notes) is DERIVED
        # from this via clip_bounds() / clipped_note_data(). See those + is_clipped.
        self.clip: tuple[int, int] | None = None

        # cumulative transpose applied to the score, in half steps (signed). Reset
        # to 0 on load; transpose() shifts the score (MIDI + music21 + NoteData)
        # incrementally and accumulates here. Pitch-only, independent of the clip.
        self.transpose_semitones: int = 0

        # metronome beat grid: (time_sec, is_downbeat) tuples, also drives the
        # GuitarHero vertical gridlines. `beats_og` is the baseline at the
        # original tempo with no resize offset; `beats` is the live grid, rebuilt
        # from it whenever the timing changes (change_tempo / resize) so the
        # gridlines track the score. `_beat_offset` mirrors transpose_notes' shift.
        self.beats: list[tuple[float, bool]] = []
        self.beats_og: list[tuple[float, bool]] = []
        self._beat_offset: float = 0.0

        # note reading
        self.i = 0 # index of current note

        if filepath is not None:
            self.load(filepath)

    def update_time(self, t: float):
        """Update the current note index based on the current time t."""
        note_data = self.note_datas[self.active_instrument]
        # get prev note data
        prev_note = note_data.read_note(i=self.i-1)
        prev_note_end_time = prev_note.end_time if prev_note else 0.0

        # reset if needed
        if self.i > len(note_data.times) or t < prev_note_end_time:
            self.i = 0
        
        while self.i < len(note_data.times) and t >= note_data.read_note(i=self.i).end_time:
            self.i += 1
        # print(f"Updated time: {t:.2f} sec, current note index: {self.i}")

    def current_note(self) -> Note | None:
        """Return the current note based on the current note index."""
        if self.active_instrument not in self.note_datas:
            return None
        note_data = self.note_datas[self.active_instrument]
        if self.i < len(note_data.times):
            return note_data.read_note(i=self.i)
        return None

    # --- CLIPPING API (the single source of truth; see self.clip) ---
    def is_clipped(self) -> bool:
        return self.clip is not None

    def set_clip(self, i0: int, i1: int) -> None:
        """Clip to the active instrument's notes [i0, i1] (inclusive indices)."""
        if i1 < i0:
            i0, i1 = i1, i0
        self.clip = (i0, i1)

    def clear_clip(self) -> None:
        self.clip = None

    def note_index_range(self, t0: float, t1: float, channel: int | None = None
                         ) -> tuple[int, int] | None:
        """Indices of the active instrument's notes whose START falls in the
        half-open span [t0, t1) — i.e. ONLY notes inside the selected measures
        (a note exactly at t1, the next measure's first note, is excluded; a note
        ending at t0 is excluded since its start is < t0). None if none match."""
        channel = self.active_instrument if channel is None else channel
        nd = self.note_datas.get(channel)
        if not nd or not nd.times:
            return None
        eps = 1e-6
        idxs = [i for i, t in enumerate(nd.times) if t0 - eps <= t < t1 - eps]
        if not idxs:
            return None
        return (idxs[0], idxs[-1])

    def clip_bounds(self, channel: int | None = None) -> tuple[float, float] | None:
        """The clip's [start, end] time window in CURRENT app-time, DERIVED from
        the live note positions (so it auto-tracks tempo/resize). None = no clip.
        Used by the slider window, GuitarHero dimming, and the Verovio grey-out."""
        if self.clip is None:
            return None
        channel = self.active_instrument if channel is None else channel
        nd = self.note_datas.get(channel)
        if not nd or not nd.times:
            return None
        i0, i1 = self.clip
        if not (0 <= i0 <= i1 < len(nd.times)):
            return None
        return (nd.read_note(i=i0).start_time, nd.read_note(i=i1).end_time)

    def clipped_note_data(self, channel: int | None = None) -> NoteData:
        """The active instrument's notes WITHIN the clip (exactly indices i0..i1),
        or the full NoteData when unclipped. This is what the StringEditor /
        MistakeChecker / alignment consume so they only ever see the clip."""
        channel = self.active_instrument if channel is None else channel
        nd = self.note_datas[channel]
        if self.clip is None:
            return nd
        i0, i1 = self.clip
        if not (0 <= i0 <= i1 < len(nd.times)):
            return nd
        sub = NoteData()
        notes = nd.read(i=i0, j=i1 + 1)  # inclusive of i1
        sub.data = {n.start_time: n for n in notes}
        sub.times = sorted(sub.data.keys())
        return sub

    def load(self, filepath: str|Path):
        """Load a score file, either MIDI or MusicXML. Converts either 
        into the other such that we have both representations available.
        Supports file types: .mid, .midi, .mxl, .musicxml, .xml
        """
        p = Path(filepath)
        ext = p.suffix.lower()
        print(f"Loading score file: {filepath}")
        # default the score title to the filename; the RecordingTree shows the
        # same value and is the source of truth from here on (set_title).
        self.title = p.stem
        
        if ext not in {'.mid', '.midi', '.mxl', '.musicxml', '.xml', '.mei'}:
            raise ValueError(f"Cannot handle file type: {ext}")
    
        self.score = converter.parse(str(p))

        if ext in {'.mxl', '.musicxml', '.xml', '.mei'}:
            # convert to midi data, write to tempfile, then load midi data
            with tempfile.NamedTemporaryFile(suffix='.mid') as temp_midi_file:
                self.score.write('midi', fp=temp_midi_file.name)
                self.midi_data = MidiData(temp_midi_file.name)
        elif ext in {'.mid', '.midi'}: 
            self.midi_data = MidiData(p)

        self.length = self.midi_data.length_og
        self.bpm = self.score.metronomeMarkBoundaries()[0][2].number if self.score.metronomeMarkBoundaries() else 120
        # remember the tempo Verovio renders the score at, so later tempo changes
        # can be mapped back into the original timeframe (and so change_tempo /
        # resize compute their factors against the true original tempo).
        self.bpm_og = self.bpm
        self.clip = None  # a freshly loaded score is unclipped
        self.transpose_semitones = 0  # a freshly loaded score is at concert pitch

        # initialize metronome beats from the score; keep an untransformed
        # baseline so tempo/resize changes can rebuild the live grid (and the
        # GuitarHero gridlines) without drift.
        self.beats = self.init_beats()
        self.beats_og = list(self.beats)
        self._beat_offset = 0.0
        self.midi_data.init_metronome(self.beats)

        # init stuff from midi
        self.note_datas = self.midi_data.make_notedatas()
        self.instruments = self.midi_data.instruments

        # reset other shit
        self.displayed_instruments = set(self.instruments.keys())
        self.playing_instruments = set(self.instruments.keys())
        # take the channel MidiData actually assigned (None if no free channel)
        self.metronome_channel = self.midi_data.metronome_channel
        self.active_instrument = self.get_default_instrument()

    def measure_onsets_og(self, channel: int | None = None) -> list[float]:
        """Barline onset times (sec) in the ORIGINAL-tempo timeframe, read off the
        same part Verovio renders. These pair 1:1 (by measure index) with
        Verovio's own measure timemap to anchor the score cursor to the MIDI /
        NoteData timeline (see ui.time.ScoreTimeMap) instead of letting Verovio
        re-derive a drifting timeline of its own.

        Faithful to the MIDI: music21's measure offsets (quarter-lengths) preserve
        the note positions the player + GuitarHero run off; converting at bpm_og
        keeps them in the same frame as the Verovio export (which is pinned to
        bpm_og — see to_musicxml_bytes)."""
        if self.score is None:
            return []
        from music21 import stream
        part = self._part_for_channel(channel) if channel is not None else None
        if part is None:
            parts = list(self.score.parts)
            part = parts[0] if parts else self.score
        measures = list(part.getElementsByClass(stream.Measure))
        if not measures:  # part-less score (rare): recurse to find its measures
            measures = list(part.recurse().getElementsByClass(stream.Measure))
        bpm_og = self.bpm_og or self.bpm or 120
        sec_per_ql = 60.0 / bpm_og
        return [round(float(m.offset) * sec_per_ql, 9) for m in measures]

    def get_default_instrument(self) -> int:
        """Called at the beginning to set a default first instrument (non-metronome)"""
        # default to the first real (non-metronome) instrument channel
        first_ch = next(
            (ch for ch in self.instruments
             if ch != self.metronome_channel),
            0,
        )
        return first_ch

    def to_musicxml_bytes(self, channel: int | None = None) -> bytes:
        """Export the current score to MusicXML bytes for Verovio.

        With `channel`, export ONLY that instrument's part (a single-instrument
        score view); otherwise export the full score. The export is pinned to
        the original tempo (`bpm_og`) so Verovio's timemap stays in the same
        timeframe the playback cursor assumes (see app._score_viewer_time) — a
        reload after a tempo change / Analyze would otherwise desync the cursor.

        This is intentionally the *only* expensive step: it runs on instrument
        change / full-score toggle / score load, never during playback. The
        per-tick cursor path (set_playback_time -> window.timeChanged) is left
        untouched, and a single-part layout makes its page flips cheaper.
        """
        if self.score is None:
            raise ValueError("No score loaded to export.")

        # always stamp our source-of-truth title onto the score so Verovio
        # renders the filename (see _stamp_title / _strip_engraving_credits).
        self._stamp_title(self.score, self.title)

        part = self._part_for_channel(channel) if channel is not None else None

        # fast path: full score still at the original tempo -> export the score
        # directly (cheapest, no deep copy), then clean the credits/title below.
        if part is None and round(self.bpm) == round(self.bpm_og):
            return self._strip_engraving_credits(self._write_musicxml(self.score))

        # otherwise build an isolated copy (single part or full) so we never
        # mutate self.score, then pin its tempo back to bpm_og before exporting.
        import copy
        from music21 import stream

        if part is None:
            source = copy.deepcopy(self.score)
        else:
            source = stream.Score()
            if self.score.metadata is not None:
                source.insert(0, copy.deepcopy(self.score.metadata))
            source.insert(0, copy.deepcopy(part))

        self._pin_tempo(source, self.bpm_og)
        return self._strip_engraving_credits(self._write_musicxml(source))

    @staticmethod
    def _write_musicxml(stream_obj) -> bytes:
        """Write a music21 stream to MusicXML on disk and return its bytes."""
        # music21 writes to disk; capture the produced file path
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "score.musicxml"
            written = stream_obj.write("musicxml", fp=str(out_path))

            # music21 may return the path it actually wrote
            written_path = Path(written) if written else out_path
            return written_path.read_bytes()

    # --- TEMPO STUFF ---
    def change_tempo(self, new_bpm: int, _factor: float=None):
        """Change the tempo of the score to new_bpm. Changes tempo in
            1. midi data (for playback)
            2. music21 score (for exporting and viewing)
            3. notedata (for editing and visualization)
        If _factor is supplied, uses that instead of calculating from new_bpm and self.bpm"""
        factor = _factor if _factor else self.bpm_og / new_bpm
        if new_bpm == self.bpm or self.score is None:
            return # no change needed
        
        # 1. change tempo in midi data (rebuilds messages from the original score)
        self.midi_data.change_tempo(factor)
        # 2. change tempo in music21 score
        for mark in self.score.recurse().getElementsByClass(tempo.MetronomeMark):
            mark.number = round(mark.number * factor)
        # 3. update metadata
        self.bpm = new_bpm
        self.length = self.midi_data.length_og * factor
        # 4. rebuild the beat grid + re-overlay the metronome clicks at the new
        # tempo BEFORE remaking notedatas, so the metronome map / GuitarHero
        # gridlines follow the new tempo (the notes were just rescaled too).
        self._rebuild_beats()
        # 5. remake notedatas (now reflects the refreshed metronome track)
        self.note_datas = self.midi_data.make_notedatas()
        print(f"Tempo changed to {new_bpm} BPM (factor: {factor:.2f}). Score length is now {self.length:.2f} sec.")

    def _rebuild_beats(self):
        """Recompute the live beat grid (`beats`) from the original baseline
        (`beats_og`) for the current tempo and resize offset, so the GuitarHero
        vertical gridlines track score-timing changes.

        Scales the baseline beats by the tempo factor (bpm_og / bpm) — matching
        how the notes are rescaled — then shifts by the resize offset that
        transpose_notes applies to the notes, keeping downbeats lined up with
        the (rescaled, transposed) barlines. Both transforms are absolute
        (computed from the baseline), so repeated tempo changes / resizes don't
        accumulate drift."""
        if not self.beats_og:
            return
        factor = (self.bpm_og / self.bpm) if self.bpm else 1.0
        self.beats = [
            (round(t * factor + self._beat_offset, 9), is_downbeat)
            for t, is_downbeat in self.beats_og
        ]
        # keep the audible metronome click track in lockstep with the grid
        if self.midi_data is not None:
            self.midi_data.set_metronome(self.beats)

    def resize(self, new_length: float):
        """Resize the score to a new length in seconds. Calls change_tempo
        under the hood with new BPM."""
        # length is inversely proportional to bpm, anchored to the original
        # length/tempo: a longer target => a slower (lower) bpm. Let change_tempo
        # recompute the factor from new_bpm so bpm and length stay consistent.
        factor = new_length / self.midi_data.length_og
        new_bpm = round(self.bpm_og / factor)
        self.change_tempo(new_bpm)

    def get_bpm(self) -> float:
        """Get BPM from music21 score. If none, default to 120 BPM."""
        if self.score is None:
            raise ValueError("No score loaded.")

        marks = list(self.score.recurse().getElementsByClass(tempo.MetronomeMark))
        for mark in marks:
            if mark.number is not None:
                return float(mark.number)

        DEFAULT_BPM = 120.0
        print(f"No tempo markings found in score; defaulting to {DEFAULT_BPM} BPM.")
        return DEFAULT_BPM
    
    def init_beats(self) -> list[tuple[float, bool]]:
        """Get the times of all metronome clicks based on the tempo markings in the score.
        
        Returns:
            A list of (time, is_downbeat) tuples. Time is in sec.
        """
        if self.score is None:
            return []
        
        flat = self.score.flatten()
        total_ql = float(flat.highestTime) # total length of the piece in quarter lengths
        sec_per_ql = 60.0 / self.get_bpm()  # seconds per quarter-length (bpm is quarter-based)

        ts_events = [] # list of (time, time_signature) tuples
        for ts in flat.recurse().getElementsByClass(meter.TimeSignature):
            ts_events.append((ts.offset, ts))
        if not ts_events: # default time signature
            ts_events.append((0.0, meter.TimeSignature('4/4')))
        if ts_events[0][0] != 0.0: # ensure first time signature event is at time 0
            ts_events.insert(0, (0.0, ts_events[0][1]))

        eps = 1e-9
        beat_events: list[tuple[float, bool]] = []
        for i, (start_ql, ts) in enumerate(ts_events):
            # quarter length where this time signature ends and the next begins
            # (or the end of the piece)
            end_ql = ts_events[i+1][0] if i+1 < len(ts_events) else total_ql
            beat_ql = float(ts.beatDuration.quarterLength)
            measure_len_ql = float(ts.barDuration.quarterLength)
            beats_per_measure = round(measure_len_ql / beat_ql)

            length_ql = end_ql - start_ql
            n_beats = int((length_ql + eps) // beat_ql)

            # place a beat every beat_ql, converting the ABSOLUTE quarter-length
            # position to seconds. start_ql is in quarter-lengths, NOT seconds —
            # the old code added it straight to a seconds value, which flung every
            # beat after a mid-piece time-signature change far past the end of the
            # song (e.g. caprice24's 2/4->3/4 at ql 192 jumped the grid to ~192s),
            # so the gridlines + metronome clicks vanished from there on.
            for k in range(n_beats):
                beat_time = round((start_ql + k * beat_ql) * sec_per_ql, 9)
                is_downbeat = (k % beats_per_measure == 0)
                beat_events.append((beat_time, is_downbeat))

        return beat_events

    def count_in_beats(self, measures: int = 1) -> list[tuple[float, bool]]:
        """Beat offsets (sec) for a metronome count-in of `measures` measures,
        based on the score's first time signature at the *current* tempo.

        e.g. 4/4 -> 4 quarter-note clicks; 6/8 -> 2 dotted-quarter clicks.
        The first beat of each measure is a downbeat.

        Returns:
            A list of (offset_sec, is_downbeat) tuples, starting at 0.0.
        """
        # find the first time signature, defaulting to 4/4
        ts = None
        if self.score is not None:
            tss = list(self.score.flatten().getElementsByClass(meter.TimeSignature))
            ts = tss[0] if tss else None
        if ts is None:
            ts = meter.TimeSignature('4/4')

        beat_ql = float(ts.beatDuration.quarterLength)
        measure_len_ql = float(ts.barDuration.quarterLength)
        beats_per_measure = round(measure_len_ql / beat_ql)
        sec_per_beat = beat_ql * (60.0 / self.bpm)  # bpm is quarter-note based

        beats: list[tuple[float, bool]] = []
        for k in range(beats_per_measure * measures):
            offset = round(k * sec_per_beat, 9)
            is_downbeat = (k % beats_per_measure == 0)
            beats.append((offset, is_downbeat))
        return beats
    
    def transpose_notes(self, offset_sec: float):
        """Shift all score notes so the piece starts `offset_sec` seconds in.

        Sets each note's absolute time from its *untransposed baseline*
        (`base_start_time`/`base_end_time`) rather than incrementing, so calling
        this repeatedly with the same offset is idempotent — repeated resizes no
        longer drift the score to the right. Also rebuilds each NoteData's `data`
        dict keys and `times` list so they stay in sync with the notes' new
        start_times (otherwise time-indexed lookups read stale keys)."""
        for notedata in self.note_datas.values():
            new_data = {}
            for note in notedata.data.values():
                note.start_time = note.base_start_time + offset_sec
                note.end_time = note.base_end_time + offset_sec
                new_data[note.start_time] = note
            notedata.data = new_data
            notedata.times = sorted(new_data.keys())

        # shift the beat grid by the same offset so the GuitarHero gridlines stay
        # aligned with the transposed notes (idempotent: rebuilt from baseline).
        self._beat_offset = offset_sec
        self._rebuild_beats()

    # --- PITCH TRANSPOSITION (NB: distinct from transpose_notes, which is TIME) ---
    def transpose(self, semitones: int):
        """Transpose the ENTIRE score up/down by `semitones` half steps, keeping
        all three representations in sync so playback, the GuitarHero piano-roll,
        and the Verovio sheet music all reflect the new pitches:
            1. midi_data  — the messages that get played
            2. score      — the music21 score the MusicXML view is exported from
            3. note_datas — the per-instrument Note pitches (GuitarHero + editing)

        This is a *pitch-only* shift: note timing, the metronome, and the clip
        (stored as note INDICES) are all untouched. Applied incrementally and
        accumulated in `transpose_semitones`, so callers can hand it a delta and
        repeated calls compose correctly (a delta of 0 is a no-op)."""
        if self.score is None or not semitones:
            return

        # 1. MIDI playback (shifts the shared note-message pitch baseline)
        self.midi_data.transpose(semitones)
        # 2. music21 score (drives the MusicXML pushed to Verovio); int == semitones
        self.score.transpose(semitones, inPlace=True)
        # 3. live NoteData pitches — shift in place so the current views update
        #    without a full rebuild. Skip the metronome channel + unvoiced (-1).
        for channel, nd in self.note_datas.items():
            if channel == self.metronome_channel:
                continue
            for note in nd.data.values():
                note.midi_num = [
                    int(max(0, min(127, m + semitones))) if m != -1 else -1
                    for m in note.midi_num
                ]

        self.transpose_semitones += semitones

    def first_note_midi(self, channel: int | None = None) -> int | None:
        """The MIDI number of the first VOICED note of the given instrument (the
        active one by default) — the reference pitch the SettingsWidget's
        transpose input anchors to. None if the instrument has no voiced notes."""
        channel = self.active_instrument if channel is None else channel
        nd = self.note_datas.get(channel)
        if not nd or not nd.times:
            return None
        for t in nd.times:
            midi = nd.data[t].midi_num
            if midi and midi[0] != -1:
                return int(round(midi[0]))
        return None

    # --- score METADATA MANAGEMENT ---
    def set_title(self, title: str):
        """Update display title -> included in next render's metadata."""
        self.title = title

    @staticmethod
    def _stamp_title(source, title: str):
        """Force `source`'s metadata title to `title` so Verovio renders it as
        the score title. music21 writes both <work-title> and <movement-title>;
        we set both to the same value for a single, consistent heading."""
        if not title:
            return
        from music21 import metadata
        if source.metadata is None:
            source.insert(0, metadata.Metadata())
        source.metadata.title = title
        source.metadata.movementName = title

    @staticmethod
    def _strip_engraving_credits(xml: bytes) -> bytes:
        """Drop the absolutely-positioned <credit> blocks MuseScore exports (laid
        out for a full page, they'd be Verovio's page header but get clipped by
        our single-system page trimming) plus the placeholder composer creator.
        With those gone, Verovio generates a clean header from the encoded
        work/movement title — i.e. it always renders the filename."""
        import re
        text = xml.decode("utf-8")
        text = re.sub(r"<credit\b[^>]*>.*?</credit>", "", text, flags=re.DOTALL)
        text = re.sub(r'<creator type="composer">.*?</creator>', "", text, flags=re.DOTALL)
        return text.encode("utf-8")

    def _part_for_channel(self, channel: int):
        """Resolve a MIDI instrument channel to its music21 Part, or None if it
        can't be resolved (caller falls back to the full score).
        """
        if self.score is None:
            return None
        real_channels = [ch for ch in self.instruments if ch != self.metronome_channel]
        if channel not in real_channels:
            return None
        idx = real_channels.index(channel)
        parts = list(self.score.parts)
        if 0 <= idx < len(parts):
            return parts[idx]
        return None

    @staticmethod
    def _pin_tempo(source, bpm: float):
        """Force `source`'s tempo to `bpm` (flattening to a single mark) so its
        Verovio timemap matches the original-tempo timeframe."""
        marks = list(source.recurse().getElementsByClass(tempo.MetronomeMark))
        if marks:
            for m in marks:
                m.number = bpm
        else:
            parts = list(source.parts) if hasattr(source, "parts") else []
            target = parts[0] if parts else source
            target.insert(0, tempo.MetronomeMark(number=bpm))