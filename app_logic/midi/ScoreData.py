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
        # the score's display title (source of truth lives in the RecordingTree;
        # defaults to the loaded file's stem). Always stamped onto the metadata
        # so Verovio renders it — see to_musicxml_bytes / _stamp_title.
        self.title: str = ""

        # instrument selection
        self.instruments: dict[int, int] = {} # {channel: program_number}
        self.active_instrument: int = 0 # channel number (TODO: decouple this)
        self.displayed_instruments: set[int] = set() # channels to display
        self.playing_instruments: set[int] = set() # channels to play
        self.metronome_channel: int = None
        # self.metronome_on: bool = True
        # clipping
        self.bounds: tuple[float, float] = (0.0, 0.0) # (start_time, end_time)

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
        self.bounds = (0.0, self.length)

        # initialize metronome beats from the score
        self.beats = self.init_beats()
        self.midi_data.init_metronome(self.beats)

        # init stuff from midi
        self.note_datas = self.midi_data.make_notedatas()
        self.instruments = self.midi_data.instruments

        # reset other shit
        self.active_instrument = 0
        self.displayed_instruments = set(self.instruments.keys())
        self.playing_instruments = set(self.instruments.keys())
        self.metronome_channel = len(self.instruments.keys()) - 1 # last channel

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

    def set_title(self, title: str):
        """Update the score's display title (the RecordingTree is the source of
        truth and calls this on rename). The next render stamps it onto the
        metadata so Verovio shows it; callers re-render the score viewer."""
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

        Maps by position among the real (non-metronome) channels, which matches
        the order music21 writes parts out to MIDI. This is the same channel-
        ordering assumption the rest of the app already relies on.
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
        
        # 1. change tempo in midi data
        self.midi_data.change_tempo(factor)
        # 2. change tempo in music21 score
        for mark in self.score.recurse().getElementsByClass(tempo.MetronomeMark):
            mark.number = round(mark.number * factor)
        # 3. remake notedatas
        self.note_datas = self.midi_data.make_notedatas()
        # new_notedatas = {}
        # for channel, notedata in self.note_datas.items():
        #     new_notedata = NoteData()
        #     new_notedata.times = [t * factor for t in notedata.times]
        #     new_notedata.data = {t * factor: n for t, n in notedata.data.items()}
        #     for n in new_notedata.data.values():
        #         n.start_time = n.start_time * factor
        #         n.end_time = n.end_time * factor
        #     new_notedatas[channel] = new_notedata
        # self.note_datas = new_notedatas
        # 4. update metadata
        self.bpm = new_bpm
        self.length = self.midi_data.length_og * factor
        print(f"Tempo changed to {new_bpm} BPM (factor: {factor:.2f}). Score length is now {self.length:.2f} sec.")

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
        sec_per_beat = 60.0 / self.get_bpm()

        ts_events = [] # list of (time, time_signature) tuples
        for ts in flat.recurse().getElementsByClass(meter.TimeSignature):
            ts_events.append((ts.offset, ts))
        if not ts_events: # default time signature
            ts_events.append((0.0, meter.TimeSignature('4/4')))
        if ts_events[0][0] != 0.0: # ensure first time signature event is at time 0
            ts_events.insert(0, (0.0, ts_events[0][1]))

        eps = 1e-9
        beat_events: list[tuple[float, bool]] = [] # in quarterlengths
        elapsed_time = 0.0 # sec
        for i, (start_ql, ts) in enumerate(ts_events):
            # quarter length at place where this time signature ends
            # and the next one begins (or end of piece)
            end_ql = ts_events[i+1][0] if i+1 < len(ts_events) else total_ql
            beat_ql = float(ts.beatDuration.quarterLength)
            measure_len_ql = float(ts.barDuration.quarterLength)
            beats_per_measure = round(measure_len_ql / beat_ql)

            length_ql = end_ql - start_ql
            n_beats = int((length_ql + eps) // beat_ql)

            # place beats at start_ql, start_ql + beat_ql, ..., up to end_ql
            for k in range(n_beats):
                beat_time = round(start_ql + k * sec_per_beat, 9)
                is_downbeat = (k % beats_per_measure == 0)
                beat_events.append((beat_time, is_downbeat))

            # if partial leftover region, don't add any beats beyond end_ql
            elapsed_time += (length_ql / beat_ql) * sec_per_beat

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