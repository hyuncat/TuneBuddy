import numpy as np
import ruptures as rpt
from PyQt6.QtCore import QObject

from app_logic.NoteData import Note, NoteData
from app_logic.user.ds.PitchData import Pitch
from app_logic.user.ds.Recording import Recording
from algorithms.Config import Config

class TransitionDetector:
    def __init__(self, recording: Recording):
        self.recording = recording
        self.config = recording.config

    def get_slope(self, pitches: list[Pitch]):
        """get slope of all voiced pitches in the window"""
        # select only voiced x and y values
        mask  = np.array([p.unvoiced_prob < self.config.unv_thresh if p else False for p in pitches]) # boolean mask

        all_x = np.linspace(start=0, stop=len(pitches), num=len(pitches))
        x_voiced = all_x[mask]
        y_voiced = np.array([p.value for p, m in zip(pitches, mask) if m])

        if x_voiced.size == 0:
            return 0.0, 0.0

        # get slope + intercept of only voiced pitches
        A = np.vstack([x_voiced, np.ones_like(x_voiced)]).T
        slope, intercept = np.linalg.lstsq(A, y_voiced, rcond=None)[0]

        return slope, intercept
    
    def detect_transitions(self, pitches: list[Pitch]):
        """Marks pitches as transitions in-place if slope of the window is too-steep
        (ie, moves up/down >0.5 semitones per window)
        """
        WINDOW = 9
        HOP = 7
        SLOPE_THRESH = 0.5 / WINDOW
        self.clear_transitions(pitches)
        if len(pitches) < WINDOW:
            return

        frames = np.lib.stride_tricks.sliding_window_view(pitches, window_shape=WINDOW)[::HOP]
        
        for frame in frames:
            if not frame[0]:
                continue
            slope, _ = self.get_slope(frame)
            if abs(slope) > SLOPE_THRESH:
                for p in frame:
                    if p:
                        p.is_transition = True

    @staticmethod
    def clear_transitions(pitches: list[Pitch]):
        """Remove derived transition flags, including ones restored from cache."""
        for pitch in pitches:
            if pitch:
                pitch.is_transition = False


class NoteDetector(QObject):

    # A marginal PELT pitch split is kept only when an independently detected
    # spectral onset lands close to it. Strong pitch steps remain self-evident.
    AMBIGUOUS_PITCH_FACTOR = 1.5
    ONSET_BOUNDARY_TOLERANCE_SEC = 0.10

    def __init__(self, recording: Recording, config: Config=None, parent: QObject=None) -> None:
        super().__init__(parent)

        # algorithm params
        self.recording = recording
        self.config = config if config else recording.config
        self.PITCH_THRESH = self.config.pitch_thresh
        self.UNV_THRESH = self.config.unv_thresh # unvoiced pitches have unv_prob > sens
        self.verbose = self.config.verbose
        self.MIN_NOTE_FACTOR = 0.6 # min note detected by note detector is the score's min note length * this

    def update_config(self, config: Config):
        """update the config and all relevant parameters"""
        self.config = config
        self.PITCH_THRESH = self.config.pitch_thresh
        self.UNV_THRESH = self.config.unv_thresh # unvoiced pitches have unv_prob > sens
        self.verbose = self.config.verbose

    def get_pitch_runs(self, pitches: list[Pitch]) -> list[list[Pitch]]:
        """Returns a list of consecutive voiced pitch runs.
        Splits notes when an unvoiced run spans Config.min_gap_factor of the
        score-derived minimum note length.
        """
        all_runs = []
        run = []
        MIN_RUN_FRAMES = max(1, round(0.5 * self.config.get_min_note_length(type="frames"))) # in frames
        MIN_GAP_FRAMES = self.config.min_note_pitch_frames(
            factor=self.config.min_gap_factor,
        )
        n_gap_frames = 0

        def append_run():
            """appends the current run to all_runs if long enough
            then resets the current run to empty"""
            nonlocal run
            # only add run if long enough, else it's discarded
            if len(run) >= MIN_RUN_FRAMES:
                all_runs.append(run)
            run = []

        for p in pitches:
            is_voiced = p and p.value != -1 and p.unvoiced_prob < self.UNV_THRESH and not p.is_transition
            if is_voiced:
                if run and n_gap_frames >= MIN_GAP_FRAMES:
                    append_run()
                run.append(p)
                n_gap_frames = 0
            elif run:
                n_gap_frames += 1
                if n_gap_frames >= MIN_GAP_FRAMES:
                    append_run()
                    n_gap_frames = 0

        # cleanup at end
        if run:
            append_run()
        return all_runs

    def detect_notes(self, pitches: list[Pitch], model: str="l2") -> NoteData:
        """Offline note detection - splits into runs of voiced pitches, then uses
        PELT to find breakpoints between notes. Merges notes that are too close together
        within pitch_thresh semitones."""
        if self.verbose:
            print(f"[NoteDetector] detecting notes (model={model})", flush=True)

        pitch_runs = self.get_pitch_runs(pitches)

        if not pitch_runs:
            if self.verbose:
                print("[NoteDetector] done: 0 runs, 0 note(s)", flush=True)
            return NoteData()

        MIN_FRAMES = max(1, round(self.MIN_NOTE_FACTOR * self.config.get_min_note_length(type="frames")))
        # L2 cost drop for a pitch_thresh shift across two min-size segments
        PELT_PENALTY = 0.5 * MIN_FRAMES * (self.config.pitch_thresh ** 2)

        nd = NoteData()
        note_idx = 0  # global across runs so note.id == sorted position
        for pitches in pitch_runs:
            # get list of breakpoints (note-end indices)
            run = [p for p in pitches if p]
            signal = np.asarray([p.value for p in run], dtype=float).reshape(-1, 1)
            if len(run) < 2 * MIN_FRAMES: # too short to split this segment into > 1 note
                breakpoints = [len(run)]
            else:
                try:
                    breakpoints = (rpt.Pelt(model=model, min_size=MIN_FRAMES, jump=self.config.h2)
                    .fit(signal).predict(pen=PELT_PENALTY))
                except Exception as e:
                    breakpoints = [len(run)]
            
            # retrace the notes from the breakpoints
            prev_idx = 0
            is_first_segment = True 
            for pt in breakpoints:
                end_idx = min(int(pt), len(run))
                if end_idx <= prev_idx:
                    continue
                midi_num = float(np.median(signal[prev_idx:end_idx]))
                start_time = self.bisect_transition(run, prev_idx)
                end_time = self.bisect_transition(run, end_idx)

                if end_time <= start_time:
                    prev_idx = end_idx
                    continue

                last_note = (
                    nd.read_note(i=len(nd.times)-1)
                    if nd.times
                    else None
                )
                # Merge a definite same-pitch segment, plus a marginal pitch
                # split that has no independent spectral-onset support. The
                # latter is the common vibrato/settling failure: PELT finds two
                # levels within one sustained note, but there was no new attack.
                pitch_change = (
                    abs(last_note.midi_num[0] - midi_num)
                    if last_note is not None
                    else float("inf")
                )
                merge = (
                    last_note is not None
                    and not is_first_segment
                    and (
                        pitch_change < self.PITCH_THRESH
                        or (
                            pitch_change < self.AMBIGUOUS_PITCH_FACTOR * self.PITCH_THRESH
                            and not self._has_onset_near(start_time)
                        )
                    )
                )
                if merge:
                    last_note.end_time = end_time
                    combined = self.recording.pitch_data.read(
                        start_time=last_note.start_time,
                        end_time=end_time,
                        clean=True,
                        include_transitions=False,
                    )
                    if combined:
                        last_note.midi_num = [
                            float(np.median([p.value for p in combined]))
                        ]
                else:
                    nd.write_note(
                        Note(
                            i=note_idx,
                            start_time=start_time,
                            end_time=end_time,
                            midi_num=[midi_num], #todo: make this a float not a list
                        )
                    )
                    note_idx += 1
                prev_idx = end_idx
                is_first_segment = False

        return nd

    def _has_onset_near(self, boundary_time: float) -> bool:
        """Whether a high-confidence spectral candidate supports a PELT split."""
        onset_data = getattr(self.recording, "onset_data", None)
        if onset_data is None or not len(onset_data):
            return False
        tol = self.ONSET_BOUNDARY_TOLERANCE_SEC
        return bool(len(onset_data.read(
            start_time=boundary_time - tol,
            end_time=boundary_time + tol,
        )))

    @staticmethod
    def bisect_transition(run: list[Pitch], i: int) -> float:
        """midpoint across the slide/gap dropped from the run, so notes meet mid-transition"""
        if i <= 0:
            return run[0].time
        if i >= len(run):
            return run[-1].time
        return 0.5 * (run[i - 1].time + run[i].time)

    def refine_with_onsets(self, note_data: NoteData, onset_times: list[float]) -> NoteData:
        """Refines the note boundaries using onset times from a separate onset detector."""
        if note_data is None or not note_data.times or onset_times is None or len(onset_times) == 0:
            return note_data

        MIN_NOTE_SEC = self.config.get_min_note_length() * self.MIN_NOTE_FACTOR
        splits: dict[int, list[float]] = {}
        for onset_time in sorted(onset_times):
            t = float(onset_time)
            if not np.isfinite(t):
                continue

            note = note_data.read_current_note(t) or note_data.read_note(start_time=t)
            if note is None or not note.start_time < t < note.end_time:
                continue

            note_splits = splits.get(note.id, [])
            last_boundary = note_splits[-1] if note_splits else note.start_time
            # ignore the onset if it creates splits that are too short
            if t-last_boundary < MIN_NOTE_SEC or note.end_time-t < MIN_NOTE_SEC:
                continue

            splits.setdefault(note.id, []).append(t)

        if not splits:
            return note_data

        # rebuild the note data with all splits applied
        refined = NoteData()
        for note in note_data.read(i=0, j=len(note_data.times)):
            note_splits = splits.get(note.id)
            if not note_splits:
                refined.write_note(Note(
                    i=len(refined.times),
                    start_time=note.start_time,
                    end_time=note.end_time,
                    midi_num=list(note.midi_num),
                    velocity=note.velocity,
                    instrument=note.instrument,
                ))
                continue

            boundaries = [note.start_time, *note_splits, note.end_time]
            for start_time, end_time in zip(boundaries, boundaries[1:]):
                if end_time <= start_time:
                    continue
                pitches = self.recording.pitch_data.read(
                    start_time=start_time,
                    end_time=end_time,
                    clean=True,
                    include_transitions=False,
                )
                midi_num = [float(np.median([p.value for p in pitches]))] if pitches else list(note.midi_num)
                refined.write_note(Note(
                    i=len(refined.times),
                    start_time=start_time,
                    end_time=end_time,
                    midi_num=midi_num,
                    velocity=note.velocity,
                    instrument=note.instrument,
                ))

        return refined
