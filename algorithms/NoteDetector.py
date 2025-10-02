import numpy as np
from app_logic.NoteData import Note, NoteData
from app_logic.user.ds.PitchData import Pitch
from PyQt6.QtCore import pyqtSignal, QObject
import threading

from app_logic.user.ds.UserData import UserData

class NoteDetector(QObject):
    note_detected = pyqtSignal(float)
    
    def __init__(self, w: int=30, hop: int=25, pitch_thresh: float=0.75, slope_thresh: float=1.5, parent: QObject|None=None):
        """initialize the note detection algorithm parameters"""
        super().__init__(parent)

        # algorithm params
        self.w = w
        self.hop = hop
        self.PITCH_THRESH = pitch_thresh
        self.SLOPE_THRESH = slope_thresh
        
        self.UNVOICED_PROP = 0.5 # if more than 50% of pitches are unvoiced
        self.SENSITIVITY = 0.9 # unvoiced pitches have unv_prob > sens
        
        # threading variables
        self.nda_thread: threading.Thread = None
        self.stop_event = threading.Event()

    def init_user_data(self, user_data: UserData):
        self.user_data = user_data

    def run(self, start_time: float=None):
        self.stop()
        self.stop_event.clear()
        self.user_data.p2n_queue.init_start_time(start_time)
        self.nda_thread = threading.Thread(
            target=self._run, daemon=True
        )
        self.nda_thread.start()

    def _run(self) -> None:
        """the note detection algorithm for real time processing.

        an onset-based approach, where a window is an *onset* if
            - it's flat enough and voiced
            - or if it's mostly unvoiced
        
        and if a window is an onset,
        we compare it to the last valid onset
            - if it's different, it's a new note
            - if it's the same, it's not a new note
        """
        prev_note = None
        prev_time = None
        i = 0
        while not self.stop_event.is_set():
            try:
                x, t = self.user_data.p2n_queue.pop(self.w, self.hop)
                if x is None or t < 0: # if invalid data read, skip frame
                    continue

                is_flat, is_unv, med_pitch = self.handle_window(x)

                print(f"this window: is_flat({is_flat}), is_unv({is_unv}), med_pitch({med_pitch}), t({t})")

                # --- finding the first note phase ---
                if prev_note is None:
                    prev_note = -1 if is_unv else med_pitch
                    prev_time = t
                    continue

                # --- the second note and beyond ---
                if abs(prev_note - med_pitch) < self.PITCH_THRESH:
                    continue

                # ignore if the current window is unvoiced or flat
                prev_time = t
                if not is_flat and not is_unv:
                    # but still advance prev_time so we stay contiguous
                    # prev_note = -1 if is_unv else med_pitch
                    continue

                # ---> if we reach here, we have a NEW NOTE!
                print(f"NEW NOTE! pitch={prev_note}, start={prev_time}, end={t}")
                n = Note(
                    i=i,
                    start_time=prev_time, 
                    end_time=t,
                    midi_num=prev_note
                )
                self.user_data.note_data.write_note(n)
                i += 1

                # update iteration variables
                prev_note = -1 if is_unv else med_pitch
                self.note_detected.emit(n.start_time)

            except Exception as e:
                print(f"[NoteDetector] frame skipped due to error: {e}")
                continue

    def stop(self):
        if self.nda_thread and self.nda_thread.is_alive():
            self.stop_event.set()
            self.nda_thread.join() # pause the main thread until recording thread recognizes the stop event

    def get_slope(self, pitches: list[Pitch]):
        """get slope of all voiced pitches in the window"""
        clean = [p for p in pitches if p is not None]
        if not clean:
            return 0.0, 0.0

        start_time = clean[0].time
        end_time   = clean[-1].time

        # boolean mask
        mask  = np.array([p.unvoiced_prob < self.SENSITIVITY for p in clean])
        # mask = np.array([p.unvoiced_prob<self.SENSITIVITY for p in pitches])

        # select only voiced x and y values
        all_x = np.linspace(0, end_time-start_time, len(clean))
        # all_x = np.linspace(0, end_time-start_time, len(pitches))
        x_voiced = all_x[mask]
        y_voiced = np.array([p.candidates[0][0] for p, m in zip(clean, mask) if m])
        # y_voiced = np.array([p.candidates[0][0] for p, m in zip(pitches, mask) if m])

        if x_voiced.size == 0:
            return 0.0, 0.0

        # get slope + intercept of only voiced pitches
        A = np.vstack([x_voiced, np.ones_like(x_voiced)]).T
        slope, intercept = np.linalg.lstsq(A, y_voiced, rcond=None)[0]

        return slope, intercept
    
    def is_unvoiced(self, unvoiced_probs: list[float]) -> bool:
        """returns whether the window is"""
        arr = [p > self.SENSITIVITY for p in unvoiced_probs]
        if sum(arr) > self.UNVOICED_PROP*len(arr):
            return True
        return False
    
    def handle_window(self, pitches: list[Pitch]):
        """
        returns key results about the window used for note processing
            (1) is_flat, (2) is_unv, (3) median_pitch, (4) start_time
        """
        unvoiced_probs = [
            p.unvoiced_prob if p is not None else 1.0 for p in pitches
        ]
        slope, _ = self.get_slope(pitches) 

        # key results
        is_flat = slope < self.SLOPE_THRESH
        is_unv = self.is_unvoiced(unvoiced_probs)
        # voiced_pitches = [
        #     p.candidates[0][0] for p in pitches
        #     if (p is not None and p.unvoiced_prob < self.SENSITIVITY)
        # ]
        # if voiced_pitches:
        #     median_pitch = float(np.median(voiced_pitches))
        # else:
        #     median_pitch = -1
        v = [[] for _ in range(3)]

        for p in pitches:
            for i in range(3):
                try:
                    v[i].append(p.candidates[i][0])
                except Exception as _:
                    v[i].append(None)

        voiced = [[] for _ in range(3)]
        for i in range(3):
            voiced[i] = [v for v in v[i] if v is not None]

        med_pitches = [None for _ in range(3)]
        for i in range(3):
            med_pitches[i] = float(np.median(voiced[i])) if voiced[i] else -1
        
        return is_flat, is_unv, med_pitches
    
    def detect_notes(self, pitches: list[Pitch]) -> NoteData:
        """writes all notes completely offline"""
        nd = NoteData()
        prev_note = None
        prev_time = None
        prev_good_time = None
        i = 0
        print("Starting note detection...")
        for i in range(0, len(pitches)-self.w-1, self.hop):
            # print(f"\rProcessing frame {i+1}/{len(pitches)}", end='')
            x = pitches[i:i+self.w]
            clean = [p for p in x if p is not None]
            if len(clean) == 0:
                continue
            t = clean[0].time
            is_flat, is_unv, med_pitches = self.handle_window(x)
            # med_pitch = med_pitches[0]
            print(
                f"  [ window({t})\n"
                f"  [ is_flat({is_flat}), is_unv({is_unv}),\n"
                f"  [ med_pitch({med_pitches[0]})"
            )

            if prev_note is None:
                if is_unv:
                    prev_note = [-1, -1, -1]
                    prev_time = t
                elif is_flat:
                    prev_note = med_pitches
                    prev_time = t
            else:
                if abs(prev_note[0] - med_pitches[0]) > self.PITCH_THRESH:
                    print(f"---\nNEW NOTE!\n"
                          f" > pitch={prev_note},\n" 
                          f" > start={prev_time},\n"
                          f" > end={t}")
                    if not is_flat and not is_unv:
                        continue
                    n = Note(
                        i=i,
                        start_time=prev_time, 
                        end_time=t,
                        midi_num=prev_note
                    )
                    nd.write_note(n)
                    # update iteration variables
                    prev_note = [-1, -1, -1] if is_unv else med_pitches
                    prev_time = t
                    i += 1
                prev_good_time = t

        # write the last note! :,)
        n = Note(
            i=i,
            start_time=prev_time, 
            end_time=prev_good_time,
            midi_num=prev_note
        )
        nd.write_note(n)
        print('\nNote detection: Done!')
        return nd