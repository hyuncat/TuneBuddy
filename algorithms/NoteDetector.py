import numpy as np
from app_logic.NoteData import Note, NoteData
from app_logic.user.ds.PitchData import Pitch
from PyQt6.QtCore import pyqtSignal, QObject
import threading

from app_logic.user.ds.UserData import UserData

class NoteDetector(QObject):
    note_detected = pyqtSignal(float)
    def __init__(self, w: int=30, hop: int=25, pitch_thresh: float=1, slope_thresh: float=1, parent: QObject|None=None):
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
        while not self.stop_event.is_set():
            try:
                x = self.user_data.p2n_queue.pop(self.w, self.hop)
                is_flat, is_unv, med_pitch, t = self.handle_window(x)

                print(f"this window: is_flat({is_flat}), is_unv({is_unv}), med_pitch({med_pitch}), t({t})")

                if prev_note is None:
                    if is_unv:
                        prev_note = -1
                        prev_time = t
                    elif is_flat:
                        prev_note = med_pitch
                        prev_time = t
                else:
                    if abs(prev_note - med_pitch) > self.PITCH_THRESH:
                        if (not is_flat and not is_unv) or t==-1:
                            continue
                        print(f"NEW NOTE! pitch={prev_note}, start={prev_time}, end={t}")
                        n = Note(
                            start_time=prev_time, 
                            end_time=t,
                            midi_num=prev_note
                        )
                        self.user_data.note_data.write_note(n)
                        # update iteration variables
                        prev_note = -1 if is_unv else med_pitch
                        prev_time = t
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
        # start_time = pitches[0].time
        # end_time = pitches[-1].time

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
        clean = [p for p in pitches if p is not None]
        unvoiced_probs = [
            p.unvoiced_prob if p is not None else 1.0
            for p in pitches
        ]
        # unvoiced_probs = [p.unvoiced_prob for p in pitches]
        slope, _ = self.get_slope(pitches) 

        # key results
        is_flat = slope < self.SLOPE_THRESH
        is_unv = self.is_unvoiced(unvoiced_probs)
        voiced_pitches = [
            p.candidates[0][0]
            for p in pitches
            if (p is not None and p.unvoiced_prob < self.SENSITIVITY)
        ]
        # voiced_pitches = [p.candidates[0][0] for p in pitches if p.unvoiced_prob<self.SENSITIVITY]

        if voiced_pitches:
            median_pitch = float(np.median(voiced_pitches))
        else:
            median_pitch = -1
            
        start_time = -1 if len(clean)==0 else clean[0].time

        return is_flat, is_unv, median_pitch, start_time
    
    def detect_notes(self, pitches: list[Pitch]) -> NoteData:
        """writes all notes completely offline"""
        nd = NoteData()
        prev_note = None
        prev_time = None
        print("Starting note detection...")
        for i in range(0, len(pitches) - self.w-1, self.hop):
            # print(f"\rProcessing frame {i+1}/{len(pitches)}", end='')
            x = pitches[i:i+self.w]
            is_flat, is_unv, med_pitch, t = self.handle_window(x)
            print(f"this window: is_flat({is_flat}), is_unv({is_unv}), med_pitch({med_pitch}), t({t})")

            if prev_note is None:
                if is_unv:
                    prev_note = -1
                    prev_time = t
                elif is_flat:
                    prev_note = med_pitch
                    prev_time = t
            else:
                if abs(prev_note - med_pitch) > self.PITCH_THRESH:
                    print(f"NEW NOTE! pitch={prev_note}, start={prev_time}, end={t}")
                    if not is_flat and not is_unv:
                        continue
                    n = Note(
                        start_time=prev_time, 
                        end_time=t,
                        midi_num=prev_note
                    )
                    nd.write_note(n)
                    # update iteration variables
                    prev_note = -1 if is_unv else med_pitch
                    prev_time = t

        print('\nNote detection: Done!')
        return nd