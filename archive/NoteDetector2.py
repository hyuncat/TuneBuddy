import numpy as np
from app_logic.NoteData import Note, NoteData
from PyQt6.QtCore import pyqtSignal, QObject
import threading

from app_logic.user.ds.UserData import UserData

class NoteDetector(QObject):
    note_detected = pyqtSignal(float)
    def __init__(self, w: int=30, hop: int=15, pitch_thresh: float=0.6, slope_thresh: float=1, parent: QObject|None=None):
        """initialize the note detection algorithm parameters"""
        super().__init__(parent)

        # algorithm params
        self.w = w
        self.hop = hop
        self.pitch_thresh = pitch_thresh
        self.slope_thresh = slope_thresh
        
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
        # first pitch p_0 = None
        prev_pitch = None
        while not self.stop_event.is_set():
            try:
                x = self.user_data.p2n_queue.pop(
                self.window_size, self.hop_size)
            except Exception as e:
                print(f"[NoteDetector] frame skipped due to error: {e}")
                continue

    def stop(self):
        if self.nda_thread and self.nda_thread.is_alive():
            self.stop_event.set()
            self.nda_thread.join() # pause the main thread until recording thread recognizes the stop event

    # methods
    def get_window_pitch(self, window, method='median') -> float:
        """returns the pitch of a window"""
        return np.median(window)

    def get_window_slope(self, window, start_time, end_time) -> tuple[float, float]:
        """computes the slope of a window of pitches with least squares regression"""
        x = np.linspace(0, end_time-start_time, len(window))

        # get the slope + intercept
        A = np.vstack([x, np.ones_like(x)]).T
        slope, intercept = np.linalg.lstsq(A, window, rcond=None)[0]

        # now use it to reconstruct what the pitches should be
        predicted_window = slope*x + intercept

        # and remove all pitches way off from the prediction
        VARIANCE_THRESHOLD = 3
        residuals = np.abs(window - predicted_window)
        window_2 = np.where(residuals < VARIANCE_THRESHOLD, window, np.nan)
        window_2 = window_2[~np.isnan(window_2)]

        # again! slope + intercept for real this time 😎
        A = np.vstack([x, np.ones_like(x)]).T
        slope, intercept = np.linalg.lstsq(A, window, rcond=None)[0]
        return slope, intercept
        

    def detect_notes(self, pitches, rank: int=0, w: int=30, hop: int=15, pitch_thresh: float=0.6, slope_thresh: float=1):
        """
        Detect different-enough pitches (midi_numbers) with a rolling median on a window, size
        = [w]. Compare the current median window pitch to the next one, and keep 
        track of when the difference exceeds [threshold].

        Args:
            start_time: start time (sec) in recording to detect pitches from
            end_time: when to stop detecting pitches, None defaults to end
            rank: probability path to detect, 0=most probable, 3=least probable
            w: size of each window for comparisons (median, slope)
            pitch_thresh: min pitch median difference to count the window as a new note
            slope_thresh: min slope flatness to count a collection of pitches as a 'note'
        """

        note_string = NoteData()
        self.w = w
        self.hop = hop
        # === Finding first note ===
        # keep going until we find a flat enough slope to call it
        # the beginning of our first note

        i_0 = 0 # index of our first note
        pitch_0 = None
        for i in range(0, len(pitches) - self.w-1, self.hop):
            window = np.array([p.midi_num for p in pitches[i : i+w]])

            # compute median + slope
            pitch = self.get_window_pitch(window)
            slope, intercept = self.get_window_slope(
                window, 
                start_time=pitches[i].time, 
                end_time=pitches[i+w].time
            )

            if abs(slope) < slope_thresh:
                # found our first note, break
                i_0 = i
                pitch_0 = pitch
                break

        # === FILLING IN NOTE_STRING ===
        # iteration variables
        last_note_start = i_0
        last_note_pitch = pitch_0

        for i in range(i_0, len(pitches) - self.w-1, self.hop):
            # get current window
            window = np.array([p.midi_num for p in pitches[i : i+w]])

            # compute median + slope
            pitch = self.get_window_pitch(window)
            slope, intercept = self.get_window_slope(
                window, 
                start_time=pitches[i].time, 
                end_time=pitches[i+w].time
            )

            # print(f"time = {pitches[i].time:.2f} | pitch = {pitch}, slope = {slope}")

            # ignore the section if it's not flat enough
            if abs(slope) > slope_thresh:
                continue
            
            # check if it's a significantly different pitch than the previous
            if abs(pitch - last_note_pitch) > pitch_thresh:
                # get median volume
                volumes = [p.volume for p in pitches[last_note_start : i+w]]
                med_volume = np.median(volumes)
                
                # create new note with data and append to note_string
                new_note = Note(
                    pitch=last_note_pitch,
                    start=pitches[last_note_start].time, 
                    end=pitches[i].time,
                    volume=med_volume
                )

                # add a bunch of note possibilities
                N_RANKS = 3
                new_notes = [new_note]
                for j in range(1, N_RANKS):
                    pitches_i = self.recording.pitches.get_pitches(
                        start_time=pitches[last_note_start].time, 
                        end_time=pitches[i].time,
                        rank=j
                    )
                    median_pitch = np.median([p.midi_num for p in pitches_i])
                    note_i = Note(
                        pitch=median_pitch,
                        start=pitches[last_note_start].time, 
                        end=pitches[i].time,
                        volume=med_volume
                    )
                    new_notes.append(note_i)

                note_string.append_note(new_notes)

                last_note_pitch = pitch
                last_note_start = i # update the start of our new note

        return note_string