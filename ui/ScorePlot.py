from PyQt6.QtWidgets import QWidget, QVBoxLayout
import pyqtgraph as pg
import numpy as np

from app_logic.midi.MidiData import MidiData
from app_logic.user.ds.UserData import UserData
from app_logic.Alignment import Alignment

class MidiBackground(pg.ImageItem):
    """
    A custom ImageItem to display a fixed MIDI background
    Immutable background stripes (0..127 MIDI), built once!
    """
    def __init__(self):
        super().__init__(axisOrder='row-major')
        self.setZValue(-1)  # set as global -1 position
        # color mappings
        self.LETTER_RGB = {
            'A': (230,  60,  60),  # red
            'B': (255, 150,  40),  # orange
            'C': (245, 220,  70),  # yellow
            'D': ( 70, 200,  90),  # green
            'E': ( 70, 140, 240),  # blue
            'F': (100,  90, 210),  # indigo
            'G': (170,  90, 210),  # purple
        }
        # MIDI number to letter mapping
        self.N_MIDI = 128
        self.PC_TO_LETTER = {
            0:'C', 1:'C', 2:'D',
            3:'D', 4:'E', 5:'F',
            6:'F', 7:'G', 8:'G',
            9:'A', 10:'A', 11:'B'
        }
        self.midi_is_sharp = lambda m: (m % 12) in {1, 3, 6, 8, 10}
        self._init_bg()

    def midi_to_rgba(self, m, alpha=50):
        """Convert a MIDI number to an RGBA color tuple."""
        letter = self.PC_TO_LETTER[m % 12]
        r,g,b = self.LETTER_RGB[letter]

        if self.midi_is_sharp(m): # make sharps darker
            r = int(r*0.70)
            g = int(g*0.70)
            b = int(b*0.70)

        return (r, g, b, alpha)
    
    def _init_bg(self):
        """
        Build a fixed 0..127 MIDI RGBA texture (height=128 rows, one per MIDI).
        This never changes, so colors are locked to absolute MIDI.
        """
        # width can be tiny; GPU stretches it. Use width=2 for stability.
        arr = np.zeros((self.N_MIDI, 2, 4), dtype=np.ubyte)

        # generate color array for each MIDI number
        for m in range(self.N_MIDI): 
            r,g,b,a = self.midi_to_rgba(m)
            arr[m, :, 0] = r
            arr[m, :, 1] = g
            arr[m, :, 2] = b
            arr[m, :, 3] = a

        self.setImage(arr[:, :, :], autoLevels=False)

        # pin the image's Y rect to [0,128] forever
        # set X span to default dummy values
        xmin, xmax = -1, 4
        rect = pg.QtCore.QRectF(xmin, 0.325, xmax - xmin, 128.325)
        self.setRect(rect)
        self.update_x(xmin, xmax)

    def update_x(self, xmin: float, xmax: float):
        """
        Update the image's X span; uses setRect() to change 
        only X, keep Y fixed 0..128.

        Args:
            xmin (float): new minimum x-value
            xmax (float): new maximum x-value
        """
        # Keep Y locked to MIDI domain 0..128 (1 unit = 1 MIDI)
        rect = pg.QtCore.QRectF(xmin, 0.325, xmax - xmin, 128.325)
        self.setRect(rect)

class MidiAxis(pg.AxisItem):
    """
    Overloaded pyqtgraph AxisItem to display y-axis as note names
    rather than as raw MIDI numbers. Eg, 60 -> C4.
    """
    NOTE_NAMES = [
        'C', 'C#', 'D', 'D#', 'E', 'F',
        'F#', 'G', 'G#', 'A', 'A#', 'B'
    ]
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setStyle(autoExpandTextSpace=True)

    def tickValues(self, minVal, maxVal, size):
        """
        Return tick levels as [(majorStep, majorValues), (minorStep, minorValues)].
        We choose a semitone-based major step based on zoom span.
        """
        span = float(maxVal - minVal)
        if span <= 0:
            return []

        # target ~8–12 major labels depending on pixel height
        target_labels = max(6, min(12, int(size / 35)))

        # candidate steps in semitones
        candidates = np.array([1, 2, 3, 4, 6, 12, 24, 36, 48], dtype=int)
        # pick the smallest step that yields <= target_labels
        labels_per_span = span / candidates
        try:
            major_step = int(candidates[np.argmax(labels_per_span <= target_labels)])
            if labels_per_span.max() > target_labels and major_step == 0:
                major_step = int(candidates[-1])
        except Exception:
            major_step = 12  # sane default
        if major_step <= 0:
            major_step = 12

        # align majors to the step boundary
        start_major = int(np.floor(minVal / major_step) * major_step)
        end_major   = int(np.ceil (maxVal / major_step) * major_step)
        majors = np.arange(start_major, end_major + 1, major_step, dtype=int)

        # minors at 1 semitone (only when not too dense)
        if major_step >= 6:
            start_minor = int(np.floor(minVal))
            end_minor   = int(np.ceil (maxVal))
            minors = np.arange(start_minor, end_minor + 1, 1, dtype=int)
            # drop those that coincide with majors
            minors = minors[~np.isin(minors, majors)]
            return [(major_step, majors), (1, minors)]
        else:
            return [(major_step, majors)]

    def tickStrings(self, values, scale, spacing):
        """
        Label only the first tick level (majors). Pyqtgraph passes majors first.
        Values for minors will be ignored by this method for that level.
        """
        # values can be floats; they are exactly integers from our tickValues
        return [self.midi_to_name(int(round(v))) for v in values]

    @staticmethod
    def midi_to_name(m: int) -> str:
        """Convert MIDI number to name, e.g. 60 -> C4."""
        pitch = m % 12
        octave = (m // 12) - 1
        return f"{MidiAxis.NOTE_NAMES[pitch]}{octave}"

class ScorePlot(QWidget):
    def __init__(self, midi_data: MidiData=None, user_data: UserData=None):
        super().__init__()
        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

        # important objects
        self.midi_data = midi_data
        self.user_data = user_data

        # --- TIMEKEEPING VARIABLES ---
        # windowing variables
        self.w, self.h = 5, 50 # seconds, midi numbers
        self.x_range, self.y_range = (-1, 4), (40, 90) # initial view ranges
        self.x_margin = 0.5 # 50% margin on the x-axis

        # timeline variables
        self.t = 0 # current time in seconds
        self.timeline_offset = 0.2 # x fraction of screen from left
        self.is_moving = False

        # ---- THE PLOT. ----
        self.y_axis = MidiAxis(orientation='left') # our custom y-axis
        self.plot = pg.PlotWidget(axisItems={'left': self.y_axis})
        self._layout.addWidget(self.plot)

        self.init_background()
        self.init_colors()
        self.init_objects()
        self.init_view()

    def init_background(self):
        self.plot.setBackground((20,20,25))
        self.bg = MidiBackground() # our colorful custom MIDI background :-)
        self.plot.addItem(self.bg, ignoreBounds=True) # don't let bg affect autorange

    def init_colors(self):
        """Define all colors used in the plot."""
        self.colors = {
            'midi': pg.mkBrush(255, 255, 255, 200), # white
            'user_note': pg.mkBrush(210, 100, 160, 150), # light pink
            'user_pitch': pg.mkBrush(250, 100, 180, 255), # pink
            'timeline': pg.mkPen(0, 255, 0, 255), # green
            'insertion': pg.mkBrush(0, 200, 0, 200), # green
            'deletion': pg.mkBrush(255, 0, 0, 200), # red
            'substitution': pg.mkBrush(255, 220, 0, 60) # translucent yellow
        }

    def init_objects(self):
        """Initialize all foreground plot items, including:
            - MIDI notes
            - user notes
            - user pitches
            - the timeline
        """
        self.NOTE_HEIGHT = 0.5 # height of note rectangles
        # ---- foreground items ----
        self.midi_notes = pg.BarGraphItem(
            x=[], height=self.NOTE_HEIGHT, 
            width=[], y0=0, 
            brush=self.colors['midi'], pen=None
        )
        self.midi_notes.setZValue(1) # above bg

        self.midi_notes_del = pg.BarGraphItem(
            x=[], height=self.NOTE_HEIGHT, y0=0, width=[],
            brush=self.colors['deletion'], pen=None
        )
        self.midi_notes_del.setZValue(1.1)

        self.user_notes = pg.BarGraphItem(
            x=[], height=self.NOTE_HEIGHT, y0=0, width=[],
            brush=self.colors['user_note'], pen=None
        )
        self.user_notes.setZValue(2) # above midi notes

        self.user_notes_ins = pg.BarGraphItem(
             x=[], height=self.NOTE_HEIGHT, y0=0, width=[],
            brush=self.colors['insertion'], pen=None
        )
        self.user_notes_ins.setZValue(2.1)

        self.user_pitches = pg.ScatterPlotItem(
            x=[], y=[], pen=pg.mkPen(None), brush=self.colors['user_pitch'], size=3
        )
        self.user_pitches.setZValue(3) # above user notes
        self.timeline = pg.InfiniteLine(
            pos=0, angle=90, pen=pg.mkPen(self.colors['timeline'])
        )
        self.timeline.setZValue(4) # above everything

        # add foreground after bg
        self.plot.addItem(self.midi_notes)
        self.plot.addItem(self.midi_notes_del)
        self.plot.addItem(self.user_notes)
        self.plot.addItem(self.user_notes_ins)
        self.plot.addItem(self.user_pitches)
        self.plot.addItem(self.timeline)

    def init_view(self):
        vb = self.plot.getViewBox()

        self.plot.enableAutoRange('xy', False)
        vb.setLimits(yMin=0, yMax=128)

        # set initial ranges
        vb.setRange(xRange=self.x_range, yRange=self.y_range, padding=0)
        vb.sigRangeChanged.connect(self.update_zoom)

        # cheap range handler: only adjust bg rect in X (keep Y fixed 0..128)
        # self.plot.getViewBox().sigRangeChanged.connect(self._on_range_changed)

    # ---------- pan/zoom handling ----------
    def update_zoom(self, viewbox, view_range):
        """updates the zoom of the plot when the viewbox range changes
        also updates the background accordingly
        """
        # ignore update_zoom calls while moving to avoid error accumulation
        if self.is_moving:
            return

        xmin, xmax = self.plot.viewRange()[0]
        self.bg.update_x(xmin, xmax)

        # store the new ranges
        self.x_range = view_range[0]
        self.y_range = view_range[1]
        self.w = self.x_range[1] - self.x_range[0]
        self.h = self.y_range[1] - self.y_range[0]

        self.update_view_items()

    def move_plot(self, t: float):
        """Move the plot to time t (sec)."""
        # print(f"--> Moving plot to {t} sec")
        self.is_moving = True # avoid accumulating errors in zoom

        # update the window boundaries
        self.t = t # update current time
        x_lower = self.t - (self.w*self.timeline_offset)
        x_upper = self.t + (self.w * (1-self.timeline_offset))
        self.x_range = (x_lower, x_upper)

        # update the background and our viewbox range
        self.bg.update_x(x_lower, x_upper)
        self.plot.getViewBox().setRange(xRange=self.x_range, yRange=self.y_range, padding=0)
        self.timeline.setPos(t) # also update the timeline pos

        self.is_moving = False # now we good

        self.update_view_items()

    # ---------- data loading ----------
    def load_midi(self, midi_data: MidiData):
        """Load a MidiData object and display its notes."""
        print("Loading MIDI data into ScorePlot...")
        
        self.midi_data = midi_data
        notes = list(midi_data.note_data.data.values())
        if not notes:
            print("No notes found in MIDI data.")
            return
        
        starts = np.array([n.start_time for n in notes], dtype=np.float32)
        ends = np.array([n.end_time for n in notes], dtype=np.float32)
        midis = np.array([n.midi_num[0] for n in notes], dtype=np.float32)

        x = 0.5 * (starts + ends)
        width = (ends - starts)
        y0 = (midis - 0.5*self.NOTE_HEIGHT)
        height = np.full_like(midis, self.NOTE_HEIGHT)

        # single call to setOpts
        self.midi_notes.setOpts(x=x, width=width, y0=y0, height=height)
        self.update_view_items()

    def load_audio(self, user_data: UserData):
        """Load a UserData object and display its notes and pitches."""
        print("Loading UserData into ScorePlot...")

        self.user_data = user_data
        # --- USER NOTES ---
        notes = [n for n in user_data.note_data.data.values()]

        starts = np.asarray([n.start_time for n in notes], dtype=np.float64)
        ends   = np.asarray([n.end_time   for n in notes], dtype=np.float64)
        midis  = np.asarray([n.midi_num[0]   for n in notes], dtype=np.float64)

        # one-semitone tall, centered on MIDI: [m-0.5, m+0.5]
        x = 0.5 * (starts + ends)
        width = (ends - starts)
        y0 = (midis - 0.5*self.NOTE_HEIGHT)
        height = np.full_like(midis, self.NOTE_HEIGHT)
        
        self.user_notes.setOpts(x=x, width=width, y0=y0, height=height)
        
        # --- USER PITCHES ---
        # PitchData.data may contain Nones; filter first
        pitches = user_data.pitch_data.data
        times = np.asarray(
            [p.time for p in pitches], dtype=np.float32
        )
        midis = np.asarray(
            [p.candidates[0][0] for p in pitches], 
            dtype=np.float32
        )
        mask = np.isfinite(times) & np.isfinite(midis)

        self.user_pitches.setData(x=times[mask], y=midis[mask])
        self.update_view_items()

    def plot_alignment(self, alignment: Alignment):
        """Plot the alignment results (user notes + mistakes)."""
        print("Plotting alignment...")
        self.midi_data.resize(new_length=self.user_data.audio_data.get_length())
        self.load_midi(self.midi_data)
        self.alignment = alignment
        self.update_view_items()
        # plot lines connecting user notes to midi notes

    def update_view_items(self):
        """Force all view items to update/redraw."""
        xmin, xmax = self.plot.viewRange()[0]

        user_pitches = self.user_data.pitch_data.read(xmin-1, xmax+1) if self.user_data else []
        user_pitches = [p for p in user_pitches if p is not None]

        user_notes = self.user_data.note_data.read(xmin-1, xmax+1) if self.user_data else []
        user_notes = [n for n in user_notes if n.midi_num[0] != -1]

        midi_notes = self.midi_data.note_data.read(xmin-1, xmax+1) if self.midi_data else []
        
        # --- USER PITCH UPDATING ---
        times = np.asarray(
            [p.time for p in user_pitches], dtype=np.float32
        )
        midis = np.asarray(
            [p.candidates[0][0] for p in user_pitches], 
            dtype=np.float32
        )
        mask = np.isfinite(times) & np.isfinite(midis)

        self.user_pitches.setData(x=times[mask], y=midis[mask])

        # --- USER NOTE UPDATING ---
        starts = np.asarray([n.start_time for n in user_notes], dtype=np.float64)
        ends   = np.asarray([n.end_time   for n in user_notes], dtype=np.float64)
        midis  = np.asarray([n.midi_num[0]   for n in user_notes], dtype=np.float64)

        x = 0.5 * (starts + ends)
        width = (ends - starts)
        y0 = (midis - 0.5*self.NOTE_HEIGHT)
        height = np.full_like(midis, self.NOTE_HEIGHT)

        self.user_notes.setOpts(x=x, width=width, y0=y0, height=height)

        # --- USER INSERTIONS OVERLAY (only those ids) ---
        if getattr(self, 'alignment', None) and user_notes:
            ins_mask = np.array(
                [getattr(n, 'id', None) in self.alignment.insertions for n in user_notes],
                dtype=bool)
            if np.any(ins_mask):
                self.user_notes_ins.setOpts(
                    x=x[ins_mask], 
                    width=width[ins_mask], 
                    y0=y0[ins_mask], 
                    height=height[ins_mask], 
                    pen=None
                )

        # --- MIDI NOTE UPDATING ---
        starts = np.array([n.start_time for n in midi_notes], dtype=np.float32)
        ends = np.array([n.end_time for n in midi_notes], dtype=np.float32)
        midis = np.array([n.midi_num[0] for n in midi_notes], dtype=np.float32)

        x = 0.5 * (starts + ends)
        width = (ends - starts)
        y0 = (midis - 0.5*self.NOTE_HEIGHT)
        height = np.full_like(midis, self.NOTE_HEIGHT)
        
        self.midi_notes.setOpts(x=x, width=width, y0=y0, height=height)

        # --- MIDI DELETIONS OVERLAY ---
        if getattr(self, 'alignment', None) and midi_notes:
            del_mask = np.array([n.id in self.alignment.deletions for n in midi_notes], dtype=bool)
            # print(f"deletion mask: {del_mask}")
            if np.any(del_mask):
                self.midi_notes_del.setOpts(
                    x=x[del_mask], 
                    width=width[del_mask], 
                    y0=y0[del_mask], 
                    height=height[del_mask], 
                    pen=None
                )