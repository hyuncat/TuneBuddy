from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtGui import QColor
from PyQt6.QtCore import pyqtSignal
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
import numpy as np
import qdarktheme


from app_logic.midi.ScoreData import ScoreData
from app_logic.user.ds.Recording import Recording
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

class GuitarHero(QWidget):
    plot_moved = pyqtSignal(float) # emits plot time in seconds
    def __init__(self, recording: Recording=None):
        super().__init__()
        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

        # important objects
        self.recording = recording
        self.score_data = recording.score_data if recording else None
        self.alignment: Alignment = recording.alignment if recording else None

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

    # --- INIT STUFF ---
    def init_background(self):
        self.plot.setBackground((20,20,25))
        self.bg = MidiBackground() # our colorful custom MIDI background :-)
        self.plot.addItem(self.bg, ignoreBounds=True) # don't let bg affect autorange

    def init_colors(self):
        """Define all colors used in the plot."""
        self.colors = {
            'midi': pg.mkBrush(255, 255, 255, 200), # white
            'midi_dim': pg.mkBrush(120, 120, 120, 150), # score notes OUTSIDE the clip
            'user_note': pg.mkBrush(55, 155, 144, 150),
            'user_pitch': pg.mkBrush(41, 177, 240, 255), 
            'timeline': pg.mkPen(0, 255, 0, 255), # green
            'insertion': pg.mkBrush(0, 200, 0, 200), # green
            'deletion': pg.mkBrush(255, 0, 0, 200), # red
            'substitution': pg.mkBrush(255, 220, 0, 60), # translucent yellow
            'measure': pg.mkPen(230, 230, 240, 20, width=4.0), # measure starts: thicker
            'beat': pg.mkPen(230, 230, 240, 20, width=1.0), # beats: thinner
        }
        # new shit
        self.rest_brush = pg.mkBrush(140, 140, 140)
        self.distance_brushes = []
        self.max_dist = 5.0
        self.distance_step = 0.05  # 100 buckets for 0..5

        num_buckets = int(self.max_dist / self.distance_step) + 1
        for i in range(num_buckets):
            d = i * self.distance_step

            if d <= 0.5:
                hue = 120
            else:
                alpha = (d - 0.5) / (self.max_dist - 0.5)
                alpha = max(0.0, min(alpha, 1.0))
                hue = int(120 * (1.0 - alpha))

            color = QColor()
            color.setHsv(hue, 255, 255)
            self.distance_brushes.append(pg.mkBrush(color))

        # --- POST-ANALYSIS (alignment-based) palette ---
        # Used after analyze() for pitches that carry an `align_distance`. Both
        # bounds scale with the recording's pitch-mistake tolerance: green
        # within that many semitones of the aligned score note, then ramps
        # green->red out to ALIGN_MAX_MULT * pitch_tolerance (insertions clamp to
        # the max bucket => solid red). Rebuilt per-recording in load_user().
        self.ALIGN_MAX_MULT = 4.0
        self.align_distance_brushes = []
        self._build_align_brushes(tolerance=0.3)  # default; rebuilt per-recording

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

        self.match_lines = pg.PlotDataItem(
            x=[], y=[],
            pen=pg.mkPen(255, 255, 255, 140, width=1.5, style=QtCore.Qt.PenStyle.DashLine)
        )
        self.match_lines.setZValue(2.2)
        self.plot.addItem(self.match_lines)

        self.user_pitches = pg.ScatterPlotItem(
            x=[], y=[], pen=pg.mkPen(None), brush=self.colors['user_pitch'], size=8
        )
        self.user_pitches.setZValue(3) # above user notes
        self.timeline = pg.InfiniteLine(
            pos=0, angle=90, pen=pg.mkPen(self.colors['timeline'])
        )
        self.timeline.setZValue(4) # above everything

        # add foreground after bg
        self.highlight_bar = pg.BarGraphItem(
            x=[], height=self.NOTE_HEIGHT * 2, y0=[], width=[],
            brush=pg.mkBrush(255, 80, 80, 130),
            pen=pg.mkPen(255, 80, 80, 255, width=2)
        )
        self.highlight_bar.setZValue(5)

        self.plot.addItem(self.midi_notes)
        self.plot.addItem(self.midi_notes_del)
        self.plot.addItem(self.user_notes)
        self.plot.addItem(self.user_notes_ins)
        self.plot.addItem(self.user_pitches)
        self.plot.addItem(self.timeline)
        self.plot.addItem(self.highlight_bar)

        # --- beat / measure gridlines ---
        # Pooled vertical InfiniteLines, generated from score_data.beats (the same
        # beatmap used to drive the metronome). Downbeats (measure starts) are drawn
        # with the thicker 'measure' pen, regular beats with the thinner 'beat' pen.
        # The pool grows lazily to the max number of beats visible at once.
        self.GRIDLINE_Z = 0  # above the MIDI background (-1), below the notes (1)
        self.gridlines: list[pg.InfiniteLine] = []

        # --- clip dimming ---
        # When the score is clipped, the regions OUTSIDE [b0, b1] are darkened: two
        # translucent black bands (left of b0, right of b1) drawn above the
        # background/gridlines but below the notes (so score notes keep their own
        # dimmed-grey brush — see update_midi_items). Hidden when not clipped.
        self.dim_brush = pg.mkBrush(0, 0, 0, 110)
        self._last_dim_bounds = None  # cache so the bands only re-position on change
        self.dim_left = pg.LinearRegionItem(
            values=(0, 0), orientation='vertical', brush=self.dim_brush,
            pen=pg.mkPen(None), hoverBrush=self.dim_brush, hoverPen=pg.mkPen(None),
            movable=False,
        )
        self.dim_right = pg.LinearRegionItem(
            values=(0, 0), orientation='vertical', brush=self.dim_brush,
            pen=pg.mkPen(None), hoverBrush=self.dim_brush, hoverPen=pg.mkPen(None),
            movable=False,
        )
        for region in (self.dim_left, self.dim_right):
            region.setZValue(0.5)  # over bg/gridlines, under notes (z>=1)
            region.hide()
            self.plot.addItem(region, ignoreBounds=True)  # never affect autorange

    def init_view(self):
        """initialize the viewbox settings"""
        vb = self.plot.getViewBox()
        self.plot.enableAutoRange('xy', False)
        vb.setLimits(yMin=0, yMax=128)

        # set initial ranges
        vb.setRange(xRange=self.x_range, yRange=self.y_range, padding=0)
        vb.sigRangeChanged.connect(self.update_zoom)


    # ---------- PAN/ZOOM HANDLING ----------
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
        """Move the plot to time t (sec).
        Update the window boundaries, the background, and our viewbox range.
        Note that we keep a "is_moving" flag on to avoid accumulating errors 
        in the auto-zoom logic.

        Args:
            t (float): time in seconds to move the plot to
        """
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
        self.plot_moved.emit(t)


    # ---------- DATA LOADING ----------
    def load_score(self, score_data: ScoreData):
        """Load a MidiData object and display its notes."""
        print("Loading MIDI data into GuitarHero...")
        self.score_data = score_data
        # self.recording = None
        # self.alignment = None
        self.update_view_items()

    def load_user(self, recording: Recording):
        """Load a Recording object and display its notes and pitches."""
        print(f"Loading Recording: {recording} into GuitarHero...")
        self.recording = recording
        self.score_data = recording.score_data
        self.alignment = recording.alignment
        # green band + red ramp track this recording's pitch-mistake tolerance
        self._build_align_brushes(tolerance=recording.config.pitch_tolerance)
        self.clear_highlight()
        self.update_view_items()

    def load_alignment(self, alignment: Alignment):
        """Plot the alignment results (user notes + mistakes)."""
        print("Plotting alignment...")
        self.alignment = alignment
        self.update_view_items()

    # --- THE ESSENTIAL PLOTTING STUFF (called every time we refresh the view) ---
    def update_view_items(self):
        """Force all view items to update/redraw. Called whenever:
        1. view range changes
        2. view items change
            2.1 midi/user data loaded
            2.2 alignment loaded
            2.3 pitch detected
        """
        PAD = 1
        xmin, xmax = self.plot.viewRange()[0]
        x_range = (xmin-PAD, xmax+PAD)
        # print(f"Updating view items for x_range={x_range}...")
        
        # --- USER PITCHES + NOTES UPDATING ---
        self.update_grid_items(x_range)
        self.update_user_items(x_range)
        self.update_midi_items(x_range)
        self.update_alignment_items(x_range)
        self.update_clip_overlay()

    def _active_clip_window(self) -> tuple[float, float] | None:
        """The clip's [b0, b1] window (derived from note indices), or None.
        Single source of truth: ScoreData.get_bounds(respect_clip=True)."""
        if self.score_data is None or not self.score_data.is_clipped():
            return None
        return self.score_data.get_bounds(respect_clip=True)

    def _active_score_notes_by_id(self) -> dict[int, object]:
        """Current score notes for the active instrument, keyed by stable note id."""
        if self.score_data is None:
            return {}
        note_data = self.score_data.note_datas.get(self.score_data.active_instrument)
        if note_data is None:
            return {}
        notes_by_id = {}
        for note in note_data.data.values():
            note_id = getattr(note, "id", None)
            if note_id is not None:
                notes_by_id[note_id] = note
        return notes_by_id

    @staticmethod
    def _resolve_score_note(note, score_notes_by_id: dict[int, object]):
        """Return this score note's current-timing counterpart when available."""
        if note is None:
            return None
        note_id = getattr(note, "id", None)
        if note_id is None:
            return note
        return score_notes_by_id.get(note_id, note)

    def _sync_alignment_score_notes(self):
        """Refresh score-side alignment refs after score timing is rebuilt.

        ScoreData.change_tempo()/resize() rebuilds NoteData with new Note
        objects. Alignment stores the old objects, so relink by stable note id
        before using alignment times for filtering, match lines, deletions, or
        mistake highlights.
        """
        if self.alignment is None or self.score_data is None:
            return
        score_notes_by_id = self._active_score_notes_by_id()
        if not score_notes_by_id:
            return

        changed = False
        replacements = {}
        pairs = []
        for user_note, score_note in self.alignment.pairs:
            current_score_note = self._resolve_score_note(score_note, score_notes_by_id)
            if current_score_note is not score_note:
                changed = True
                if score_note is not None:
                    replacements[id(score_note)] = current_score_note
            pairs.append((user_note, current_score_note))

        if not changed:
            return

        self.alignment.pairs = pairs
        for mistakes in (self.alignment.pitch_mistakes, self.alignment.timing_mistakes):
            for mistake in mistakes:
                midi_note = getattr(mistake, "midi_note", None)
                if midi_note is None:
                    continue
                current_midi_note = replacements.get(id(midi_note))
                if current_midi_note is None:
                    current_midi_note = self._resolve_score_note(midi_note, score_notes_by_id)
                if current_midi_note is not midi_note:
                    mistake.midi_note = current_midi_note
        self.alignment.init_2(pairs)

    def update_clip_overlay(self):
        """Darken the area OUTSIDE the clip: two dim bands left of b0 / right of b1
        (hidden when unclipped). Only re-positions the bands when the clip window
        actually CHANGES — otherwise pyqtgraph just transforms the existing static
        bands as the view scrolls, which avoids the per-tick setRegion flicker."""
        clip = self._active_clip_window()
        if clip == self._last_dim_bounds:
            return
        self._last_dim_bounds = clip
        if clip is None:
            self.dim_left.hide()
            self.dim_right.hide()
            return
        b0, b1 = clip
        BIG = 1e6  # well past any view; small enough to avoid transform precision jitter
        self.dim_left.setRegion((-BIG, b0))
        self.dim_right.setRegion((b1, BIG))
        self.dim_left.show()
        self.dim_right.show()


    def update_user_items(self, x_range: tuple[float, float]):
        """Update the currently plotted user items (pitches, notes) to fit the given x_range"""
        if self.recording is None:
            print("nothing :(")
            self.user_pitches.setData(x=[], y=[])
            self.user_notes.setOpts(x=[], width=[], y0=[], height=[])
            return
        
        # read the current pitches and notes in the x_range
        user_pitches = self.recording.pitch_data.read(x_range[0], x_range[1], clean=True) if self.recording else []
        user_notes = self.recording.note_data.read(
            start_time=x_range[0], 
            end_time=x_range[1], 
            clean=True
        ) if self.recording else []

        # --- update PITCHES ---
        xs, ys, brushes = [], [], []
        for p in user_pitches:
            if not p.candidate_pitches:
                continue
            # high-slope transition frames (slides between notes) are always
            # neutral grey: their pitch is mid-slide, so we never score them by
            # distance — even post-analyze where _update_pitch_distances has
            # given them a (meaningless) live_distance.
            if getattr(p, "is_transition", False):
                brush = self.rest_brush
            else:
                # after analyze() pitches carry an alignment-based distance; color
                # by that. while recording (or pre-analysis) it's None, so fall
                # back to the live per-frame distance coloring.
                ad = getattr(p, "aligned_distance", None)
                if ad is not None:
                    brush = self.get_align_distance_brush(ad)
                else:
                    brush = self.get_distance_brush(getattr(p, "live_distance", None))
            # polyphonic frames carry SIMULTANEOUS pitches -> draw them all;
            # mono candidates are competing hypotheses -> draw only the best
            cands = (p.candidate_pitches if getattr(p, "polyphonic", False)
                     else p.candidate_pitches[:1])
            for midi_num, _salience in cands:
                xs.append(p.time)
                ys.append(midi_num)
                brushes.append(brush)
                    
        # get_alpha = lambda p: int(50 + 205*(1 - p.candidate_pitches[0][1]))
        # alphas = np.asarray([get_alpha(p) for p in user_pitches], dtype=np.float32)
        # brushes = [pg.mkBrush(41, 177, 240, a) for a in alphas]

        self.user_pitches.setData(x=xs, y=ys, brush=brushes)

        if self.recording.note_data is None:
            self.user_notes.setOpts(x=[], width=[], y0=[], height=[])
            return
        
        # --- update NOTES ---
        # get the note parameters for the BarGraphItem
        # starts = np.asarray([n.start_time for n in user_notes], dtype=np.float64)
        # ends = np.asarray([n.end_time for n in user_notes], dtype=np.float64)
        # midis = np.asarray([n.midi_num for n in user_notes], dtype=np.float64)

        starts, ends, midis = [], [], []
        for n in user_notes:
            for m in n.midi_num:
                starts.append(n.start_time)
                ends.append(n.end_time)
                midis.append(m)
                break
        
        starts = np.array(starts, dtype=np.float64)
        ends = np.array(ends, dtype=np.float64)
        midis = np.array(midis, dtype=np.float64)


        x = 0.5 * (starts + ends) # each rect starts at the center
        width = (ends - starts) # width is duration
        y0 = (midis - 0.5*self.NOTE_HEIGHT) # bottom y-pos
        height = np.full_like(midis, self.NOTE_HEIGHT) # constant height

        self.user_notes.setOpts(x=x, width=width, y0=y0, height=height)

    def update_midi_items(self, x_range: tuple[float, float]):
        """Update the currently plotted MIDI items (notes) to fit the given x_range"""
        if self.score_data is None:
            self.midi_notes.setOpts(x=[], width=[], y0=[], height=[])
            return

        # --- MIDI NOTE UPDATING ---
        # read the current midi notes in the x_range
        note_data = self.score_data.note_datas.get(self.score_data.active_instrument, None)
        midi_notes = note_data.read(x_range[0], x_range[1]) if note_data else []

        # Score notes can be CHORDS: NoteData stores one Note per onset with EVERY
        # simultaneous pitch in Note.midi_num (see MidiData.make_notedatas), so draw
        # a bar for each pitch — otherwise only the first note of each chord shows.
        # When clipped, score notes OUTSIDE the clip are drawn dimmer grey (a note
        # is "in the clip" iff its START is in [b0, b1), matching the alignment).
        clip = self._active_clip_window()
        starts, ends, midis, brushes = [], [], [], []
        for n in midi_notes:
            in_clip = clip is None or (clip[0] - 1e-6 <= n.start_time < clip[1] - 1e-6)
            brush = self.colors['midi'] if in_clip else self.colors['midi_dim']
            for m in n.midi_num:
                if m == -1:  # rest / unvoiced placeholder, nothing to draw
                    continue
                starts.append(n.start_time)
                ends.append(n.end_time)
                midis.append(m)
                brushes.append(brush)

        starts = np.array(starts, dtype=np.float32)
        ends = np.array(ends, dtype=np.float32)
        midis = np.array(midis, dtype=np.float32)

        x = 0.5 * (starts + ends)
        width = (ends - starts)
        y0 = (midis - 0.5*self.NOTE_HEIGHT)
        height = np.full_like(midis, self.NOTE_HEIGHT)

        # brushes only needed when clipped (else every bar uses the default white).
        self.midi_notes.setOpts(x=x, width=width, y0=y0, height=height,
                                brush=self.colors['midi'],
                                brushes=(brushes if clip is not None else None))

    def _get_gridline(self, idx: int) -> pg.InfiniteLine:
        """Return the idx-th pooled gridline, lazily creating (and adding) it."""
        while idx >= len(self.gridlines):
            line = pg.InfiniteLine(angle=90, pen=self.colors['beat'])
            line.setZValue(self.GRIDLINE_Z)
            # ignoreBounds so the gridlines never affect autorange (like the bg)
            self.plot.addItem(line, ignoreBounds=True)
            self.gridlines.append(line)
        return self.gridlines[idx]

    def update_grid_items(self, x_range: tuple[float, float]):
        """Update the beat/measure gridlines to fit the given x_range.

        Reuses score_data.beats (the metronome beatmap): each entry is a
        (time_sec, is_downbeat) tuple. Downbeats are measure starts and get the
        thicker 'measure' pen; the rest get the thinner 'beat' pen.
        """
        beats = getattr(self.score_data, "beats", None) if self.score_data else None
        if not beats:
            for line in self.gridlines:
                line.hide()
            return

        xmin, xmax = x_range
        idx = 0
        for i, (beat_time, is_downbeat) in enumerate(beats):
            # if not is_downbeat:
            #     continue
            if beat_time < xmin or beat_time > xmax:
                continue
            line = self._get_gridline(idx)
            line.setPos(beat_time)
            line.setPen(self.colors['measure'] if is_downbeat else self.colors['beat'])
            line.show()
            idx += 1

        # hide any pooled lines left over from a wider/denser view
        for j in range(idx, len(self.gridlines)):
            self.gridlines[j].hide()

    def update_alignment_items(self, x_range: tuple[float, float]):
        """Update the alignment overlay items (insertions, deletions, match lines)
        to fit the given x_range."""
        if self.alignment is None:
            self.user_notes_ins.setOpts(x=[], width=[], y0=[], height=[])
            self.midi_notes_del.setOpts(x=[], width=[], y0=[], height=[])
            self.match_lines.setData(x=[], y=[])
            return
        self._sync_alignment_score_notes()
        
        # --- CORRECTIONS OVERLAY ---
        # retrieve all alignment related components for the xrange
        goods, subs, ins, dels = self.alignment.get_alignment(x_range[0], x_range[1])

        # ---> MATCH LINES --->
        matches = goods + subs
        xs, ys = [], []
        for n, m in matches:
            if n is None or m is None:
                continue
            # compute midpoints for USER and MIDI notes
            ux = 0.5 * (n.start_time + n.end_time)
            uy = float(n.midi_num[0])
            mx = 0.5 * (m.start_time + m.end_time)
            my = float(m.midi_num[0])

            # rk: np.nan separates line segments from e/o
            xs.extend([ux, mx, np.nan]) 
            ys.extend([uy, my, np.nan])

        self.match_lines.setData(x=np.asarray(xs, dtype=np.float32),
                                y=np.asarray(ys, dtype=np.float32))
        # print("plotted alignment updates")

        # ---> USER INSERTIONS OVERLAY --->
        if ins:
            starts = np.asarray([n.start_time for n in ins], dtype=np.float64)
            ends   = np.asarray([n.end_time   for n in ins], dtype=np.float64)
            midis  = np.asarray([n.midi_num[0]   for n in ins], dtype=np.float64)

            x = 0.5 * (starts + ends)
            width = (ends - starts)
            y0 = (midis - 0.5*self.NOTE_HEIGHT)
            height = np.full_like(midis, self.NOTE_HEIGHT)

            self.user_notes_ins.setOpts(
                x=x, width=width, y0=y0, height=height
            )
        else:
            # clear stale insertions (e.g. when switching to a recording that
            # has none in view) — otherwise the old bars linger
            self.user_notes_ins.setOpts(x=[], width=[], y0=[], height=[])
        # ---> MIDI DELETIONS OVERLAY --->
        if dels:
            starts = np.asarray([n.start_time for n in dels], dtype=np.float32)
            ends   = np.asarray([n.end_time   for n in dels], dtype=np.float32)
            midis  = np.asarray([n.midi_num[0]   for n in dels], dtype=np.float32)

            x = 0.5 * (starts + ends)
            width = (ends - starts)
            y0 = (midis - 0.5*self.NOTE_HEIGHT)
            height = np.full_like(midis, self.NOTE_HEIGHT)

            self.midi_notes_del.setOpts(
                x=x, width=width, y0=y0, height=height
            )
        else:
            # clear stale deletions, same reasoning as insertions above
            self.midi_notes_del.setOpts(x=[], width=[], y0=[], height=[])

    def highlight_mistake(self, mistake):
        """Pan to and highlight the note(s) involved in a mistake."""
        self._sync_alignment_score_notes()
        if not mistake.user_note and not mistake.midi_note:
            return
        
        notes = []
        if mistake.type == "substitution":
            notes.extend([mistake.user_note, mistake.midi_note])
        elif mistake.type == "insertion":
            notes.append(mistake.user_note)
        elif mistake.type == "deletion":
            notes.append(mistake.midi_note)
        else:
            # timing mistakes (early / late / short / long): both notes exist, so
            # box them both to make the onset/duration discrepancy visible.
            notes.extend(n for n in (mistake.user_note, mistake.midi_note)
                         if n is not None)

        if not notes:
            return
        med_time = np.mean([0.5*(n.start_time + n.end_time) for n in notes])
        self.move_plot(med_time)
        starts = np.array([n.start_time for n in notes], dtype=np.float64)
        ends   = np.array([n.end_time   for n in notes], dtype=np.float64)
        midis  = np.array([n.midi_num[0] for n in notes], dtype=np.float64)
        self.highlight_bar.setOpts(
            x=0.5*(starts+ends), width=ends-starts,
            y0=midis - self.NOTE_HEIGHT, height=np.full_like(midis, self.NOTE_HEIGHT * 2)
        )
        self.update_highlight_override(mistake.is_overridden())

    def update_highlight_override(self, overridden: bool):
        """Swap the highlight color: green if overridden, red if not."""
        if overridden:
            self.highlight_bar.setOpts(brush=pg.mkBrush(80, 255, 80, 130),
                                       pen=pg.mkPen(80, 255, 80, 255, width=2))
        else:
            self.highlight_bar.setOpts(brush=pg.mkBrush(255, 80, 80, 130),
                                       pen=pg.mkPen(255, 80, 80, 255, width=2))

    def clear_highlight(self):
        self.highlight_bar.setOpts(x=[], width=[], y0=[], height=[])
        self.update_highlight_override(False)  # reset color for next use

    def get_distance_brush(self, d: float | None):
        if d is None:
            return self.rest_brush

        d = abs(float(d))
        d = min(d, self.max_dist)

        idx = int(d / self.distance_step)
        idx = min(idx, len(self.distance_brushes) - 1)
        return self.distance_brushes[idx]

    def _build_align_brushes(self, tolerance: float):
        """(Re)build the post-analysis palette from the pitch-mistake
        pitch_tolerance: green within that many semitones of the aligned note,
        ramping green->red out to ALIGN_MAX_MULT * tolerance."""
        green_thresh = max(float(tolerance), 0.0)
        # keep at least one bucket of ramp even if tolerance is ~0
        max_dist = max(self.ALIGN_MAX_MULT * green_thresh, green_thresh + self.distance_step)
        self.align_green_thresh = green_thresh
        self.align_max_dist = max_dist

        self.align_distance_brushes = []
        num_buckets = int(max_dist / self.distance_step) + 1
        for i in range(num_buckets):
            d = i * self.distance_step
            if d <= green_thresh:
                hue = 120
            else:
                frac = (d - green_thresh) / (max_dist - green_thresh)
                frac = max(0.0, min(frac, 1.0))
                hue = int(120 * (1.0 - frac))

            color = QColor()
            color.setHsv(hue, 255, 255)
            self.align_distance_brushes.append(pg.mkBrush(color))

    def get_align_distance_brush(self, d: float):
        """Brush for an alignment-based distance. inf (insertions) clamps to the
        max bucket => solid red."""
        d = min(abs(float(d)), self.align_max_dist)
        idx = int(d / self.distance_step)
        idx = min(idx, len(self.align_distance_brushes) - 1)
        return self.align_distance_brushes[idx]



class RunGuitarHero:
    def __init__(self, recording: Recording=None, app=None):
        import sys
        from PyQt6.QtWidgets import QApplication, QMainWindow

        if app is None:
            self.app = QApplication(sys.argv)
        else:
            self.app = app

        self.main_window = QMainWindow()
        self.app.setStyleSheet(qdarktheme.load_stylesheet("dark"))
        self.main_window.setWindowTitle("Attune [Guitar Hero]")
        self.main_window.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.central_layout = QVBoxLayout(self.central_widget)
        self.main_window.setCentralWidget(self.central_widget)

        # initialize the visualizer widget adding it to the layout
        self.vis = GuitarHero()
        self.central_layout.addWidget(self.vis)
        self.init_toolbar()

        if recording is not None:
            self.vis.load_user(recording)

        self.main_window.show()
        self.app.exec()

    def init_toolbar(self):
        from PyQt6.QtWidgets import QToolBar
        from PyQt6.QtCore import Qt

        self.toolbar = QToolBar()
        self.toolbar.setOrientation(Qt.Orientation.Horizontal)
        self.toolbar.addAction("Exit", self.close)
        self.main_window.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.toolbar)

    def close(self):
        self.app.quit()
