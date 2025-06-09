from PyQt6.QtGui import QColor
from PyQt6.QtCore import QRectF
from PyQt6.QtWidgets import QWidget, QVBoxLayout
import pyqtgraph as pg
import json

from app_logic.midi.MidiData import MidiData
from app_logic.user.ds.PitchData import Pitch
from app_logic.NoteData import NoteData, Note

class PitchPlot(QWidget):
    def __init__(self, clef: str="treble"):
        super().__init__()

        self.current_time = 0
        self.window_time = 0
        self.x_range = (-1, 4)
        self.y_range = (40, 90)
        self.clef = clef

        self.colors = {
            'background': "#ffffff",    # dark gray
            'label_text': '#454444',    # black
            'staff_lines': '#C4C4C4',   # grey
            'timeline': '#FF0000',      # red
            'pitch': '#F57C9C'          # pink
        }

        self.midi_config = {
            'normal_color': '#53555C',  # grey
            'played_color': '#3A3B40',  # darker grey
            'bar_height': 1
        }

        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

        # init the pyqtgraph plot
        self.plot = pg.PlotWidget()
        self._layout.addWidget(self.plot)
        
        self.plot.setBackground(self.colors['background'])

        # margins (to make room around labels)
        self.plot.getPlotItem().layout.setContentsMargins(10, 10, 10, 15) 
        
        # label tick colors
        self.plot.getAxis('bottom').setPen(self.colors['label_text'])
        self.plot.getAxis('left').setPen(self.colors['label_text'])
        self.plot.getAxis('bottom').setTextPen(self.colors['label_text'])
        self.plot.getAxis('left').setTextPen(self.colors['label_text'])

        # x/y axis labels
        self.plot.setLabel('bottom', 'Time (s)', color=self.colors['label_text'])  
        self.plot.setLabel('left', 'Pitch (MIDI #)', color=self.colors['label_text'])

        self.plot.setXRange(self.x_range[0], self.x_range[1])
        self.plot.setYRange(self.y_range[0], self.y_range[1])

        # add our staff+timeline
        self.plot_staff_lines(clef)
        self.plot_timeline(self.current_time)

        # plot items
        self.midi_notes = []
        self.pitches = []
        self.user_notes = []
        self.pitch_scatter = pg.ScatterPlotItem(
            size=5, pen=None, brush=pg.mkBrush(0, 100, 255)
        )
        self.plot.addItem(self.pitch_scatter)


    def plot_staff_lines(self, clef: str="treble"):
        """plots the staff lines as according to the given clef"""
        with open('resources/staff_lines.json', 'r') as file:
            data = json.load(file)

        staff_lines = data['clefs'][clef]['staff_lines']

        self.staff_lines = []
        for line in staff_lines:
            y = line['midi']
            # name = line['note']
            line = pg.InfiniteLine(
                pos=y,
                angle=0,
                pen=self.colors['staff_lines'],
                # label=name,
                # labelOpts={
                #     'position': 0.1,        # left end of the line
                #     'anchor': (1, 0.5),     # right-aligned, vertically centered
                #     'color': self.colors['staff_lines']
                # }
            )
            self.staff_lines.append(line)
            self.plot.addItem(line)

    def plot_timeline(self, current_time: float):
        """plots a thin red line at the specified time"""
        # add the TIMELINE
        self.timeline = pg.InfiniteLine(
            pos=current_time,
            angle=90,
            pen={'color': self.colors['timeline'], 'width': 2})
        self.plot.addItem(self.timeline)

    def plot_midi(self, midi_data: MidiData):
        """plots the given midi data as notes on the pitchplot"""

        # clear other midi notes
        for midi_note in self.midi_notes:
            self.plot.removeItem(midi_note)
        self.midi_notes = []

        # plot the new ones!
        for _, midi_note in midi_data.note_df.iterrows():

            start, duration, midi_num = (midi_note['start'], midi_note['duration'], midi_note['midi_num'])
            note_x = start + (duration / 2)
            # display colors behind the timeline as darker
            color = self.midi_config['normal_color'] if note_x >= self.current_time else self.midi_config['played_color']

            midi_note = pg.BarGraphItem(
                x=note_x,  
                y=midi_num,
                height=self.midi_config['bar_height'],
                width=duration,
                brush=color,
                pen=None,
                name='MIDI')
            self.midi_notes.append(midi_note)
            self.plot.addItem(midi_note)

            # print(f'plotted {midi_note}: pitch={pitch}')

    def plot_pitches(self, pitches: list[Pitch], from_scratch: bool=False):
        """
        plots all pitches, from a list of pitches, all at once
            => not for piecemeal pitch plotting since we erase the board 
               every time before running
        """
        # print("Plotting pitches...")

        # clear other pitches
        if from_scratch:
            self.clear_plot()

        # normalize volumes (min-max normalization)
        # volumes = [pitch.volume for pitch in pitches]
        # min_vol = min(volumes)
        # max_vol = max(volumes)
        
        # # avoid division by zero in case all volumes are the same
        # if max_vol == min_vol:
        #     normalized_volumes = [0.5] * len(pitches)
        # else:
        #     normalized_volumes = [(v-min_vol) / (max_vol-min_vol) for v in volumes]

        # brushes = []
        # for i, pitch in enumerate(pitches):
            # volume = normalized_volumes[i]

            # convert volume to hue:
            # 0 -> blue (HSV hue ~240), 0.5 -> yellow (HSV hue ~60) 1 -> red (HSV hue ~0)
            # if volume < 0.5:
            #     hue = 240 - (240-180) * (volume/0.5)  # blue to yellow
            # else:
            #     hue = 180 - (180 - 0) * ((volume-0.5) / 0.5)  # yellow to red

            # QColor uses HSV (hue, saturation, value)
            # color = QColor.fromHsv(int(hue), 255, 255) 

            # set opacity based on pitch probability
            # color.setAlphaF(1)  # opacity (alpha) based on pitch probability (0 to 1)
            # brushes.append(pg.mkBrush(color))  # Use mkBrush to create the brush with color

        self.pitches = pg.ScatterPlotItem(
            x=[pitch.time for pitch in pitches],
            y=[pitch.candidates[0][0] for pitch in pitches],
            size=5,
            pen=None,
            brush=self.colors['pitch'],
            name='Pitch'
        )
        self.plot.addItem(self.pitches)
        print("Done!")

    def plot_pitch(self, pitch: Pitch):
        self.pitch_scatter.addPoints(
            x=[pitch.time],
            y=[pitch.candidates[0][0]]
        )

    def clear_plot(self):
        """clears user stuff - pitches, notes - from plot"""
        for pitch in self.pitches:
            self.plot.removeItem(pitch)
        self.pitches = []
        for note in self.user_notes:
            self.plot.removeItem(note)
        self.user_notes = []

    def plot_notes(self, note_data: NoteData, from_scratch: bool=False):
        """plotting the notes in a note data as lines"""
        if from_scratch:
            self.clear_plot()

        bar_h = self.midi_config["bar_height"]  # e.g. 0.8 or 1.0

        for note in note_data.data.values():
            x0 = note.start_time
            w  = note.end_time - note.start_time
            y0 = note.midi_num - bar_h / 2
            h  = bar_h

            rect = pg.QtWidgets.QGraphicsRectItem(QRectF(x0, y0, w, h))
            rect.setPen   (pg.mkPen(None))                    # no outline
            rect.setBrush (pg.mkBrush(self.colors["staff_lines"]))  # your chosen color

            self.plot.addItem(rect)
            self.user_notes.append(rect)

    def plot_note(self, note: Note, from_scratch: bool=False):
        """plotting the note in a note data as lines"""
        if from_scratch:
            self.clear_plot()

        bar_h = self.midi_config["bar_height"]  # e.g. 0.8 or 1.0

        x0 = note.start_time
        w  = note.end_time - note.start_time
        y0 = note.midi_num - bar_h / 2
        h  = bar_h

        rect = pg.QtWidgets.QGraphicsRectItem(QRectF(x0, y0, w, h))
        rect.setPen   (pg.mkPen(None))                    # no outline
        rect.setBrush (pg.mkBrush(self.colors["staff_lines"]))  # your chosen color

        self.plot.addItem(rect)
        self.user_notes.append(rect)
        
    def move_plot(self, current_time: float):
        """move the plot to the specified time"""
        self.current_time = current_time

        self.timeline.setPos(current_time)

        x_lower = current_time + self.x_range[0]
        x_upper = current_time + self.x_range[1]
        self.plot.setXRange(x_lower, x_upper)
        
        # update midi color
        for midi_note in self.midi_notes:
            color = self.midi_config['normal_color'] if midi_note.opts['x'] >= self.current_time else self.midi_config['played_color']
            midi_note.setOpts(brush=color)
        