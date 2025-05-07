from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QWidget, QVBoxLayout
import pyqtgraph as pg

from app_logic.midi.MidiData import MidiData
from app_logic.user.ds.PitchData import Pitch

class PitchPlot(QWidget):
    def __init__(self):
        super().__init__()

        self.current_time = 0
        self.window_time = 0
        self.x_range = (-1, 4)
        self.y_range = (40, 90)

        self.colors = {
            'background': '#1D1D1D',    # dark gray
            'label_text': '#A4A4A4',    # black
            'timeline': '#FF0000',      # red
            'pitch': '#F57C9C'          # pink
        }

        self.midi_config = {
            'normal_color': '#53555C',  # Grey
            'played_color': '#3A3B40',  # Darker grey
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

        # add our timeline
        self.plot_timeline(self.current_time)

        # plot items
        self.midi_notes = []
        self.pitches = []


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
            for pitch in self.pitches:
                self.plot.removeItem(pitch)
            self.pitches = []

        # normalize volumes (min-max normalization)
        volumes = [pitch.volume for pitch in pitches]
        min_vol = min(volumes)
        max_vol = max(volumes)
        
        # avoid division by zero in case all volumes are the same
        if max_vol == min_vol:
            normalized_volumes = [0.5] * len(pitches)
        else:
            normalized_volumes = [(v-min_vol) / (max_vol-min_vol) for v in volumes]

        brushes = []
        for i, pitch in enumerate(pitches):
            volume = normalized_volumes[i]

            # convert volume to hue:
            # 0 -> blue (HSV hue ~240), 0.5 -> yellow (HSV hue ~60) 1 -> red (HSV hue ~0)
            if volume < 0.5:
                hue = 240 - (240-180) * (volume/0.5)  # blue to yellow
            else:
                hue = 180 - (180 - 0) * ((volume-0.5) / 0.5)  # yellow to red

            # QColor uses HSV (hue, saturation, value)
            color = QColor.fromHsv(int(hue), 255, 255) 

            # set opacity based on pitch probability
            color.setAlphaF(1)  # opacity (alpha) based on pitch probability (0 to 1)
            brushes.append(pg.mkBrush(color))  # Use mkBrush to create the brush with color

        self.pitches = pg.ScatterPlotItem(
            x=[pitch.time for pitch in pitches],
            y=[pitch.midi_num for pitch in pitches],
            size=5,
            pen=None,
            brush=brushes,
            name='Pitch'
        )
        self.plot.addItem(self.pitches)
        print("Done!")
    
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
        