# import os
import pyqtgraph as pg
from PyQt6.Widget import QWidget, QVBoxLayout

class PitchPlot(QWidget):
    def __init__(self):
        super().__init__()

        self.time = 0
        self.x_range = (0, 5)
        self.y_range = (40, 90)

        self._layout = QVBoxLayout()
        self.setLayout(self.layout)

        # init the pyqtgraph plot
        self.plot = pg.PlotWidget()
        self._layout.addWidget(self.plot)
        
        self.plot.setBackground(self.colors['background'])
        
        # label tick colors
        self.plot.getAxis('bottom').setPen(self.colors['label_text'])
        self.plot.getAxis('left').setPen(self.colors['label_text'])
        self.plot.getAxis('bottom').setTextPen(self.colors['label_text'])
        self.plot.getAxis('left').setTextPen(self.colors['label_text'])

        # x/y axis labels
        self.plot.setLabel('bottom', 'Time (s)', color=self.colors['label_text'])  
        self.plot.setLabel('left', 'Pitch (MIDI #)', color=self.colors['label_text'])

        self.plot.setXRange(self.x_range)
        self.plot.setYRange(self.y_range)