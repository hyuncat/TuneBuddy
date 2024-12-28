import sys
from PyQt6.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsLineItem
from PyQt6.QtGui import QPixmap, QPen, QPainter, QBrush, QColor
from PyQt6.QtCore import Qt, QPointF, QRectF
from music21 import converter, environment

class ScorePlayer(QGraphicsView):
    def __init__(self, xml_file, parent=None):
        super(ScorePlayer, self).__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.scene.setBackgroundBrush(QBrush(QColor(255, 255, 255)))
        self.load_score(xml_file)
        self.setRenderHints(self.renderHints() | QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)

    def load_score(self, xml_file):
        score = converter.parse(xml_file)
        score.write('musicxml.png', fp='score.png')  # Export score to a PNG file
        pixmap = QPixmap('score.png')
        self.scene.addPixmap(pixmap)
        
        # Set scene dimensions
        # Convert QRect to QRectF
        rect = pixmap.rect()
        sceneRect = QRectF(rect)
        self.setSceneRect(sceneRect)

    def draw_playback_position(self, position):
        # Clear existing lines
        for item in self.scene.items():
            if isinstance(item, QGraphicsPixmapItem):
                # Do not remove pixmaps
                continue
            elif isinstance(item, QGraphicsLineItem):
                self.scene.removeItem(item)
        
        # Draw a red line at the current playback position
        line = QGraphicsLineItem(position, 0, position, self.scene.height())
        line.setPen(QPen(Qt.GlobalColor.red, 2))
        self.scene.addItem(line)

    def add_custom_annotation(self, x, y, width, height):
        # Draw a rectangle at the given position with specified width and height
        rect = self.scene.addRect(x, y, width, height, QPen(Qt.GlobalColor.blue, 2))
        return rect

def midi_to_musicxml(midi_file):
    # Set the environment to write MusicXML files
    environment.set('musicxmlPath', '/Applications/MuseScore 3.app/')  # Adjust this path to your MuseScore installation
    # Load the MIDI file
    score = converter.parse(midi_file)
    # Write the MusicXML file
    output_file = midi_file.replace('.mid', '.xml')
    score.write('musicxml', fp=output_file)

# Main application
if __name__ == '__main__':

    # midi_to_musicxml('data/fast_fugue_midi.mid')

    app = QApplication(sys.argv)
    viewer = ScorePlayer('data/fast_fugue_midi.xml')
    viewer.show()

    # Example of drawing playback position and custom annotations
    viewer.draw_playback_position(100)  # Draw red line at x=100
    viewer.add_custom_annotation(150, 100, 50, 20)  # Add custom rectangle annotation

    sys.exit(app.exec())
