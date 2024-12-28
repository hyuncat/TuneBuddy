import sys
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QToolBar, QMainWindow, QApplication
from PyQt6.QtGui import QPixmap, QColor, QAction
from PyQt6.QtCore import Qt
import os

class ScoreViewer(QWidget):
    def __init__(self, score_image_path=None):
        super().__init__()
        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

        # Setup QGraphicsView for displaying the score
        self.graphics_view = QGraphicsView()
        self.graphics_scene = QGraphicsScene()
        self.graphics_scene.setBackgroundBrush(QColor("white"))
        self.graphics_view.setScene(self.graphics_scene)
        self._layout.addWidget(self.graphics_view)

        # Enable panning with ScrollHandDrag mode
        self.graphics_view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

        # Initialize zoom variables
        self.zoom_level = 1.0
        self.zoom_step = 0.1
        self.max_zoom_in = 2.5
        self.max_zoom_out = 0.3

        if score_image_path:
            self.load_score(score_image_path)

    def load_score(self, image_path):
        # Clear existing scores
        self.graphics_scene.clear()

        # Load image into QPixmap
        pixmap = QPixmap(image_path)
        pixmap_item = QGraphicsPixmapItem(pixmap)
        self.graphics_scene.addItem(pixmap_item)

        # Adjust view to fit the image initially
        # self.graphics_view.fitInView(pixmap_item, mode=Qt.AspectRatioMode.KeepAspectRatio)

        # Fit view width to image width by calculating the required scaling factor
        view_width = self.graphics_view.viewport().width()
        image_width = pixmap.width()
        if image_width > 0:  # Avoid division by zero
            scale_factor = view_width / image_width
            self.graphics_view.resetTransform()  # Reset any previous transformations
            self.graphics_view.scale(scale_factor, scale_factor)
            self.zoom_level = scale_factor


    def zoom_in(self):
        if self.zoom_level < self.max_zoom_in:
            self.graphics_view.scale(1 + self.zoom_step, 1 + self.zoom_step)
            self.zoom_level += self.zoom_step

    def zoom_out(self):
        if self.zoom_level > self.max_zoom_out:
            self.graphics_view.scale(1 - self.zoom_step, 1 - self.zoom_step)
            self.zoom_level -= self.zoom_step


class RunScoreViewer:
    def __init__(self, app=None, score_image_path=None):
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
        from app.config import AppConfig

        if app is None:
            self.app = QApplication(sys.argv)
        else:
            self.app = app
        
        AppConfig.initialize(self.app)
        self.main_window = QMainWindow()
        # self.main_window.setGeometry(600, 600, 600, 600)

        # Create a central widget and set the layout for it
        self.central_widget = QWidget()
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_window.setCentralWidget(self.central_widget)

        # Initialize ScoreViewer and add it to the layout
        self.score_viewer = ScoreViewer(score_image_path)
        self.main_layout.addWidget(self.score_viewer)

        # Create and configure the toolbar
        self.toolbar = QToolBar("Main Toolbar", self.main_window)
        self.toolbar.setOrientation(Qt.Orientation.Horizontal)

        # Add zoom in and zoom out buttons to the toolbar
        zoom_in_action = QAction("Zoom In", self.main_window)
        zoom_in_action.triggered.connect(self.score_viewer.zoom_in)
        self.toolbar.addAction(zoom_in_action)

        zoom_out_action = QAction("Zoom Out", self.main_window)
        zoom_out_action.triggered.connect(self.score_viewer.zoom_out)
        self.toolbar.addAction(zoom_out_action)

        # Add an exit button
        exit_action = QAction("Exit", self.main_window)
        exit_action.triggered.connect(self.close)
        self.toolbar.addAction(exit_action)

        # Add toolbar to the main window
        self.main_window.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.toolbar)

        self.main_window.show()
        self.app.exec()

    def close(self):
        self.app.quit()
