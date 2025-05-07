

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
        # clear existing scores
        self.graphics_scene.clear()

        # load image into QPixmap
        pixmap = QPixmap(image_path)
        pixmap_item = QGraphicsPixmapItem(pixmap)
        self.graphics_scene.addItem(pixmap_item)

        # fit image to graphics viewer
        view_width = self.graphics_view.viewport().width()
        image_width = pixmap.width()
        if image_width <= 0: 
            print("error: image width <= 0?..")

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

class AnalyzeTab(QWidget):
    def __init__(self):
        super().__init__()
        self._layout = QVBoxLayout(self)
        self.setLayout(self._layout)

        self.score_viewer = ScoreViewer(score_image_path="data/fugue_annot.png")
        self._layout.addWidget(self.score_viewer)
