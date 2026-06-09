from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTreeWidget, QTreeWidgetItem,
)
from app_logic.Alignment import Mistake


#converts number of seconds to a well-formatted time to display to the user in the mistake widget
def _format_time(seconds: float) -> str:
    #Convert seconds to decimal notation if under a minute, MM:SS for over a minute.
    if seconds < 60:
        return f"{seconds:.2f}"
    else:
        minutes = int(seconds) // 60
        secs = int(seconds) % 60
        return f"{minutes}:{secs:02d}"


class MistakeWidget(QWidget):
    """
    Right-side panel listing all analyzed mistakes for the active recording.

    Columns: Index | Pair | Type | Intended | Actual | Override

    The Override column is rendered as plain cell text (rather than an
    embedded QPushButton per row) and the cell click is captured via the
    tree's itemClicked signal. This avoids the macOS-specific GPU pressure
    of creating N native widgets when there are many mistakes — which on
    long pieces was starving the QtWebEngine GPU process and causing
    segfaults on resize/repaint.
    """
    selected = pyqtSignal(int)         # emits mistake index on row click
    override_toggled = pyqtSignal(int) # emits mistake index when Override cell is clicked

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(6, 6, 6, 6)

        self._mistakes: list[Mistake] = []
        self._OVERRIDE_COL = 5

        self.init_ui()
        self.init_signals()

    def init_ui(self):
        self.header_label = QLabel("Mistakes")
        self.header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._layout.addWidget(self.header_label)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(6)
        self.tree.setHeaderLabels(["#", "Time", "Type", "Intended", "Actual", "Override"])
        self.tree.setIndentation(0)
        self.tree.setRootIsDecorated(False)

        self.tree.setColumnWidth(0, 30)
        self.tree.setColumnWidth(1, 50)
        self.tree.setColumnWidth(2, 54)
        self.tree.setColumnWidth(3, 60)
        self.tree.setColumnWidth(4, 60)
        self.tree.setColumnWidth(5, 90)

        self._layout.addWidget(self.tree)

        self.setMinimumWidth(200)
        self.setMaximumWidth(380)

    def init_signals(self):
        self.tree.itemSelectionChanged.connect(self._on_selection_changed)
        self.tree.itemClicked.connect(self._on_item_clicked)

    # --- PUBLIC API ---
    def load_mistakes(self, mistakes: list[Mistake]):
        """Populate the tree with a new list of mistakes."""
        self._mistakes = mistakes
        self.tree.clear()
        # Batch the inserts: addTopLevelItems is much cheaper than N calls
        # to addTopLevelItem when the model is being watched.
        items = [self._make_item(idx, m) for idx, m in enumerate(mistakes)]
        self.tree.addTopLevelItems(items)

    def clear(self):
        self._mistakes = []
        self.tree.clear()

    def refresh_override(self, idx: int):
        """Update the override-cell appearance for a single mistake."""
        if not (0 <= idx < len(self._mistakes)):
            return
        item = self.tree.topLevelItem(idx)
        if item is None:
            return
        self._set_override_cell(item, self._mistakes[idx].is_overridden())

    # --- INTERNAL ---
    _TYPE_ABBREV = {"insertion": "INS", "deletion": "DEL", "substitution": "SUB"}
    _GREY = QColor("#888888")
    _WHITE = QColor("#ffffff")

    @staticmethod
    def _note_name(note) -> str:
        if note is None:
            return "—"
        return note.get_note_name()

    def _make_item(self, idx: int, mistake: Mistake) -> QTreeWidgetItem:
        #time is based on the MIDI Note rather than the user note
        time = _format_time(mistake.midi_note.start_time) if mistake.midi_note else "—"
        intended = self._note_name(mistake.midi_note)
        actual = self._note_name(mistake.user_note)
        type_label = self._TYPE_ABBREV.get(mistake.type, mistake.type)

        item = QTreeWidgetItem([str(idx), time, type_label, intended, actual, ""])
        item.setData(0, Qt.ItemDataRole.UserRole, idx)
        for col in range(6):
            item.setTextAlignment(col, Qt.AlignmentFlag.AlignCenter)
        self._set_override_cell(item, mistake.is_overridden())
        return item

    def _set_override_cell(self, item: QTreeWidgetItem, overridden: bool):
        item.setText(self._OVERRIDE_COL, "Overridden" if overridden else "Override")
        item.setForeground(self._OVERRIDE_COL, self._GREY if overridden else self._WHITE)

    def _on_selection_changed(self):
        item = self.tree.currentItem()
        if item is None:
            return
        idx = item.data(0, Qt.ItemDataRole.UserRole)
        if idx is not None:
            self.selected.emit(idx)

    def _on_item_clicked(self, item: QTreeWidgetItem, column: int):
        if column != self._OVERRIDE_COL:
            return
        idx = item.data(0, Qt.ItemDataRole.UserRole)
        if idx is not None:
            self.override_toggled.emit(idx)
