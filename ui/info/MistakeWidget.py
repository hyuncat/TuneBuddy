from pathlib import Path

from PyQt6.QtCore import Qt, QRectF, QSize, pyqtSignal
from PyQt6.QtGui import QIcon, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTreeWidget, QTreeWidgetItem,
    QStyledItemDelegate, QStyle, QStyleOptionViewItem, QApplication,
)
from app_logic.Alignment import Mistake

_ICON_DIR = Path(__file__).resolve().parents[2] / "resources" / "icons"


def _svg_icon(filename: str, px: int = 64) -> QIcon:
    """Rasterize an SVG (already white-colored on disk) into a QIcon, rendered
    at `px` so it stays crisp when the view scales it down. The SVG's aspect
    ratio is preserved and centered within the square pixmap so the tall, narrow
    flat/sharp glyphs aren't stretched."""
    renderer = QSvgRenderer(str(_ICON_DIR / filename))
    pix = QPixmap(px, px)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    size = renderer.defaultSize()
    w, h = size.width(), size.height()
    if w > 0 and h > 0:
        scale = min(px / w, px / h)
        tw, th = w * scale, h * scale
        renderer.render(painter, QRectF((px - tw) / 2.0, (px - th) / 2.0, tw, th))
    else:
        renderer.render(painter)
    painter.end()
    return QIcon(pix)


class _CenteredIconDelegate(QStyledItemDelegate):
    """Draws the cell's decoration icon centered (the default delegate left-
    aligns it). Used for the icon-only Type and Override columns. A single
    delegate handles every row, so we avoid creating N per-row native widgets —
    which the comment on MistakeWidget warns starves the GPU process on macOS."""

    def __init__(self, columns: set[int], icon_px: int = 18, parent=None):
        super().__init__(parent)
        self._columns = set(columns)
        self._icon_px = icon_px

    def paint(self, painter, option, index):
        if index.column() not in self._columns:
            super().paint(painter, option, index)
            return
        # draw the background / selection highlight, but no text or (left-
        # aligned) decoration of its own
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.text = ""
        opt.icon = QIcon()
        widget = opt.widget
        style = widget.style() if widget else QApplication.style()
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, painter, widget)
        # draw the icon centered in the cell
        icon = index.data(Qt.ItemDataRole.DecorationRole)
        if isinstance(icon, QIcon) and not icon.isNull():
            s = self._icon_px
            r = option.rect
            x = r.x() + (r.width() - s) // 2
            y = r.y() + (r.height() - s) // 2
            icon.paint(painter, x, y, s, s)


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

    Columns: Index | Time | Type | Intended | Actual | Override

    The Type column shows an icon instead of text: a plus for an insertion, a
    minus for a deletion, and a musical flat / sharp for a substitution
    (depending on whether the user played under or over the target pitch). The
    Override column is an icon too: a trash-can to override (dismiss)
    a flagged mistake, and an undo arrow to take that back.

    Both icon columns are rendered via item decorations + a centered-icon
    delegate (rather than an embedded widget per row). This avoids the macOS-
    specific GPU pressure of creating N native widgets when there are many
    mistakes — which on long pieces was starving the QtWebEngine GPU process and
    causing segfaults on resize/repaint.
    """
    selected = pyqtSignal(int)         # emits mistake index on row click
    override_toggled = pyqtSignal(int) # emits mistake index when Override cell is clicked

    _TYPE_COL = 2
    _OVERRIDE_COL = 5

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(6, 6, 6, 6)

        self._mistakes: list[Mistake] = []

        # icons are fixed, so build them once and share across rows
        self._icons = {
            "insertion": _svg_icon("plus.svg"),
            "deletion": _svg_icon("minus.svg"),
            "flat": _svg_icon("flatsign.svg"),
            "sharp": _svg_icon("sharpsign.svg"),
            "trash": _svg_icon("trash-2.svg"),
            "undo": _svg_icon("undo-2.svg"),
        }

        self.init_ui()
        self.init_signals()

    def init_ui(self):
        self.header_label = QLabel("Mistakes")
        self.header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._layout.addWidget(self.header_label)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(6)
        self.tree.setHeaderLabels(["#", "Time", "Type", "Intended", "Actual", ""])
        self.tree.setIndentation(0)
        self.tree.setRootIsDecorated(False)
        self.tree.setIconSize(QSize(20, 20))
        self.tree.setItemDelegate(
            _CenteredIconDelegate({self._TYPE_COL, self._OVERRIDE_COL}, parent=self.tree)
        )

        self.tree.setColumnWidth(0, 30)
        self.tree.setColumnWidth(1, 50)
        self.tree.setColumnWidth(2, 44)
        self.tree.setColumnWidth(3, 60)
        self.tree.setColumnWidth(4, 60)
        self.tree.setColumnWidth(5, 40)

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
    @staticmethod
    def _note_name(note) -> str:
        if note is None:
            return "—"
        return note.get_note_name()

    def _type_icon_and_tip(self, mistake: Mistake) -> tuple[QIcon, str]:
        """Pick the Type-column icon + tooltip for a mistake."""
        if mistake.type == "insertion":
            return self._icons["insertion"], "Insertion (extra note played)"
        if mistake.type == "deletion":
            return self._icons["deletion"], "Deletion (note missed)"
        # substitution: flat if the user played under the target pitch, sharp if over
        user = mistake.user_note.midi_num[0] if mistake.user_note else 0
        target = mistake.midi_note.midi_num[0] if mistake.midi_note else 0
        if user < target:
            return self._icons["flat"], "Substitution (played flat)"
        return self._icons["sharp"], "Substitution (played sharp)"

    def _make_item(self, idx: int, mistake: Mistake) -> QTreeWidgetItem:
        #time is based on the MIDI Note rather than the user note
        time = _format_time(mistake.midi_note.start_time) if mistake.midi_note else "—"
        intended = "—" if mistake.type == "insertion" else self._note_name(mistake.midi_note)
        actual = "—" if mistake.type == "deletion" else self._note_name(mistake.user_note)

        item = QTreeWidgetItem([str(idx), time, "", intended, actual, ""])
        item.setData(0, Qt.ItemDataRole.UserRole, idx)
        for col in range(6):
            item.setTextAlignment(col, Qt.AlignmentFlag.AlignCenter)

        type_icon, type_tip = self._type_icon_and_tip(mistake)
        item.setIcon(self._TYPE_COL, type_icon)
        item.setToolTip(self._TYPE_COL, type_tip)

        self._set_override_cell(item, mistake.is_overridden())
        return item

    def _set_override_cell(self, item: QTreeWidgetItem, overridden: bool):
        if overridden:
            item.setIcon(self._OVERRIDE_COL, self._icons["undo"])
            item.setToolTip(self._OVERRIDE_COL, "Undo override")
        else:
            item.setIcon(self._OVERRIDE_COL, self._icons["trash"])
            item.setToolTip(self._OVERRIDE_COL, "Override (dismiss this mistake)")

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
