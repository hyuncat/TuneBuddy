from pathlib import Path

from PyQt6.QtCore import Qt, QRectF, QSize, QEvent, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QIcon, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTreeWidget, QTreeWidgetItem,
    QStyledItemDelegate, QStyle, QStyleOptionViewItem, QApplication,
)
from app_logic.Alignment import Mistake

_ICON_DIR = Path(__file__).resolve().parents[2] / "resources" / "icons"

# per-row flag (read off column 0) marking an overridden/dismissed mistake, so
# the delegate can darken the whole row
_OVERRIDE_ROLE = Qt.ItemDataRole.UserRole + 1


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

    def __init__(self, columns: set[int], override_brush: QBrush,
                 override_fg: QColor, icon_px: int = 18, parent=None):
        super().__init__(parent)
        self._columns = set(columns)
        self._icon_px = icon_px
        self._override_brush = override_brush
        self._override_fg = override_fg

    def paint(self, painter, option, index):
        overridden = bool(index.data(_OVERRIDE_ROLE))
        is_icon_col = index.column() in self._columns

        if not overridden:
            # default path: text columns rendered by the base delegate, icon
            # columns drawn with a centered (white) glyph
            if is_icon_col:
                self._paint_icon_cell(painter, option, index, tint=None)
            else:
                super().paint(painter, option, index)
            return

        # --- overridden row: fully custom-drawn so the grey survives selection.
        # The palette can't dim selected text here because qdarktheme sets the
        # selected-text color via the `selection-color` style-sheet property,
        # which QStyleSheetStyle uses instead of the palette. So we draw the
        # selection chrome with no text, then paint the text/icon ourselves grey.
        # First darken the (unselected) background; a selected row's blue fill is
        # drawn over it by the chrome below.
        painter.fillRect(option.rect, self._override_brush)
        if is_icon_col:
            self._paint_icon_cell(painter, option, index, tint=self._override_fg)
        else:
            self._paint_text_cell(painter, option, index, self._override_fg)

    def _paint_icon_cell(self, painter, option, index, tint):
        """Draw the cell chrome (background/selection) with a centered icon. The
        icon is recolored to `tint` when given (overridden rows), else drawn as-is."""
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.text = ""
        opt.icon = QIcon()
        widget = opt.widget
        style = widget.style() if widget else QApplication.style()
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, painter, widget)
        icon = index.data(Qt.ItemDataRole.DecorationRole)
        if isinstance(icon, QIcon) and not icon.isNull():
            s = self._icon_px
            r = option.rect
            x = r.x() + (r.width() - s) // 2
            y = r.y() + (r.height() - s) // 2
            if tint is not None:
                self._draw_tinted_icon(painter, icon, x, y, s, tint)
            else:
                icon.paint(painter, x, y, s, s)

    def _paint_text_cell(self, painter, option, index, color: QColor):
        """Draw the cell chrome (background/selection) then the text in `color`,
        bypassing the style's own (selection-aware) text coloring."""
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        text = opt.text
        opt.text = ""
        widget = opt.widget
        style = widget.style() if widget else QApplication.style()
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, painter, widget)

        text_rect = style.subElementRect(
            QStyle.SubElement.SE_ItemViewItemText, opt, widget
        )
        elided = opt.fontMetrics.elidedText(
            text, Qt.TextElideMode.ElideRight, text_rect.width()
        )
        painter.save()
        painter.setPen(color)
        painter.setFont(opt.font)
        painter.drawText(text_rect, int(opt.displayAlignment), elided)
        painter.restore()

    @staticmethod
    def _draw_tinted_icon(painter, icon: QIcon, x: int, y: int, s: int, color: QColor):
        """Draw `icon` recolored to a flat `color`, preserving its alpha mask, so
        the icons on overridden rows match the dimmed text."""
        pm = icon.pixmap(s, s)
        tp = QPainter(pm)
        tp.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
        tp.fillRect(pm.rect(), color)
        tp.end()
        painter.drawPixmap(x, y, pm)


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
    cleared = pyqtSignal()             # emits when the selection is cleared (e.g. click empty space)

    _TYPE_COL = 2
    _OVERRIDE_COL = 5

    # translucent dark tint laid over an overridden (dismissed) row so it reads
    # as "set aside" — pairs with the green highlight the same note gets in
    # GuitarHero. A null brush clears it back to the default row background.
    _OVERRIDE_BG = QBrush(QColor(0, 0, 0, 115))
    # grey the row's text + icons when overridden (qdarktheme's disabled color)
    _OVERRIDE_FG = QColor(105, 113, 119)

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
            "pencil": _svg_icon("pencil.svg"),
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
            _CenteredIconDelegate(
                {self._TYPE_COL, self._OVERRIDE_COL},
                self._OVERRIDE_BG,
                self._OVERRIDE_FG,
                parent=self.tree,
            )
        )

        self.tree.setColumnWidth(0, 24)   # "#"        — 1-2 digit index
        self.tree.setColumnWidth(1, 50)   # "Time"     — "59:59" or "45.67"
        self.tree.setColumnWidth(2, 38)   # "Type"     — icon only
        self.tree.setColumnWidth(3, 72)   # "Intended" — header is the wide element
        self.tree.setColumnWidth(4, 64)   # "Actual"
        self.tree.setColumnWidth(5, 36)   # pencil     — icon only

        # prevent the last column from stretching to fill remaining header width
        self.tree.header().setStretchLastSection(False)

        # center all column header labels
        self.tree.header().setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)

        # pencil icon in the override column header
        self.tree.headerItem().setIcon(self._OVERRIDE_COL, self._icons["pencil"])

        self._layout.addWidget(self.tree)

        col_total = sum(self.tree.columnWidth(i) for i in range(self.tree.columnCount()))
        m = self._layout.contentsMargins()
        content_width = col_total + m.left() + m.right() + self.tree.frameWidth() * 2
        self.setMinimumWidth(content_width)
        self.setMaximumWidth(content_width)

    def init_signals(self):
        self.tree.itemSelectionChanged.connect(self._on_selection_changed)
        self.tree.itemClicked.connect(self._on_item_clicked)
        # clicking empty space in the tree should clear the selection (and thus the
        # GuitarHero highlight); QTreeWidget doesn't do this on its own
        self.tree.viewport().installEventFilter(self)

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
        # flag every column so the delegate darkens the whole row, and each cell
        # repaints when toggled live (see _CenteredIconDelegate)
        for col in range(self.tree.columnCount()):
            item.setData(col, _OVERRIDE_ROLE, overridden)

    def _on_selection_changed(self):
        # read the selection, not currentItem(): clicking empty space clears the
        # selection but leaves currentItem set, so we'd never detect the deselect.
        items = self.tree.selectedItems()
        if not items:
            self.cleared.emit()
            return
        idx = items[0].data(0, Qt.ItemDataRole.UserRole)
        if idx is not None:
            self.selected.emit(idx)

    def _on_item_clicked(self, item: QTreeWidgetItem, column: int):
        if column != self._OVERRIDE_COL:
            return
        idx = item.data(0, Qt.ItemDataRole.UserRole)
        if idx is not None:
            self.override_toggled.emit(idx)

    def eventFilter(self, obj, event):
        # a press on empty tree space (no item under the cursor) clears the
        # selection, which fires _on_selection_changed -> `cleared`
        if (obj is self.tree.viewport()
                and event.type() == QEvent.Type.MouseButtonPress
                and self.tree.itemAt(event.position().toPoint()) is None):
            self.tree.clearSelection()
        return super().eventFilter(obj, event)
