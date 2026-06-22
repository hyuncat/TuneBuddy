from PyQt6.QtCore import Qt, pyqtSignal, QPoint
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem,
    QMenu, QMessageBox, QInputDialog,
)
from pathlib import Path
from app_logic.user.ds.Recording import Recording
from app_logic.midi.ScoreData import ScoreData
from resources.program_map import program_to_name, name_to_program


class RecordingTree(QWidget):
    """
    Left-side panel showing recordings for the currently loaded MIDI.
    Uses QTreeWidget for simplicity.

    String-only design:
      - Each recording item stores its name (str) in UserRole.
      - Signals emit names (str).
    """
    selected = pyqtSignal(str)          # emits str | None
    score_renamed = pyqtSignal(str)     # emits the new score title (MIDI_ROOT renamed)

    def __init__(self, recordings: dict[str, Recording], parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(6, 6, 6, 6)

        self.MIDI_ROOT: QTreeWidgetItem | None = None
        self.score_data: ScoreData | None = None # allows new recs to reference it
        self.recordings = recordings  # reference to the parent recordings dict
        self.active_recording: str | None = None # reference to the currently selected recording
        
        # helpers to suppress signals during item/selection changes
        self._suppress_item_changed = False  # prevent rename loops
        self._suppress_selection_changed = False # prevent selection loops

        self.init_ui()
        self.init_signals()

    def init_ui(self):
        """Initialize the UI components of the tree."""
        # the tree itself
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(14)
        self._layout.addWidget(self.tree)

        # set min/max widths of the widget holding the tree
        self.setMinimumWidth(180)
        self.setMaximumWidth(320)

    def init_signals(self):
        """initialize all the signals for
            - right click: open context menu
            - selection: emit selected recording name
            - double click: rename recording
        """
        # right click (context menu)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.open_context_menu)
        # selection (change active recording)
        self.tree.itemSelectionChanged.connect(self.on_selection_changed)
        self.tree.itemChanged.connect(self.on_item_changed)
        # double click (rename)
        self.tree.itemDoubleClicked.connect(self.rename_recording)
        self.tree.setEditTriggers(
            self.tree.EditTrigger.EditKeyPressed |
            self.tree.EditTrigger.SelectedClicked
        )

    def init_score(self, filepath: str|Path, score_data: ScoreData=None):
        """Reset tree for a new score from the given filepath."""
        self._suppress_selection_changed = True # sandwich
        # reset trees and recordings
        self.tree.clear() # clear any existing items
        self.recordings.clear() # clear the recordings dict
        self.active_recording = None # reset active recording reference

        # get the score name from filepath
        score_name = Path(filepath).stem
        self.MIDI_ROOT = QTreeWidgetItem([score_name])
        # the root label is the Score Title (source of truth): store it so a
        # rename can be diffed/reverted, like recording items do.
        self.MIDI_ROOT.setData(0, Qt.ItemDataRole.UserRole, score_name)

        # the root is editable (rename = retitle the score) but not selectable
        # (it isn't a recording).
        flags = self.MIDI_ROOT.flags()
        self.MIDI_ROOT.setFlags(
            (flags & ~Qt.ItemFlag.ItemIsSelectable) | Qt.ItemFlag.ItemIsEditable
        )
        self.tree.addTopLevelItem(self.MIDI_ROOT)
        self.MIDI_ROOT.setExpanded(True)

        self.score_data = score_data
        self._suppress_selection_changed = False

    def open_context_menu(self, pos: QPoint):
        """Open context menu with create, rename, delete actions."""
        menu = QMenu(self)
        new_action = menu.addAction("New Recording…")

        # if an item is selected, also offer rename and delete actions
        item = self.tree.itemAt(pos)
        if item is None or item is self.MIDI_ROOT:
            rename_action = None
            delete_action = None
        else:
            rename_action = menu.addAction("Rename")
            delete_action = menu.addAction("Delete")
            inst_select_action = menu.addAction("Select Instrument")

        # then execute the menu and get the selected action
        action = menu.exec(self.tree.viewport().mapToGlobal(pos))
        if action is None: 
            return # user clicked outside the menu, do nothing

        # --- action handler junction ---
        if action == new_action:
            self.new_recording()
        elif action == rename_action:
            self.rename_recording(item)
        elif action == delete_action:
            name = item.data(0, Qt.ItemDataRole.UserRole)
            if self.confirm_delete(name):
                self.delete_recording(item)
        elif action == inst_select_action:
            self.select_instrument(item)
        else:
            print("Unknown context menu action:", action)
        
        return
    
    def new_recording(self):
        """Prompt user for new recording name, create it, and select it."""
        name, ok = QInputDialog.getText(self, "New Recording", "Recording name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        if name in self.recordings:
            QMessageBox.warning(self, "Name already exists", f"A recording named '{name}' already exists.")
            return
        self._add_recording(name)

    def rename_recording(self, item: QTreeWidgetItem | None=None, col: int=0):
        """
        Start in-place edit of the item's name. Works for both recordings and
        the MIDI_ROOT (whose label is the editable Score Title).
        """
        if item is not None:
            self.tree.editItem(item, col)
    
    def delete_recording(self, item: QTreeWidgetItem):
        """Delete the recording with the given name after confirmation."""
        # find the item with this name
        if item is None or item is self.MIDI_ROOT:
            return
        name = item.data(0, Qt.ItemDataRole.UserRole) or item.text(0)
        
        # remove from dict and tree
        self.recordings.pop(name, None)
        parent = item.parent() or self.tree.invisibleRootItem()
        parent.removeChild(item)

        if self.active_recording == name:
            self.tree.setCurrentItem(None)
            self.active_recording = None
            self.selected.emit(None)

    def select_instrument(self, item: QTreeWidgetItem):
        """Prompt user to select an instrument for this recording."""
        if item is None or item is self.MIDI_ROOT:
            return
        name = item.data(0, Qt.ItemDataRole.UserRole) or item.text(0)
        rec = self.recordings.get(name)
        if rec is None:
            return
        
        # get list of instruments from score data
        score_data = rec.score_data
        print("rec active instrument:", rec.score_data.active_instrument)
        instruments = score_data.instruments
        if not instruments:
            QMessageBox.warning(self, "No instruments found", "The loaded score has no instruments to select.")
            return
        
        # prompt user to select an instrument from the list
        items = [f"{program_to_name(prog)}" for _, prog in instruments.items()]
        items.pop(-1) # remove metronome sound from selection
        item, ok = QInputDialog.getItem(self, "Select Instrument", "Instruments:", items, 0, False)
        if not ok or not item:
            return
        
        # parse channel number from selected item
        try:
            prog_num = name_to_program(item)
            ch_num = next(ch for ch, prog in instruments.items() if prog == prog_num)
            score_data.active_instrument = ch_num
            
            print("rec active instrument:", rec.score_data.active_instrument)

            self.selected.emit(name) # re-emit selected signal to trigger guitarhero refresh
        except Exception as e:
            print("Error parsing selected instrument:", e)
            QMessageBox.warning(self, "Invalid selection", "Could not parse the selected instrument.")

    # --- INTERNAL ---
    def confirm_delete(self, name: str) -> bool:
        """Ask the user to confirm deletion of the recording with the given name."""
        if not name:
            return False
        ok = QMessageBox.question(
            self,
            "Delete recording",
            f"Delete recording: {name}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        return ok == QMessageBox.StandardButton.Yes

    def _add_recording(self, name: str):
        """Helper to add a recording with the given name to the tree and dict."""
        if name in self.recordings.keys():
            print(f"Recording with name '{name}' already exists. Skipping creation.")
            return
        # create the recording and add to the dict
        rec = Recording(score_data=self.score_data)
        self.recordings[name] = rec
        # add to the tree under MIDI_ROOT
        item = QTreeWidgetItem([name])
        item.setData(0, Qt.ItemDataRole.UserRole, name)
        
        item.setFlags( # makes sure it's editable, selectable, and enabled
            item.flags() | Qt.ItemFlag.ItemIsEditable 
            | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
        )  
        self.MIDI_ROOT.addChild(item)
        self.MIDI_ROOT.setExpanded(True)
        # select the new item
        self.tree.setCurrentItem(item)
        # self.selected.emit(name)

    def on_selection_changed(self):
        """
        When the user selects a different recording in the tree, 
        update the active recording reference and emit the selected signal.
        """
        if self._suppress_selection_changed:
            return
        item = self.tree.currentItem()
        if item is not None and item is not self.MIDI_ROOT:
            self.active_recording = item.data(0, Qt.ItemDataRole.UserRole)
            self.selected.emit(self.active_recording)
        else:
            self.active_recording = None

    def get_selection(self) -> QTreeWidgetItem | None:
        """
        Helper to get the currently selected recording item, or None 
        if no valid selection. 
        
        Used for context menu right click verification. 
            (Even when active_recording is set, the item might be None 
            if the user right-clicked on empty space or the MIDI_ROOT.)
        """
        item = self.tree.currentItem()
        if item is not None and item is not self.MIDI_ROOT:
            return item
        return None
    
    def on_item_changed(self, item: QTreeWidgetItem, col: int):
        """
        Called after editing the item text.
        """
        if self._suppress_item_changed:
            return
        if item is None:
            return
        # the root label is the Score Title, handled separately
        if item is self.MIDI_ROOT:
            self._handle_score_rename(item, col)
            return

        old_name = item.data(0, Qt.ItemDataRole.UserRole)
        new_name = item.text(col).strip()

        # --- ALL POSSIBLE SCENARIOS WHERE WE DON'T MODIFY RECORDINGS ---
        # no change
        if new_name == old_name:
            return
        # if empty, revert
        if not new_name:
            self._revert_item(item, old_name)
            return 
        if new_name in self.recordings:
            QMessageBox.warning(self, "Name already exists", f"A recording named '{new_name}' already exists.")
            self._revert_item(item, old_name)
            return
        
        # SUCCESS
        rec = self.recordings.pop(old_name, None) # remove old name from dict
        if rec is None: # ermm... return!!
            self._revert_item(item, old_name)
            return
        self.recordings[new_name] = rec # add new name to dict

        # update stored UserRole
        self._suppress_item_changed = True
        item.setData(col, Qt.ItemDataRole.UserRole, new_name)
        self._suppress_item_changed = False

        # if this item is currently selected, update active_recording reference
        if self.active_recording == old_name:
            self.active_recording = new_name
            self.selected.emit(new_name)

    def _handle_score_rename(self, item: QTreeWidgetItem, col: int):
        """Apply an edit to the MIDI_ROOT label as a new Score Title. Reverts on
        an empty name, otherwise stores it and emits `score_renamed` so the score
        viewer re-renders with the new title."""
        old_title = item.data(0, Qt.ItemDataRole.UserRole) or ""
        new_title = item.text(col).strip()

        if not new_title:
            self._revert_item(item, old_title)
            return
        if new_title == old_title:
            return

        self._suppress_item_changed = True
        item.setText(col, new_title)
        item.setData(col, Qt.ItemDataRole.UserRole, new_title)
        self._suppress_item_changed = False
        self.score_renamed.emit(new_title)

    def _revert_item(self, item: QTreeWidgetItem, old_name: str):
        """
        Call this from TuneBuddy if rename was rejected.
        Reverts the currently edited item's label back to old_name.
        """
        self._suppress_item_changed = True
        item.setText(0, old_name)
        item.setData(0, Qt.ItemDataRole.UserRole, old_name)
        self._suppress_item_changed = False