from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal, QPoint
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem,
    QMenu, QMessageBox, QInputDialog,
)

from app_logic.user.ds.Recording import Recording
from app_logic.midi.ScoreData import ScoreData
from app_logic.JsonHandler import JsonHandler
from resources.program_map import program_to_name, name_to_program


class RecordingTree(QWidget):
    """
    Left-side panel showing the active score's recordings, or a folder library of
    score folders. Only the active score's children are backed by Recording
    objects; every other item stores file paths and is hydrated by app.py on
    selection.
    """
    selected = pyqtSignal(object)           # active recording name, or None
    score_renamed = pyqtSignal(str)         # active score title changed
    score_file_selected = pyqtSignal(str, object)  # score path, optional recording path
    recording_renamed = pyqtSignal(str, str) # old recording name, new recording name

    SCORE_EXTENSIONS = {".mid", ".midi", ".mxl", ".musicxml", ".xml", ".mei"}
    AUDIO_EXTENSIONS = {
        ".wav", ".wave", ".aif", ".aiff", ".flac", ".ogg", ".oga",
        ".mp3", ".m4a", ".aac", ".opus",
    }

    NAME_ROLE = Qt.ItemDataRole.UserRole
    KIND_ROLE = Qt.ItemDataRole.UserRole + 1
    SCORE_PATH_ROLE = Qt.ItemDataRole.UserRole + 2
    RECORDING_PATH_ROLE = Qt.ItemDataRole.UserRole + 3

    KIND_FOLDER = "folder"
    KIND_SCORE = "score"
    KIND_RECORDING = "recording"

    def __init__(self, recordings: dict[str, Recording], parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(6, 6, 6, 6)

        self.MIDI_ROOT: QTreeWidgetItem | None = None
        self.score_data: ScoreData | None = None
        self.recordings = recordings
        self.active_recording: str | None = None
        self.current_score_path: str | None = None
        self.library_root_path: Path | None = None
        self._score_items_by_path: dict[str, QTreeWidgetItem] = {}

        self._suppress_item_changed = False
        self._suppress_selection_changed = False

        self.init_ui()
        self.init_signals()

    def init_ui(self):
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(14)
        self._layout.addWidget(self.tree)

        self.setMinimumWidth(180)
        self.setMaximumWidth(320)

    def init_signals(self):
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.open_context_menu)
        self.tree.itemSelectionChanged.connect(self.on_selection_changed)
        self.tree.itemChanged.connect(self.on_item_changed)
        self.tree.itemDoubleClicked.connect(self.rename_recording)
        self.tree.setEditTriggers(
            self.tree.EditTrigger.EditKeyPressed |
            self.tree.EditTrigger.SelectedClicked
        )

    # --- SCORE / FOLDER INITIALIZATION ---
    def init_score(self, filepath: str | Path, score_data: ScoreData = None):
        """Reset tree for one directly uploaded score."""
        self._suppress_selection_changed = True
        self.tree.clear()
        self.recordings.clear()
        self.active_recording = None
        self.library_root_path = None
        self._score_items_by_path = {}

        score_path = self._path_key(filepath)
        score_name = Path(filepath).stem
        self.MIDI_ROOT = QTreeWidgetItem([score_name])
        self._configure_score_item(self.MIDI_ROOT, score_path, score_name)
        # Preserve the old single-score behavior: the score title is editable but
        # not selectable as a recording.
        flags = self.MIDI_ROOT.flags()
        self.MIDI_ROOT.setFlags(
            (flags & ~Qt.ItemFlag.ItemIsSelectable) | Qt.ItemFlag.ItemIsEditable
        )
        self.tree.addTopLevelItem(self.MIDI_ROOT)
        self.MIDI_ROOT.setExpanded(True)

        self.score_data = score_data
        self.current_score_path = score_path
        self._score_items_by_path[score_path] = self.MIDI_ROOT
        self._suppress_selection_changed = False

    def folder_contains_score(self, folder: str | Path) -> bool:
        """Whether `folder` holds any loadable score file (SCORE_EXTENSIONS) —
        the precondition for init_folder to produce a non-empty library."""
        try:
            return any(
                path.is_file() and path.suffix.lower() in self.SCORE_EXTENSIONS
                for path in Path(folder).rglob("*")
            )
        except OSError as e:
            print(f"Could not scan folder '{folder}': {e}")
            return False

    def init_folder(self, folder_path: str | Path) -> list[str]:
        """Scan `folder_path` and populate a recursive score/recording library."""
        root_path = Path(folder_path)
        self._suppress_selection_changed = True
        self.tree.clear()
        self.recordings.clear()
        self.active_recording = None
        self.current_score_path = None
        self.score_data = None
        self.MIDI_ROOT = None
        self.library_root_path = root_path
        self._score_items_by_path = {}

        root_item = QTreeWidgetItem([root_path.name])
        self._configure_folder_item(root_item)
        self.tree.addTopLevelItem(root_item)
        self._populate_directory(root_path, root_item)
        root_item.setExpanded(True)

        self._suppress_selection_changed = False
        return list(self._score_items_by_path.keys())

    def set_active_score(self, filepath: str | Path, score_data: ScoreData = None):
        """Mark the already-displayed score item as the active in-memory score."""
        score_path = self._path_key(filepath)
        self.current_score_path = score_path
        self.score_data = score_data
        self.MIDI_ROOT = self._score_items_by_path.get(score_path)
        if self.MIDI_ROOT is None:
            self.MIDI_ROOT = QTreeWidgetItem([Path(filepath).stem])
            self._configure_score_item(self.MIDI_ROOT, score_path, Path(filepath).stem)
            self.tree.addTopLevelItem(self.MIDI_ROOT)
            self._score_items_by_path[score_path] = self.MIDI_ROOT
        self.MIDI_ROOT.setExpanded(True)

    def score_paths(self) -> list[str]:
        return list(self._score_items_by_path.keys())

    def score_title(self, filepath: str | Path) -> str | None:
        item = self._score_items_by_path.get(self._path_key(filepath))
        return item.text(0).strip() if item is not None else None

    def recording_entries_for_score(self, filepath: str | Path) -> list[tuple[str, Path]]:
        """Direct recording children for a score item, as (name, file_path)."""
        score_item = self._score_items_by_path.get(self._path_key(filepath))
        if score_item is None:
            return []
        entries: list[tuple[str, Path]] = []
        for i in range(score_item.childCount()):
            child = score_item.child(i)
            if child.data(0, self.KIND_ROLE) != self.KIND_RECORDING:
                continue
            rec_path = child.data(0, self.RECORDING_PATH_ROLE)
            if rec_path:
                entries.append((child.data(0, self.NAME_ROLE) or child.text(0), Path(rec_path)))
        return entries

    def recording_name_for_path(self, score_path: str | Path, recording_path: str | Path) -> str | None:
        item = self._find_recording_item_by_path(score_path, recording_path)
        return item.data(0, self.NAME_ROLE) if item is not None else None

    # --- CONTEXT MENU ---
    def open_context_menu(self, pos: QPoint):
        menu = QMenu(self)
        item = self.tree.itemAt(pos)
        kind = item.data(0, self.KIND_ROLE) if item is not None else None

        new_action = menu.addAction("New Recording…") if self.current_score_path else None
        rename_action = None
        delete_action = None
        inst_select_action = None

        if kind == self.KIND_RECORDING:
            rename_action = menu.addAction("Rename")
            delete_action = menu.addAction("Delete")
            inst_select_action = menu.addAction("Select Instrument")
        elif kind == self.KIND_SCORE:
            rename_action = menu.addAction("Rename")

        action = menu.exec(self.tree.viewport().mapToGlobal(pos))
        if action is None:
            return

        if action == new_action:
            self.new_recording()
        elif action == rename_action:
            self.rename_recording(item)
        elif action == delete_action:
            if self.confirm_delete(item):
                self.delete_recording(item)
        elif action == inst_select_action:
            self.select_instrument(item)
        else:
            print("Unknown context menu action:", action)

    def new_recording(self):
        """Prompt user for a new in-memory recording under the active score."""
        if self.MIDI_ROOT is None or self.score_data is None:
            QMessageBox.warning(self, "No score selected", "Please select a score first.")
            return
        name, ok = QInputDialog.getText(self, "New Recording", "Recording name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        if name in self.recordings:
            QMessageBox.warning(self, "Name already exists", f"A recording named '{name}' already exists.")
            return
        self._add_recording(name)

    def rename_recording(self, item: QTreeWidgetItem | None = None, col: int = 0):
        if item is not None:
            self.tree.editItem(item, col)

    def delete_recording(self, item: QTreeWidgetItem):
        if item is None or item.data(0, self.KIND_ROLE) != self.KIND_RECORDING:
            return
        name = item.data(0, self.NAME_ROLE) or item.text(0)
        recording_path = item.data(0, self.RECORDING_PATH_ROLE)
        if not recording_path and name in self.recordings:
            audio_path = self.recordings[name].audio_filepath
            recording_path = str(audio_path) if audio_path else None

        if recording_path and not self._delete_recording_file(recording_path):
            return

        self.recordings.pop(name, None)
        parent = item.parent() or self.tree.invisibleRootItem()
        parent.removeChild(item)

        if self.active_recording == name:
            self.tree.setCurrentItem(None)
            self.active_recording = None
            self.selected.emit(None)

    def select_instrument(self, item: QTreeWidgetItem):
        if item is None or item.data(0, self.KIND_ROLE) != self.KIND_RECORDING:
            return
        name = item.data(0, self.NAME_ROLE) or item.text(0)
        rec = self.recordings.get(name)
        if rec is None:
            QMessageBox.warning(self, "Recording not loaded", "Select this recording before changing its instrument.")
            return

        score_data = rec.score_data
        instruments = score_data.instruments
        if not instruments:
            QMessageBox.warning(self, "No instruments found", "The loaded score has no instruments to select.")
            return

        items = [f"{program_to_name(prog)}" for _, prog in instruments.items()]
        if items:
            items.pop(-1)  # remove metronome sound from selection
        item, ok = QInputDialog.getItem(self, "Select Instrument", "Instruments:", items, 0, False)
        if not ok or not item:
            return

        try:
            prog_num = name_to_program(item)
            ch_num = next(ch for ch, prog in instruments.items() if prog == prog_num)
            score_data.active_instrument = ch_num
            self.selected.emit(name)
        except Exception as e:
            print("Error parsing selected instrument:", e)
            QMessageBox.warning(self, "Invalid selection", "Could not parse the selected instrument.")

    # --- RECORDING ITEM API USED BY APP.PY ---
    def set_recording_name(self, new_name: str, old_name: str | None = None) -> str | None:
        new_name = (new_name or "").strip()
        if not new_name:
            return None
        if old_name is None:
            old_name = self.active_recording
        if old_name is None or old_name not in self.recordings:
            return None

        item = self._find_item(old_name)
        if item is None:
            return None

        new_name = self._unique_name(new_name, ignore=old_name)
        if new_name == old_name:
            return old_name

        self.recordings[new_name] = self.recordings.pop(old_name)
        self._suppress_item_changed = True
        item.setText(0, new_name)
        item.setData(0, self.NAME_ROLE, new_name)
        self._suppress_item_changed = False

        if self.active_recording == old_name:
            self.active_recording = new_name
        return new_name

    def update_recording_file(self, name: str, filepath: str | Path, score_path: str | Path | None = None):
        """Attach/refresh the saved file path for a recording item."""
        item = self._find_item(name) if score_path is None else self._find_recording_item(score_path, name)
        if item is None:
            item = self.ensure_recording_item(name, filepath=filepath, score_path=score_path)
        if item is None:
            return
        item.setData(0, self.RECORDING_PATH_ROLE, self._path_key(filepath))

    def ensure_recording_item(
        self,
        name: str,
        filepath: str | Path | None = None,
        score_path: str | Path | None = None,
        select: bool = False,
    ) -> QTreeWidgetItem | None:
        """Create a recording tree item if one does not already exist."""
        score_item = self._score_item(score_path)
        if score_item is None:
            return None
        if filepath is not None:
            existing = self._find_recording_item_by_path(
                score_item.data(0, self.SCORE_PATH_ROLE), filepath
            )
            if existing is not None:
                return existing
        existing = self._find_recording_item(score_item.data(0, self.SCORE_PATH_ROLE), name)
        if existing is not None:
            return existing

        item = self._make_recording_item(
            name=name,
            score_path=score_item.data(0, self.SCORE_PATH_ROLE),
            filepath=filepath,
        )
        score_item.addChild(item)
        score_item.setExpanded(True)
        if select:
            self.tree.setCurrentItem(item)
        return item

    def select_recording_name(
        self,
        name: str,
        score_path: str | Path | None = None,
        emit: bool = True,
    ) -> bool:
        item = self._find_recording_item(score_path or self.current_score_path, name)
        if item is None:
            return False
        was_current = self.tree.currentItem() is item
        self._suppress_selection_changed = not emit
        self.tree.setCurrentItem(item)
        if not emit:
            self.active_recording = name
        self._suppress_selection_changed = False
        if emit and was_current:
            self.on_selection_changed()
        return True

    def select_recording_path(
        self,
        score_path: str | Path,
        recording_path: str | Path,
        emit: bool = True,
    ) -> bool:
        item = self._find_recording_item_by_path(score_path, recording_path)
        if item is None:
            return False
        was_current = self.tree.currentItem() is item
        self._suppress_selection_changed = not emit
        self.tree.setCurrentItem(item)
        if not emit:
            self.active_recording = item.data(0, self.NAME_ROLE)
        self._suppress_selection_changed = False
        if emit and was_current:
            self.on_selection_changed()
        return True

    def select_score(self, score_path: str | Path, emit: bool = True) -> bool:
        item = self._score_items_by_path.get(self._path_key(score_path))
        if item is None:
            return False
        self._suppress_selection_changed = not emit
        self.tree.setCurrentItem(item)
        self._suppress_selection_changed = False
        return True

    # --- INTERNAL TREE HELPERS ---
    def _add_recording(self, name: str):
        """Create a new unsaved Recording under the active score and select it."""
        if name in self.recordings:
            print(f"Recording with name '{name}' already exists. Skipping creation.")
            return
        rec = Recording(score_data=self._new_score_data_for_current_score())
        self.recordings[name] = rec
        self.ensure_recording_item(name, select=True)

    def _new_score_data_for_current_score(self) -> ScoreData:
        """Fresh score instance for a new recording under the active score."""
        if not self.current_score_path:
            return self.score_data
        score_data = ScoreData()
        score_data.load(self.current_score_path)
        title = self.score_title(self.current_score_path)
        if title:
            score_data.set_title(title)
        return score_data

    def _find_item(self, name: str) -> QTreeWidgetItem | None:
        return self._find_recording_item(self.current_score_path, name)

    def _find_recording_item(self, score_path: str | Path | None, name: str) -> QTreeWidgetItem | None:
        score_item = self._score_item(score_path)
        if score_item is None:
            return None
        for i in range(score_item.childCount()):
            child = score_item.child(i)
            if child.data(0, self.KIND_ROLE) == self.KIND_RECORDING and child.data(0, self.NAME_ROLE) == name:
                return child
        return None

    def _find_recording_item_by_path(
        self, score_path: str | Path | None, recording_path: str | Path
    ) -> QTreeWidgetItem | None:
        score_item = self._score_item(score_path)
        if score_item is None:
            return None
        rec_path = self._path_key(recording_path)
        for i in range(score_item.childCount()):
            child = score_item.child(i)
            if child.data(0, self.KIND_ROLE) != self.KIND_RECORDING:
                continue
            if child.data(0, self.RECORDING_PATH_ROLE) == rec_path:
                return child
        return None

    def _score_item(self, score_path: str | Path | None = None) -> QTreeWidgetItem | None:
        if score_path is None:
            return self.MIDI_ROOT
        return self._score_items_by_path.get(self._path_key(score_path))

    def _unique_name(self, base: str, ignore: str | None = None) -> str:
        taken = set(self.recordings.keys())
        taken.discard(ignore)
        if base not in taken:
            return base
        n = 2
        while f"{base} ({n})" in taken:
            n += 1
        return f"{base} ({n})"

    def _unique_child_name(self, score_item: QTreeWidgetItem, base: str) -> str:
        taken = set()
        for i in range(score_item.childCount()):
            child = score_item.child(i)
            if child.data(0, self.KIND_ROLE) == self.KIND_RECORDING:
                taken.add(child.data(0, self.NAME_ROLE) or child.text(0))
        if base not in taken:
            return base
        n = 2
        while f"{base} ({n})" in taken:
            n += 1
        return f"{base} ({n})"

    def confirm_delete(self, item: QTreeWidgetItem | None) -> bool:
        if item is None:
            return False
        name = item.data(0, self.NAME_ROLE) or item.text(0)
        if not name:
            return False
        ok = QMessageBox.question(
            self,
            "Delete recording",
            "Are you sure? This permanently deletes your recording from your machine.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        return ok == QMessageBox.StandardButton.Yes

    def _delete_recording_file(self, recording_path: str | Path) -> bool:
        path = Path(recording_path)
        if not path.exists():
            JsonHandler.delete_cache_for(path)
            return True
        if not path.is_file():
            QMessageBox.warning(
                self,
                "Delete failed",
                f"Could not delete recording:\n{path} is not a file.",
            )
            return False
        try:
            path.unlink()
            JsonHandler.delete_cache_for(path)
        except OSError as e:
            QMessageBox.warning(
                self,
                "Delete failed",
                f"Could not delete recording:\n{e}",
            )
            return False
        return True

    def on_selection_changed(self):
        if self._suppress_selection_changed:
            return
        item = self.tree.currentItem()
        if item is None:
            self.active_recording = None
            return

        kind = item.data(0, self.KIND_ROLE)
        if kind == self.KIND_RECORDING:
            name = item.data(0, self.NAME_ROLE)
            score_path = item.data(0, self.SCORE_PATH_ROLE)
            recording_path = item.data(0, self.RECORDING_PATH_ROLE)
            if score_path != self.current_score_path:
                self.score_file_selected.emit(score_path, recording_path)
                return
            if name not in self.recordings:
                print(f"Recording '{name}' is not loaded for the active score.")
                return
            self.active_recording = name
            self.selected.emit(name)
        elif kind == self.KIND_SCORE:
            self.active_recording = None
            score_path = item.data(0, self.SCORE_PATH_ROLE)
            if score_path != self.current_score_path:
                self.score_file_selected.emit(score_path, None)
        else:
            self.active_recording = None

    def get_selection(self) -> QTreeWidgetItem | None:
        item = self.tree.currentItem()
        if item is not None and item.data(0, self.KIND_ROLE) == self.KIND_RECORDING:
            return item
        return None

    def on_item_changed(self, item: QTreeWidgetItem, col: int):
        if self._suppress_item_changed or item is None:
            return

        kind = item.data(0, self.KIND_ROLE)
        if kind == self.KIND_SCORE:
            self._handle_score_rename(item, col)
            return
        if kind != self.KIND_RECORDING:
            return

        old_name = item.data(0, self.NAME_ROLE)
        new_name = item.text(col).strip()
        if new_name == old_name:
            return
        if not new_name:
            self._revert_item(item, old_name)
            return
        if Path(new_name).name != new_name:
            QMessageBox.warning(self, "Invalid name", "Recording names cannot contain path separators.")
            self._revert_item(item, old_name)
            return
        if new_name in self.recordings:
            QMessageBox.warning(self, "Name already exists", f"A recording named '{new_name}' already exists.")
            self._revert_item(item, old_name)
            return

        rec = self.recordings.get(old_name)
        new_audio_path = None
        if rec is not None:
            try:
                new_audio_path = JsonHandler.rename_recording_files(rec, new_name)
            except FileExistsError as e:
                target = e.filename or (e.args[0] if e.args else "")
                QMessageBox.warning(
                    self,
                    "Rename failed",
                    f"Could not rename recording:\n{Path(target).name} already exists.",
                )
                self._revert_item(item, old_name)
                return
            except OSError as e:
                QMessageBox.warning(self, "Rename failed", f"Could not rename recording:\n{e}")
                self._revert_item(item, old_name)
                return

        if old_name in self.recordings:
            rec = self.recordings.pop(old_name)
            self.recordings[new_name] = rec

        self._suppress_item_changed = True
        item.setData(col, self.NAME_ROLE, new_name)
        if new_audio_path is not None:
            item.setData(col, self.RECORDING_PATH_ROLE, self._path_key(new_audio_path))
        self._suppress_item_changed = False

        if self.active_recording == old_name:
            self.active_recording = new_name
            self.selected.emit(new_name)
        self.recording_renamed.emit(old_name, new_name)

    def _handle_score_rename(self, item: QTreeWidgetItem, col: int):
        old_title = item.data(0, self.NAME_ROLE) or ""
        new_title = item.text(col).strip()

        if not new_title:
            self._revert_item(item, old_title)
            return
        if new_title == old_title:
            return

        self._suppress_item_changed = True
        item.setText(col, new_title)
        item.setData(col, self.NAME_ROLE, new_title)
        self._suppress_item_changed = False

        if item.data(0, self.SCORE_PATH_ROLE) == self.current_score_path:
            self.score_renamed.emit(new_title)

    def _revert_item(self, item: QTreeWidgetItem, old_name: str):
        self._suppress_item_changed = True
        item.setText(0, old_name)
        item.setData(0, self.NAME_ROLE, old_name)
        self._suppress_item_changed = False

    # --- FOLDER SCANNING ---
    def _populate_directory(self, directory: Path, item: QTreeWidgetItem) -> bool:
        same_named_score = self._same_named_score(directory)
        relevant = False

        if same_named_score is not None:
            self._configure_score_item(item, self._path_key(same_named_score), directory.name)
            self._score_items_by_path[self._path_key(same_named_score)] = item
            for audio_path in self._audio_files(directory):
                name = self._unique_child_name(item, audio_path.stem)
                item.addChild(self._make_recording_item(name, self._path_key(same_named_score), audio_path))
            relevant = True
        else:
            self._configure_folder_item(item)
            for score_path in self._score_files(directory):
                score_item = QTreeWidgetItem([score_path.stem])
                self._configure_score_item(score_item, self._path_key(score_path), score_path.stem)
                self._score_items_by_path[self._path_key(score_path)] = score_item
                item.addChild(score_item)
                relevant = True

        for child_dir in sorted((p for p in directory.iterdir() if p.is_dir()), key=lambda p: p.name.lower()):
            child_item = QTreeWidgetItem([child_dir.name])
            if self._populate_directory(child_dir, child_item):
                item.addChild(child_item)
                relevant = True

        if relevant:
            item.setExpanded(True)
        return relevant

    def _same_named_score(self, directory: Path) -> Path | None:
        for path in self._score_files(directory):
            if path.stem == directory.name:
                return path
        return None

    def _score_files(self, directory: Path) -> list[Path]:
        return sorted(
            (p for p in directory.iterdir()
             if p.is_file() and p.suffix.lower() in self.SCORE_EXTENSIONS),
            key=lambda p: p.name.lower(),
        )

    def _audio_files(self, directory: Path) -> list[Path]:
        return sorted(
            (p for p in directory.iterdir()
             if p.is_file() and p.suffix.lower() in self.AUDIO_EXTENSIONS),
            key=lambda p: p.name.lower(),
        )

    def _configure_folder_item(self, item: QTreeWidgetItem):
        item.setData(0, self.KIND_ROLE, self.KIND_FOLDER)
        flags = item.flags()
        item.setFlags((flags & ~Qt.ItemFlag.ItemIsEditable) | Qt.ItemFlag.ItemIsEnabled)

    def _configure_score_item(self, item: QTreeWidgetItem, score_path: str, title: str):
        item.setText(0, title)
        item.setData(0, self.NAME_ROLE, title)
        item.setData(0, self.KIND_ROLE, self.KIND_SCORE)
        item.setData(0, self.SCORE_PATH_ROLE, score_path)
        item.setData(0, self.RECORDING_PATH_ROLE, None)
        item.setFlags(
            item.flags() | Qt.ItemFlag.ItemIsEditable
            | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
        )

    def _make_recording_item(
        self,
        name: str,
        score_path: str,
        filepath: str | Path | None = None,
    ) -> QTreeWidgetItem:
        item = QTreeWidgetItem([name])
        item.setData(0, self.NAME_ROLE, name)
        item.setData(0, self.KIND_ROLE, self.KIND_RECORDING)
        item.setData(0, self.SCORE_PATH_ROLE, score_path)
        item.setData(0, self.RECORDING_PATH_ROLE, self._path_key(filepath) if filepath else None)
        item.setFlags(
            item.flags() | Qt.ItemFlag.ItemIsEditable
            | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
        )
        return item

    @staticmethod
    def _path_key(path: str | Path | None) -> str | None:
        if path is None:
            return None
        return str(Path(path).expanduser().resolve())
