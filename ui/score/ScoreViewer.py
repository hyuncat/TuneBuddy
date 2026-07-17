import base64
import json
from pathlib import Path
from PyQt6.QtCore import Qt, QUrl, QObject, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QWidget, QLabel, QStackedLayout, QApplication
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineSettings

from app_logic.midi.ScoreData import ScoreData
from ui.Colors import Colors


class _ViewerBridge(QObject):
    """JS -> Python push channel, registered as 'bridge' on the page's
    QWebChannel. The rest of the JS API is pull-only (Python runs JS and reads
    the result), but clicks originate in the page, so they need a push path."""
    note_clicked = pyqtSignal(float)  # clicked note's onset, Verovio-timeline sec
    annotation_clicked = pyqtSignal(float)  # clicked annotation target, app-time sec

    @pyqtSlot(float)
    def noteClicked(self, sec: float):
        self.note_clicked.emit(float(sec))

    @pyqtSlot(float)
    def annotationClicked(self, sec: float):
        self.annotation_clicked.emit(float(sec))


class _WebView(QWebEngineView):
    """QWebEngineView that suppresses the default context menu (Back/Reload/...
    is useless here) and reports right-clicks to the owning ScoreViewer, which
    turns them into trim confirmations while measure selection is armed."""
    right_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._right_click_emit_pending = False

    def contextMenuEvent(self, event):
        event.accept()
        self._emit_right_clicked_after_release()

    def _emit_right_clicked_after_release(self, attempts: int = 0):
        """Suppress the web context menu, then emit after the right button is up.

        Showing the trim dialog synchronously from contextMenuEvent can leave Qt's
        global button state looking like L+R to the next widget. Deferring until
        release keeps GuitarHero's native PyQtGraph drag handling intact.
        """
        if attempts == 0:
            if self._right_click_emit_pending:
                return
            self._right_click_emit_pending = True

        buttons = QApplication.mouseButtons()
        if not (buttons & Qt.MouseButton.RightButton) or attempts >= 30:
            self._right_click_emit_pending = False
            self.right_clicked.emit()
            return

        QTimer.singleShot(16, lambda: self._emit_right_clicked_after_release(attempts + 1))


class ScoreViewer(QWidget):
    """
    Wrapper around the Verovio webviewer. Loads viewer.html (which loads the
    Verovio toolkit and exposes a JS API for loading scores + controlling
    playback) into an internal QWebEngineView.

    The widget owns its own loading state: until Verovio's JS API has finished
    loading it shows a centered "Loading..." placeholder, then swaps to the live
    web view. Hosts just embed a ScoreViewer and wait for ``load_finished`` —
    they no longer build the loading screen / stacked layout themselves.
    """
    load_finished = pyqtSignal(bool)
    note_clicked = pyqtSignal(float)  # a note was clicked (Verovio-timeline sec)
    annotation_clicked = pyqtSignal(float)  # a mistake marker was clicked (app-time sec)
    trim_requested = pyqtSignal()     # right-click while measure selection is armed
    content_height_changed = pyqtSignal(float)  # rendered score height (px), after each load

    def __init__(self, parent=None):
        super().__init__(parent)
        self.js_ready = False
        self.selection_mode = False
        self._mistake_annotations = {"notes": {}, "insertions": [], "noteMeta": {}, "volumes": {}}
        self._annotation_color_mode = "pitch"

        # the real Verovio web view, hidden behind a "Loading..." placeholder
        # until its JS API is ready (swapped in _init_finished).
        self._view = _WebView()
        self._view.right_clicked.connect(self._on_right_clicked)
        # allow local file access for loading scores
        s = self._view.settings()
        s.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        # s.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)

        # JS->Python push channel (note clicks); registered before setUrl so the
        # transport is injected into the page as it loads.
        self._bridge = _ViewerBridge()
        self._bridge.note_clicked.connect(self.note_clicked)
        self._bridge.annotation_clicked.connect(self.annotation_clicked)
        self._channel = QWebChannel(self._view.page())
        self._channel.registerObject("bridge", self._bridge)
        self._view.page().setWebChannel(self._channel)

        self._loading = QLabel("Loading...")
        self._loading.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # stacked: loading placeholder on top until the JS API signals ready
        self._stack = QStackedLayout(self)
        self._stack.setContentsMargins(0, 0, 0, 0)
        self._stack.addWidget(self._loading)
        self._stack.addWidget(self._view)
        self._stack.setCurrentWidget(self._loading)

        self._view.loadFinished.connect(self._init_finished)
        # the Verovio web app (viewer.html/js/css + WASM) lives alongside this
        # class in ui/score/verovio
        html_path = Path(__file__).resolve().parent / "verovio" / "viewer.html"
        self._view.setUrl(QUrl.fromLocalFile(str(html_path)))

    def _init_finished(self, ok: bool):
        """Called when viewer.html is done loading. Switch the JS API on, swap the
        loading placeholder for the live viewer, then emit load_finished.
        """
        print(f"[ScoreViewer] loadFinished: {ok}")
        self.js_ready = True
        self._push_theme()
        self._stack.setCurrentWidget(self._view if ok else self._loading)
        self.load_finished.emit(ok)

    def _push_theme(self) -> None:
        """Hand the page its palette (ui.Colors owns every color in the app; the
        page holds none of its own). Pushed before any score loads, so the CSS
        custom properties viewer.css reads are set by the first render."""
        if not self.js_ready:
            return
        self._view.page().runJavaScript(
            f'window.setThemeColors({json.dumps(Colors.score_theme())});'
        )

    # --- JS API wrappers ---
    def load_score(self, score: ScoreData, channel: int | None = None) -> int:
        """
        Load a score into the viewer. Reads MusicXML bytes, encodes as base64,
        then sends to the JS API to load into Verovio. Rk: Expensive!

        Args:
            score: ScoreData object to load into the viewer
            channel: if given, render ONLY that instrument's part; otherwise
                render the full score.

        Returns:
            0 on success, 1 if JS API not ready yet (score not loaded)
        """
        if not self.js_ready:
            print("[ScoreViewer] load_score called before JS API ready, ignoring.")
            return 1
        xml_bytes = score.to_musicxml_bytes(channel=channel)
        b64 = base64.b64encode(xml_bytes).decode("ascii")
        active_part_index = self._active_part_index(score, channel)

        def _after_load(_=None):
            self.set_annotation_color_mode(self._annotation_color_mode)
            self._apply_mistake_annotations()
            self._view.page().runJavaScript(
                'window.getContentHeight();', self._emit_content_height)

        self._view.page().runJavaScript(
            f'window.loadScore("{b64}", {active_part_index});',
            _after_load,
        )
        return 0

    def _emit_content_height(self, height):
        """Rendered-content height (CSS px) pulled after a load; hosts use it to
        size the score pane so the whole system fits without scrolling."""
        if isinstance(height, (int, float)) and height > 0:
            self.content_height_changed.emit(float(height))

    @staticmethod
    def _active_part_index(score: ScoreData, channel: int | None) -> int:
        """Part index inside the MusicXML pushed to Verovio.

        A single-instrument render contains only one part, so annotations target
        part 0. A full-score render contains the real score parts in channel
        order; the active instrument's index lets JS map score-note indices only
        onto that staff instead of counting notes from every displayed part.
        """
        if channel is not None:
            return 0
        real_channels = [
            ch for ch in score.instruments
            if ch != score.metronome_channel
        ]
        try:
            return real_channels.index(score.active_instrument)
        except ValueError:
            return 0

    def set_mistake_annotations(self, annotations: dict | None) -> None:
        """Push score-note-indexed mistake markers into the web score.

        Shape:
          {"notes": {"12": [mistake, ...]}, "insertions": [slot, ...],
           "noteMeta": {"12": meta}, "volumes": {"12": volume}}
        Python computes score-note indices from NoteData/alignment; JS only maps
        those indices onto current Verovio note IDs and draws the page-local SVG.
        """
        self._mistake_annotations = annotations or {
            "notes": {}, "insertions": [], "noteMeta": {}, "volumes": {},
        }
        self._apply_mistake_annotations()

    def clear_mistake_annotations(self) -> None:
        self.set_mistake_annotations({
            "notes": {}, "insertions": [], "noteMeta": {}, "volumes": {},
        })

    def set_annotation_color_mode(self, mode: str) -> None:
        """Switch the web score among pitch mistakes, timing mistakes, and volume."""
        normalized = str(mode).lower()
        self._annotation_color_mode = (
            normalized if normalized in {"pitch", "timing", "volume"} else "pitch"
        )
        if not self.js_ready:
            return
        self._view.page().runJavaScript(
            f'window.setAnnotationColorMode({json.dumps(self._annotation_color_mode)});'
        )

    def _apply_mistake_annotations(self) -> None:
        if not self.js_ready:
            return
        payload = json.dumps(self._mistake_annotations)
        self._view.page().runJavaScript(f'window.setMistakeAnnotations({payload});')

    def set_playback_time(self, sec: float) -> None:
        """Set the current playback time in seconds. Should be called
        during playback to update the currently highlighted notehead
        in the score viewer.

        Args:
            sec: current playback/recording time in seconds
        """
        if not self.js_ready:
            print("[ScoreViewer] set_playback_time called before JS API ready, ignoring.")
            return

        self._view.page().runJavaScript(f'window.timeChanged({sec:.6f});')

    def get_measure_timemap(self, callback) -> None:
        """Pull Verovio's per-measure onset times (sec, original-tempo timeframe)
        as an ordered list — the barline anchors the host pairs with the score's
        own measure onsets to keep the cursor on the MIDI/NoteData timeline (see
        ui.time.ScoreTimeMap). `callback` is invoked (async) with the list, or
        None if the map isn't available yet."""
        if not self.js_ready:
            callback(None)
            return
        self._view.page().runJavaScript('window.getMeasureTimemap();', callback)

    # --- CLIP SELECTION BRIDGE ---
    # Measure-range clipping is driven from the JS (the user clicks a start/end
    # measure there); these wrappers PULL the current selection on demand and
    # PUSH the active clip-focus range back. Clipping deliberately never calls
    # load_score / re-lays-out the score — only CSS classes change in the page.
    def set_selection_mode(self, on: bool) -> None:
        """Arm/disarm measure-range picking in the page (the Clip menu's 'Select
        measures'). Armed: measure clicks build the start/end selection and
        note-click seeks are suspended; a right-click emits trim_requested so
        the host can confirm + clip. Disarmed: clicks fall back to note seeks."""
        self.selection_mode = bool(on)
        if not self.js_ready:
            return
        js_on = "true" if self.selection_mode else "false"
        self._view.page().runJavaScript(f'window.setSelectionMode({js_on});')

    def _on_right_clicked(self):
        """Right-click in the web view: a trim confirmation while selecting,
        otherwise nothing (the default context menu is suppressed either way)."""
        if self.selection_mode:
            self.trim_requested.emit()

    def get_clip_selection(self, callback) -> None:
        """Pull the in-progress measure selection from the JS. `callback` is
        invoked (async) with a dict {'startIdx', 'endIdx'} of inclusive measure
        indices (score order), or None if nothing is selected. Indices, not
        seconds, so the clip can't drift from Verovio's rendered timeline."""
        if not self.js_ready:
            callback(None)
            return
        self._view.page().runJavaScript('window.getClipSelection();', callback)

    def clear_clip_selection(self) -> None:
        """Clear the in-progress selection + its highlight in the viewer."""
        if not self.js_ready:
            return
        self._view.page().runJavaScript('window.clearClipSelection();')

    def set_clip_range(self, start_idx: int, end_idx: int) -> None:
        """Grey out every measure OUTSIDE the inclusive measure-index span
        [start_idx, end_idx] (the clip focus). Indices in score order, derived by
        Python from the clip's notes (ScoreData.clip_measure_range)."""
        if not self.js_ready:
            return
        self._view.page().runJavaScript(
            f'window.setClipRange({int(start_idx)}, {int(end_idx)});'
        )

    def clear_clip_range(self) -> None:
        """Remove the clip-focus grey-out (un-dim all measures)."""
        if not self.js_ready:
            return
        self._view.page().runJavaScript('window.clearClipRange();')
