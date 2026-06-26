import base64
from pathlib import Path
from PyQt6.QtCore import Qt, QUrl, pyqtSignal
from PyQt6.QtWidgets import QWidget, QLabel, QStackedLayout
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineSettings

from app_logic.midi.ScoreData import ScoreData


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

    def __init__(self, project_root: Path, parent=None):
        super().__init__(parent)
        self._root = Path(project_root).resolve()
        self.js_ready = False

        # the real Verovio web view, hidden behind a "Loading..." placeholder
        # until its JS API is ready (swapped in _init_finished).
        self._view = QWebEngineView()
        # allow local file access for loading scores
        s = self._view.settings()
        s.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        # s.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)

        self._loading = QLabel("Loading...")
        self._loading.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # stacked: loading placeholder on top until the JS API signals ready
        self._stack = QStackedLayout(self)
        self._stack.setContentsMargins(0, 0, 0, 0)
        self._stack.addWidget(self._loading)
        self._stack.addWidget(self._view)
        self._stack.setCurrentWidget(self._loading)

        self._view.loadFinished.connect(self._init_finished)
        html_path = self._root / "resources" / "verovio" / "viewer.html"
        self._view.setUrl(QUrl.fromLocalFile(str(html_path)))

    def _init_finished(self, ok: bool):
        """Called when viewer.html is done loading. Switch the JS API on, swap the
        loading placeholder for the live viewer, then emit load_finished.
        """
        print(f"[ScoreViewer] loadFinished: {ok}")
        self.js_ready = True
        self._stack.setCurrentWidget(self._view if ok else self._loading)
        self.load_finished.emit(ok)

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
        self._view.page().runJavaScript(f'window.loadScore("{b64}");')
        return 0

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
