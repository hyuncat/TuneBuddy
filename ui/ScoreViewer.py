import base64
from pathlib import Path
from PyQt6.QtCore import QUrl, pyqtSignal
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineSettings

from app_logic.midi.ScoreData import ScoreData


class ScoreViewer(QWebEngineView):
    """
    Wrapper around Verovio webviewer. Loads viewer.html file, 
    which loads the Verovio toolkit and provides a JS API for 
    loading scores and controlling playback.
    """
    load_finished = pyqtSignal(bool)

    def __init__(self, project_root: Path, parent=None):
        super().__init__(parent)
        self._root = Path(project_root).resolve()

        # allow local file access for loading scores
        s = self.settings()
        s.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        # s.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)

        self.js_ready = False
        self.loadFinished.connect(self._init_finished)

        html_path = self._root / "resources" / "verovio" / "viewer.html"
        self.setUrl(QUrl.fromLocalFile(str(html_path)))

    def _init_finished(self, ok: bool):
        """Called when viewer.html file is done loading.
        Switches JS api on, then emits load_finished signal.
        """
        print(f"[ScoreViewer] loadFinished: {ok}")
        self.js_ready = True
        self.load_finished.emit(ok)

    # --- JS API wrappers ---
    def load_score(self, score: ScoreData, channel: int | None = None) -> int:
        """
        Load a score into the viewer. Reads MusicXML bytes, encodes as base64,
        then sends to the JS API to load into Verovio.

        This is the expensive (re-layout) path and is only meant to run on
        instrument change / full-score toggle / score load — never per playback
        tick (that's set_playback_time).

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
        self.page().runJavaScript(f'window.loadScore("{b64}");')
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

        self.page().runJavaScript(f'window.timeChanged({sec:.6f});')