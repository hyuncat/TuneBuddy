from pathlib import Path

from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QGuiApplication, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer

_ICON_DIR = Path(__file__).resolve().parents[1] / "resources" / "icons"


def svg_pixmap(filename: str, px: int) -> QPixmap:
    """Rasterize one of the white-on-disk icon SVGs into a square pixmap,
    rendered at the screen's device pixel ratio so it stays crisp on retina."""
    screen = QGuiApplication.primaryScreen()
    dpr = screen.devicePixelRatio() if screen else 1.0
    renderer = QSvgRenderer(str(_ICON_DIR / filename))
    pix = QPixmap(round(px * dpr), round(px * dpr))
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    renderer.render(painter, QRectF(0, 0, pix.width(), pix.height()))
    painter.end()
    pix.setDevicePixelRatio(dpr)
    return pix
