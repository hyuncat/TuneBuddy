from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel

from ui.Icons import svg_pixmap
from ui.info.Gradient import Gradient


class Legend:
    """Shared legend-row pieces (the GuitarHero legend + the PerformTab score
    legend build their rows from these)."""

    @staticmethod
    def swatch(rgb: tuple[int, int, int], text: str) -> QWidget:
        """A rounded-square color swatch + label pair."""
        item = QWidget()
        lay = QHBoxLayout(item)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(5)
        square = QLabel()
        square.setFixedSize(12, 12)
        square.setStyleSheet(f"background-color: rgb{tuple(rgb)}; border-radius: 3px;")
        lay.addWidget(square)
        lay.addWidget(QLabel(text))
        return item

    @staticmethod
    def gradient_strip(gradient: Gradient, width: int = 150) -> QWidget:
        """A ramp legend: a (?) carrying the gradient's `help_text`, then the
        strip flanked by its own two end captions. Pass the gradient with no
        `frac` so it draws plain, with no handle."""
        item = QWidget()
        lay = QHBoxLayout(item)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(5)
        lay.addWidget(Legend.help_icon(gradient.help_text))
        low, high = gradient.ends()
        gradient.setFixedWidth(width)
        lay.addWidget(low)
        lay.addWidget(gradient)
        lay.addWidget(high)
        return item

    @staticmethod
    def help_icon(text: str) -> QWidget:
        """A (?) whose tooltip explains a legend. Word-wraps because these
        blurbs are sentences, not the one-word tooltips Qt sizes for."""
        icon = QLabel()
        icon.setPixmap(svg_pixmap("circle-help.svg", 14))
        icon.setToolTip(f"<div style='max-width: 260px;'>{text}</div>")
        return icon
