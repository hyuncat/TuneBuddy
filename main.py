import sys
from PyQt6.QtWidgets import QApplication
from app.app import virtuOS
from app.config import AppConfig


def run() -> int:
    """
    Initializes the application and runs it.

    Returns:
        int: The exit status code.
    """
    app: QApplication = QApplication(sys.argv)
    AppConfig.initialize(app)

    window: virtuOS = virtuOS()
    window.show()
    return sys.exit(app.exec())


if __name__ == '__main__':
    sys.exit(run())