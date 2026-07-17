from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout


class ClipDialog(QDialog):
    """Confirmation popup shown on right-click while measure selection is armed:
    a centered "Trim?" with No / Yes below. Use ClipDialog.ask(parent) -> bool."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Clip")
        self.setModal(True)

        layout = QVBoxLayout(self)
        label = QLabel("Clip?")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)

        buttons = QHBoxLayout()
        no_button = QPushButton("No")
        yes_button = QPushButton("Yes")
        no_button.clicked.connect(self.reject)
        yes_button.clicked.connect(self.accept)
        yes_button.setDefault(True)
        buttons.addStretch()
        buttons.addWidget(no_button)
        buttons.addWidget(yes_button)
        buttons.addStretch()
        layout.addLayout(buttons)

    @staticmethod
    def ask(parent=None) -> bool:
        """Show the popup modally; True iff the user confirmed the trim."""
        return ClipDialog(parent).exec() == QDialog.DialogCode.Accepted
