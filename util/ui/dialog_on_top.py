from PyQt5 import QtWidgets, QtCore


class DialogOnTop(QtWidgets.QDialog):
    """QDialog that shows on top of the main window."""
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)
        self.setWindowModality(QtCore.Qt.ApplicationModal)  # Prevent interaction with main window
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
