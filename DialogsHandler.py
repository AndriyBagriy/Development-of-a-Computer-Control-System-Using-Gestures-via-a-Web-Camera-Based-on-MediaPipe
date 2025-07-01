import re
from PyQt6.QtWidgets import QDialog, QMessageBox, QVBoxLayout, QFormLayout, QLineEdit, QHBoxLayout, QPushButton

from add_gesture import Ui_Dialog


class AddGestureDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        model_cb = self.ui.cbModelGestD
        model_cb.addItems(['Right', 'Left'])

        num_data_cb = self.ui.cbDataGestD
        num_data_cb.addItems(['50', '100', '150', '200', '250', '300', '350'])
        num_data_cb.setCurrentIndex(3)

        self.ui.acceptBtn.clicked.connect(self.validate)
        self.ui.rejectBtn.clicked.connect(self.reject)

    def validate(self):
        data = self.get_data()
        if not re.fullmatch(r"^[a-zA-Z0-9_]{1,14}$", data["name"]):
            QMessageBox.warning(self, "Ошибка",
                                "Имя должно содержать только буквы, цифры и '_' и быть не длиннее 14 символов.")
            return
        if not data["name"] or not data["model"] or not data["data"]:
            QMessageBox.warning(self, "Ошибка", "Все поля (кроме описания) должны быть заполнены.")
            return
        self.accept()

    def get_data(self):
        return {
            "name": str(self.ui.nameGestD.text()),
            "model": self.ui.cbModelGestD.currentText(),
            "data": int(self.ui.cbDataGestD.currentText()),
            "descript": str(self.ui.descGestD.toPlainText())
        }


class NewCopyPresetDialog(QDialog):
    def __init__(self, parent, existing_names):
        super().__init__(parent)
        self.setWindowTitle("Create or Copy Preset")
        self.resize(300, 120)

        self.existing = set(existing_names)

        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.nameEdit = QLineEdit(self)
        form.addRow("Name:", self.nameEdit)
        layout.addLayout(form)

        btns = QHBoxLayout()
        self.btnCreateNew = QPushButton("Create new", self)
        self.btnCreateCopy = QPushButton("Create copy", self)
        self.btnCancel = QPushButton("Cancel", self)
        btns.addWidget(self.btnCreateNew)
        btns.addWidget(self.btnCreateCopy)
        btns.addStretch()
        btns.addWidget(self.btnCancel)
        layout.addLayout(btns)

        self.btnCancel.clicked.connect(self.reject)
        self.btnCreateNew.clicked.connect(self._on_create_new)
        self.btnCreateCopy.clicked.connect(self._on_create_copy)

    def _on_create_new(self):
        name = self.nameEdit.text().strip()
        if not name:
            QMessageBox.warning(self, "Error", "Name cannot be empty")
            return
        if name in self.existing:
            QMessageBox.warning(self, "Error", "Preset already exists")
            return
        self.result = ("new", name)
        self.accept()

    def _on_create_copy(self):
        name = self.nameEdit.text().strip()
        if not name:
            QMessageBox.warning(self, "Error", "Name cannot be empty")
            return
        if name in self.existing:
            QMessageBox.warning(self, "Error", "Preset already exists")
            return
        self.result = ("copy", name)
        self.accept()


class RenamePresetDialog(QDialog):
    def __init__(self, parent, old_name, existing_names):
        super().__init__(parent)
        self.setWindowTitle("Rename Preset")
        self.resize(300, 100)

        self.existing = set(existing_names) - {old_name}

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.nameEdit = QLineEdit(old_name, self)
        form.addRow("New name:", self.nameEdit)
        layout.addLayout(form)

        h = QHBoxLayout()
        self.btnRename = QPushButton("Rename", self)
        self.btnCancel = QPushButton("Cancel", self)
        h.addStretch()
        h.addWidget(self.btnRename)
        h.addWidget(self.btnCancel)
        layout.addLayout(h)

        self.btnCancel.clicked.connect(self.reject)
        self.btnRename.clicked.connect(self._on_rename)

    def _on_rename(self):
        name = self.nameEdit.text().strip()
        if not name:
            QMessageBox.warning(self, "Error", "Name cannot be empty")
            return
        if name in self.existing:
            QMessageBox.warning(self, "Error", "Preset already exists")
            return
        self.new_name = name
        self.accept()