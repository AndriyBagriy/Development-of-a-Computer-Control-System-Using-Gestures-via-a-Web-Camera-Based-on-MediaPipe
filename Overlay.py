from PyQt6.QtGui import QPainter, QColor
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QSizePolicy
from PyQt6.QtCore import Qt, QTimer


class Overlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.SubWindow)
        self.setGeometry(parent.rect())
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.label = QLabel("Wait hand on cam...", self)
        self.label.setStyleSheet("color: white; font-size: 26px;")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.timer_label = QLabel("", self)
        self.timer_label.setStyleSheet("color: white; font-size:32px;")
        self.timer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout(self)
        layout.addWidget(self.label)
        layout.addWidget(self.timer_label)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.is_transparent = False
        self.hide_text()

        self.countdown_timer = QTimer(self)
        self.countdown_value = 5
        self.collecting = False
        # self.hand_present = False

        self.hand = None

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.is_transparent:
            painter.setBrush(QColor(0, 0, 0, 0))
        else:
            painter.setBrush(QColor(0, 0, 0, 100))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(self.rect())
        super().paintEvent(event)

    def resizeEvent(self, event):
        self.setGeometry(self.parent().rect())
        super().resizeEvent(event)

    def resizeMy(self, geometry):
        self.setGeometry(geometry)

    def show_text(self, message: str = "Wait hand on cam..."):
        self.label.setText(message)
        self.is_transparent = False
        self.label.show()
        self.timer_label.hide()

    def start_countdown(self, start_collecting_callback, processor, hand: str):
        self.hand = hand
        self.collecting = False
        self.show_text("Hand detected")
        QTimer.singleShot(1000, lambda: self.label.setText("Data collection will start in..."))
        QTimer.singleShot(1000, self.start_timer)

        self.start_collecting_callback = start_collecting_callback

        self.countdown_check_timer = QTimer(self)
        self.countdown_check_timer.timeout.connect(
            lambda: self.check_for_hand(start_collecting_callback, processor))
        self.countdown_check_timer.start(200)

    def check_for_hand(self, start_collecting_callback, processor):
        handedness_dict = processor.get_handedness()
        if handedness_dict.get(self.hand) != self.hand:
            self.stop_countdown(start_collecting_callback, processor)
        # if processor.get_handedness() != "Right":
        #     self.stop_countdown(start_collecting_callback, processor)

    def start_timer(self):
        self.countdown_value = 5
        self.timer_label.show()
        self.update_timer_label()
        try:
            self.countdown_timer.timeout.disconnect(self.update_timer)
        except TypeError:
            pass
        self.countdown_timer.timeout.connect(self.update_timer)
        self.countdown_timer.start(1000)

    def update_timer(self):
        self.countdown_value -= 1
        if self.countdown_value == 0:
            self.countdown_timer.stop()
            self.start_collection()
        self.update_timer_label()

    def update_timer_label(self):
        self.timer_label.setText(str(self.countdown_value))

    def start_collection(self):
        if self.collecting:
            return
        self.collecting = True
        self.hide_text()
        self.start_collecting_callback()

    def stop_countdown(self, start_collecting_callback, processor):
        self.countdown_timer.stop()
        self.countdown_check_timer.stop()
        self.collecting = False
        self.show_text("Wait hand on cam...")

        def wait_for_hand():
            handedness = processor.get_handedness()
            if handedness.get(self.hand) == self.hand:
                self.start_countdown(start_collecting_callback, processor, self.hand)
            else:
                QTimer.singleShot(100, wait_for_hand)

        wait_for_hand()
        # def wait_for_hand(collecting_callback, inner_processor):
        #     if processor.get_handedness() is not None:
        #         self.start_countdown(collecting_callback, inner_processor, self.hand)
        #     else:
        #         QTimer.singleShot(100, lambda: wait_for_hand(start_collecting_callback, processor))
        # wait_for_hand(start_collecting_callback, processor)

    def hide_text(self):
        self.label.hide()
        self.timer_label.hide()
        self.is_transparent = True
        self.update()

    def hide_overlay(self):
        self.hide()
