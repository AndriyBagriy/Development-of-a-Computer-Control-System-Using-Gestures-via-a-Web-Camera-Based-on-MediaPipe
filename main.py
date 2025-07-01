import copy
import csv
import json
import traceback
from collections import deque
from datetime import datetime
from typing import Optional

from colorama import Fore, Style
import re
import cv2
import os

from ActionController import GestureBinder
from DialogsHandler import AddGestureDialog, NewCopyPresetDialog, RenamePresetDialog
from GestureListWidget import GestureListWidget, PlaceholderItem
from MouseController import MouseController
from Overlay import Overlay

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import mediapipe as mp
import sys
from PyQt6 import uic
from PyQt6.QtCore import QTimer, Qt, QPoint, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QCursor, QGuiApplication, QShortcut, QKeySequence, QIcon, QColor
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QFrame, QScrollArea, QWidget, QHBoxLayout, QPushButton, \
    QVBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QDialog, QMessageBox, QListWidgetItem, \
    QGroupBox, QListWidget, QGridLayout
from screeninfo import get_monitors

from FrameProcessor import FrameProcessor
from GestureClassifier import GestureClassifier
from GestureDataCollector import GestureDataCollector
from PredictionThread import PredictionThread
from gui import Ui_MainWindow
from add_gesture import Ui_Dialog

os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"

QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
# tf.debugging.set_log_device_placement(True)


# class AddGestureDialog(QDialog):
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.ui = Ui_Dialog()
#         self.ui.setupUi(self)
#
#         model_cb = self.ui.cbModelGestD
#         model_cb.addItems(['Right', 'Left'])
#
#         num_data_cb = self.ui.cbDataGestD
#         num_data_cb.addItems(['50', '100', '150', '200', '250', '300', '350'])
#         num_data_cb.setCurrentIndex(3)
#
#         self.ui.acceptBtn.clicked.connect(self.validate)
#         self.ui.rejectBtn.clicked.connect(self.reject)
#
#     def validate(self):
#         data = self.get_data()
#         if not re.fullmatch(r"^[a-zA-Z0-9_]{1,14}$", data["name"]):
#             QMessageBox.warning(self, "Ошибка",
#                                 "Имя должно содержать только буквы, цифры и '_' и быть не длиннее 14 символов.")
#             return
#         if not data["name"] or not data["model"] or not data["data"]:
#             QMessageBox.warning(self, "Ошибка", "Все поля (кроме описания) должны быть заполнены.")
#             return
#         self.accept()
#
#     def get_data(self):
#         return {
#             "name": str(self.ui.nameGestD.text()),
#             "model": self.ui.cbModelGestD.currentText(),
#             "data": int(self.ui.cbDataGestD.currentText()),
#             "descript": str(self.ui.descGestD.toPlainText())
#         }

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.left_hand_landmarks = None
        for m in get_monitors():
            if m.is_primary:
                self.monitor_width = m.width
                self.monitor_height = m.height

        self.processor = None
        self.classifier = None

        uic.loadUi('gui.ui', self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.camView = self.ui.camView
        self.camViewTraining = self.ui.camViewTraining
        self.testObj = self.ui.testObj


        # Right hand
        self.dataset_path = 'model/keypoint_data.csv'
        model_save_path = 'model/keypoint_classifier.keras'
        self.tflite_right_path = 'model/keypoint_classifier.tflite'
        backup_path = 'model/backup_keypoint.csv'
        self.gesture_path = 'model/gestures.csv'
        self.gesture_buffer_path = "buffer/gesture_buffer.csv"
        self.classifier = GestureClassifier(self.dataset_path, model_save_path, self.tflite_right_path, backup_path)

        # Left hand
        self.dataset_path_left = 'model/keypoint_data_left.csv'
        model_save_path_left = 'model/keypoint_classifier_left.keras'
        self.tflite_left_path = 'model/keypoint_classifier_left.tflite'
        backup_path_left = 'model/backup_keypoint_left.csv'
        self.gesture_path_left = 'model/gestures_left.csv'
        self.gesture_buffer_path_left = "buffer/gesture_buffer_left.csv"
        self.classifier_left = GestureClassifier(self.dataset_path_left, model_save_path_left, self.tflite_left_path,
                                                 backup_path_left)

        self.bindings_path = 'model/bindings.json'
        self.load_bindings()

        self.variant_lists = []
        self.initGui()
        self.init_bindings_ui()

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_drawing = mp.solutions.drawing_utils

        self.camera_width = 1920  # self.monitor_width
        self.camera_height = 1080  # self.monitor_height
        margin_x = int(self.camera_width * 0.05)
        margin_y = int(self.camera_height * 0.15)
        self.control_area_width = int(854 * (self.monitor_width / self.monitor_width))  # 1920))
        self.control_area_height = int(480 * (self.monitor_height / self.monitor_height))  # 1080))
        self.control_area_x = self.camera_width - self.control_area_width - margin_x
        self.control_area_y = self.camera_height - self.control_area_height - margin_y
        control_area = (self.control_area_x, self.control_area_y, self.control_area_width, self.control_area_height)
        self.buffer = int(self.control_area_height * 0.05)

        self.gesture_buffer = deque([-1] * 10, maxlen=10)
        self.gesture_buffer_left = deque([-1] * 10, maxlen=10)

        self.current_streak_right = {"gesture_id": None, "count": 0}
        self.current_streak_left = {"gesture_id": None, "count": 0}

        self.stability_threshold = 3

        self.mouse_controller = MouseController(
            self, self.monitor_width, self.gesture_buffer, self.monitor_height, control_area, self.buffer
        )

        self.gesture_collector_right = GestureDataCollector(self.gesture_buffer_path, self.gesture_path)
        self.gesture_collector_left = GestureDataCollector(self.gesture_buffer_path_left, self.gesture_path_left)

        self.shortcut = QShortcut(QKeySequence("F9"), self)
        self.shortcut.activated.connect(self.collect_gesture_data)
        self.shortcut1 = QShortcut(QKeySequence("F10"), self)
        self.shortcut1.activated.connect(self.predict)
        # self.shortcut2 = QShortcut(QKeySequence("F11"), self)
        # self.shortcut2.activated.connect(self.start_model)
        self.thread = None


        self.pred_thread_right = None
        self.prediction_timer_right = QTimer()
        self.prediction_timer_right.timeout.connect(lambda: self.launch_prediction('Right'))
        # self.prediction_timer_right.timeout.connect(
        #     lambda: self.predict(self.classifier, self.gesture_collector_right, 'Right'))

        self.pred_thread_left = None
        self.prediction_timer_left = QTimer()
        self.prediction_timer_left.timeout.connect(lambda: self.launch_prediction('Left'))
        # self.prediction_timer_left.timeout.connect(
        #     lambda: self.predict(self.classifier_left, self.gesture_collector_left, 'Left'))





        QTimer.singleShot(0, self.update_sizes)

        self.scroll_widget_buffer = self.ui.scrollWidgetBuffer
        self.scroll_widget_layout = QVBoxLayout(self.scroll_widget_buffer)
        self.scroll_widget_layout.setSpacing(10)
        self.scroll_widget_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_widget_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.update_gest_buffer()

        self.gesture_binder = GestureBinder()

        self.overlay = Overlay(self)
        self.overlay.hide()
        # self.shortcut1 = QShortcut(QKeySequence("F10"), self)
        # self.shortcut1.activated.connect(self.start_model())

        # self.effective_area_width = self.control_area_width - 2 * self.buffer
        # self.effective_area_height = self.control_area_height - 2 * self.buffer
        # self.prev_cursor_x = None
        # self.prev_cursor_y = None
        # self.smoothing_factor = 0.5
        # self.movement_threshold = 7

    # def keyPressEvent(self, event):
    #     if event.key() == Qt.Key.Key_F3:
    #         self.tracking = not self.tracking

    def launch_prediction(self, hand: str):
        thread_attr = 'pred_thread_' + hand.lower()
        if getattr(self, thread_attr) is not None and getattr(self, thread_attr).isRunning():
            return

        if hand == 'Right':
            tflite_path = self.tflite_right_path
            collector = self.gesture_collector_right
        else:
            tflite_path = self.tflite_left_path
            collector = self.gesture_collector_left

        thread = PredictionThread(
            hand_label=hand,
            processor=self.processor,
            tflite_model_path=tflite_path,
            gesture_collector=collector
        )
        thread.prediction_done.connect(self.handle_prediction_result)
        setattr(self, thread_attr, thread)
        thread.start()

    def collect_gesture_data(self, gesture_name, data_count, hand):
        self.overlay.show()
        self.overlay.show_text()
        self.prediction_timer_right.stop()
        self.prediction_timer_left.stop()
        self.hand_detected_count = 0
        collector = self.gesture_collector_right if hand == 'Right' else self.gesture_collector_left

        def wait_for_hand():
            handedness_dict = self.processor.get_handedness()
            print("hand = " + str(handedness_dict))

            if handedness_dict.get(hand) == hand:
                self.hand_detected_count += 1
                if self.hand_detected_count >= 3:
                    self.block_interface()
                    self.overlay.start_countdown(
                        lambda: collector.start_collecting(
                            self.processor,
                            gesture_name,
                            data_count,
                            on_finish=self.enable_interactivity,
                            hand=hand
                        ),
                        self.processor,
                        hand
                    )
                else:
                    QTimer.singleShot(100, wait_for_hand)
            else:
                self.hand_detected_count = 0
                QTimer.singleShot(100, wait_for_hand)

        wait_for_hand()
        # self.overlay.show()
        # self.overlay.show_text()
        # self.hand_detected = 0
        #
        # def wait_for_hand():
        #     if self.processor.get_handedness()["Right"] is not None:
        #         self.hand_detected = self.hand_detected + 1
        #         if self.hand_detected >= 3:
        #             self.start(self.ui.camViewTraining)
        #             self.block_interface()
        #             if self.processor:
        #                 self.overlay.start_countdown(lambda: self.gesture_collector_right.start_collecting(
        #                     self.processor, gesture_name, data_count, self.enable_interactivity
        #                 ), self.processor)
        #                 # self.gesture_collector.start_collecting(
        #                 #     self.processor, gesture_name, data_count, self.enable_interactivity)
        #         else:
        #             QTimer.singleShot(100, wait_for_hand)
        #     else:
        #         QTimer.singleShot(100, wait_for_hand)
        #
        # wait_for_hand()

    def block_interface(self):
        self.overlay.hide_text()

    def enable_interactivity(self):
        self.overlay.hide_overlay()
        self.prediction_timer_right.start(100)
        self.prediction_timer_left.start(100)

    def predict(self, classifier, collector, hand):
        #self.thread = PredictionThread(self.processor, self.classifier, self.gesture_collector, hand)
        self.thread = PredictionThread(self.processor, classifier, collector, hand)

        self.thread.prediction_done.connect(self.handle_prediction_result)
        self.thread.start()

    def handle_prediction_result(self, hand_label, gesture_id):
        # if gesture_id == -1:
        #     return
        # print(Fore.GREEN+f"Gesture id = {gesture_id}")
        #
        # if self.current_streak["gesture_id"] == gesture_id:
        #     self.current_streak["count"] += 1
        # else:
        #     self.current_streak["gesture_id"] = gesture_id
        #     self.current_streak["count"] = 1
        #
        # if (self.current_streak["count"] >= self.stability_threshold
        #         and (not self.gesture_buffer or self.gesture_buffer[-1] != gesture_id)):
        #     self.gesture_buffer.append(gesture_id)
        #     #self.gesture_binder.execute(self.gesture_buffer)
        #
        # print(f"Буфер жестов: {list(self.gesture_buffer)}")
        if gesture_id == -1:
            return
        buffer = self.gesture_buffer if hand_label == 'Right' else self.gesture_buffer_left
        streak = self.current_streak_right if hand_label == 'Right' else self.current_streak_left

        if streak['gesture_id'] == gesture_id:
            streak['count'] += 1
        else:
            streak['gesture_id'] = gesture_id
            streak['count'] = 1

        if streak['count'] >= self.stability_threshold and (not buffer or buffer[-1] != gesture_id):
            buffer.append(gesture_id)
            print(f"{hand_label} buffer updated: {list(buffer)}")

    #TODO оптимизировать под две руки
    def train_model(self):
        """
        dataset_path = 'keypoint_data.csv'
        model_save_path = 'keypoint_classifier.keras'
        tflite_save_path = 'keypoint_classifier.tflite'
        backup_path = 'backup_keypoint.csv'

        classifier = GestureClassifier(dataset_path, model_save_path, tflite_save_path, backup_path)
        """
        self.save_gesture()
        if self.classifier:
            self.classifier.update_model()
            self.classifier.save_as_tflite()

    def train_model_new(self, classifier, buffer_path, gesture_path, dataset_path):
        self._save_gesture(buffer_path, gesture_path, dataset_path)
        classifier.update_model()
        classifier.save_as_tflite()

    # TODO оптимизировать под две руки
    def save_gesture(self):
        if os.path.getsize(self.gesture_buffer_path) == 0:
            return
        with open(self.gesture_buffer_path, mode="r", newline="") as file:
            reader = csv.reader(file)
            lines = list(reader)

        if not lines:
            return

        with open(self.gesture_path, mode="a", newline="") as gesture_file, \
                open(self.dataset_path, mode="a", newline="") as dataset_file:

            gesture_writer = csv.writer(gesture_file)
            dataset_writer = csv.writer(dataset_file)
            i = 0
            gest_id = sum(1 for _ in open(self.gesture_path))
            while i < len(lines):
                line = lines[i]
                if line[0].startswith("@") and line[2] == "Right":
                    gesture_writer.writerow(line[1:])
                    # gest_id = self.gesture_collector.get_gesture_id(line[1])
                    print("id = ", gest_id)
                    i += 1
                    while i < len(lines) and not lines[i][0].startswith("@"):
                        lines[i][0] = gest_id
                        dataset_writer.writerow(lines[i])
                        i += 1
                    gest_id += 1
                else:
                    i += 1

    def _save_gesture(self, buffer_path, gesture_path, dataset_path):
        if not os.path.exists(buffer_path) or os.path.getsize(buffer_path) == 0:
            return

        with open(buffer_path, mode="r", newline="", encoding="utf-8") as f:
            lines = list(csv.reader(f))
        if not lines:
            return
        with open(gesture_path, mode="a", newline="", encoding="utf-8") as gf, \
                open(dataset_path, mode="a", newline="", encoding="utf-8") as df:
            gesture_writer = csv.writer(gf)
            dataset_writer = csv.writer(df)
            gest_id = sum(1 for _ in open(gesture_path, 'r', encoding='utf-8'))

            i = 0
            while i < len(lines):
                row = lines[i]
                if row and row[0].startswith("@"):
                    name, hand, data_count, desc, date = row[1:]
                    gesture_writer.writerow([name, hand, data_count, desc, date])
                    i += 1
                    while i < len(lines) and not lines[i][0].startswith("@"):
                        line = lines[i]
                        line[0] = str(gest_id)
                        dataset_writer.writerow(line)
                        i += 1
                    gest_id += 1
                else:
                    i += 1

        open(buffer_path, "w").close()

    #TODO оптимизировать под две руки
    def get_getsures_data(self):
        # file_path = "model/gestures.csv"
        # data = []
        # with open(file_path, mode='r') as file:
        #     reader = csv.reader(file)
        #     next(reader)
        #     for row in reader:
        #         data.append(row)
        # return data
        data = []
        for path in (self.gesture_path, self.gesture_path_left):
            if not os.path.exists(path):
                continue
            with open(path, mode='r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if len(row) >= 3:
                        data.append(row)
        return data

    def start(self, cam_view, ):
        self.camView = cam_view
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # self.monitor_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # self.monitor_height)
            # self.timer.start(60)
            self.processor = FrameProcessor(self.cap, self.mp_hands, self.mp_drawing,
                                            self.gesture_buffer, self.gesture_buffer_left,
                                            self.gesture_path, self.gesture_path_left)
            self.processor.frame_ready.connect(self.update_frame)

            # ////////self.processor.hand_data_ready.connect(self.process_hand_data)

            self.processor.right_hand_data_ready.connect(self.process_right_hand_data)
            self.processor.left_hand_data_ready.connect(self.process_left_hand_data)

            self.processor.start()
            self.prediction_timer_right.start(100)
            self.prediction_timer_left.start(100)

    def stop(self):
        self.prediction_timer_right.stop()
        self.prediction_timer_left.stop()
        if self.processor:
            self.processor.stop()
            self.processor = None

        if self.cap is not None:
            # self.timer.stop()
            self.cap.release()
            self.cap = None
        self.camView.clear()

    def update_frame(self, q_img):
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(self.camView.size(), Qt.AspectRatioMode.KeepAspectRatio)
        self.camView.setPixmap(scaled_pixmap)

    def process_right_hand_data(self, index_finger, middle_finger, thumb_finger, ring_finger, pinky_finger, frame_width,
                                frame_height, handedness):
        if handedness == 'Right':
            self.mouse_controller.update_tracking(index_finger, middle_finger, frame_width, frame_height)
            self.mouse_controller.move_cursor(middle_finger, frame_width, frame_height)
            self.mouse_controller.check_for_click(thumb_finger, ring_finger, pinky_finger, frame_width, frame_height)

            # self.mouse_controller.check_for_click(thumb_finger, index_finger, middle_finger, frame_width, frame_height)

    def process_left_hand_data(self, index, middle, thumb, ring, pinky, w, h, handedness, detected):
        if detected:
            self.left_hand_landmarks = [(index, middle, thumb, ring, pinky), w, h]
        else:
            self.left_hand_landmarks = None

    def update_sizes(self):
        parent_widget = self.camView.parent()
        parent_width = parent_widget.width()
        parent_height = parent_widget.height()

        new_width = int(parent_height * 16 / 9)
        new_height = int(parent_width * 9 / 16)

        if new_width > parent_width:
            new_width = parent_width
        if new_height > parent_height:
            new_height = parent_height

        self.camView.setFixedWidth(new_width)
        self.camView.setFixedHeight(new_height)

        self.camView.setMinimumSize(160, 90)

        if new_width == parent_width and new_height == parent_height:
            parent_widget.setFixedSize(parent_width, parent_height)

        right_frame = self.testObj.parent()
        left_frame_min_width = 400
        cur_size = right_frame.width()
        if cur_size < left_frame_min_width:
            cur_size = left_frame_min_width

        obj_new_width = cur_size + (parent_width - new_width)
        right_frame.setFixedWidth(obj_new_width - 22)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.overlay.resizeMy(self.rect())
        self.update_sizes()



    def initGui(self):
        self.ui.startBtn.clicked.connect(lambda: self.start(self.ui.camView))
        self.ui.stopBtn.clicked.connect(self.stop)
        self.ui.addGesture.clicked.connect(self.add_gesture)
        # self.ui.clearBuffer.clicked.connect(lambda: (self.start(self.ui.camViewTraining),
        #                                              self.collect_gesture_data("Test 2 Gesture", 20)))
        self.ui.clearBufferBtn.clicked.connect(self.clearBuffer)
        # self.ui.trainModel.clicked.connect(self.save_gesture)


        # TODO запуск тренировки
        # self.ui.trainModel.clicked.connect(self.train_model)
        self.ui.trainRightModel.clicked.connect(lambda: self.train_model_new(
            classifier=self.classifier,
            buffer_path=self.gesture_buffer_path,
            gesture_path=self.gesture_path,
            dataset_path=self.dataset_path
            )
        )

        self.ui.trainLeftModel.clicked.connect(lambda: self.train_model_new(
            classifier=self.classifier_left,
            buffer_path=self.gesture_buffer_path_left,
            gesture_path=self.gesture_path_left,
            dataset_path=self.dataset_path_left
            )
        )

        # self.startBtn.clicked.connect(self.start)
        # self.stopBtn.clicked.connect(self.stop)

        # scroll_widget = self.ui.scrollWidget
        # scroll_layout = QVBoxLayout(scroll_widget)
        # scroll_layout.setSpacing(10)
        # scroll_layout.setContentsMargins(0, 0, 0, 0)
        # scroll_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.presetListLayout = QVBoxLayout(self.ui.presetScrollWidget)
        self.presetListLayout.setSpacing(5)
        self.presetListLayout.setContentsMargins(0, 0, 0, 0)
        self.presetListLayout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.init_preset_list()



        self.ui.modelBtn.clicked.connect(lambda: (self.ui.mainStackedWidget.setCurrentIndex(1),
                                                  setattr(self, 'camView', self.ui.camViewTraining),
                                                  self.update_sizes()))
        self.ui.backBtn.clicked.connect(lambda: (self.ui.mainStackedWidget.setCurrentIndex(0),
                                                 setattr(self, 'camView', self.ui.camView),
                                                 self.update_sizes()))

        self.ui.gestureTab.mousePressEvent = lambda event: self.ui.modelStackedWidget.setCurrentIndex(0)
        self.ui.trainingTab.mousePressEvent = lambda event: self.ui.modelStackedWidget.setCurrentIndex(1)
        self.ui.settingTab.mousePressEvent = lambda event: self.ui.modelStackedWidget.setCurrentIndex(2)

        # for i in range(4):
        #     self.add_preset_row(scroll_layout, f"Preset {i + 1}")


        data = self.get_getsures_data()
        table = self.ui.tableWidget
        table.setRowCount(len(data))
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Name", "Hand", "Num of training data"])
        for row_index, row_data in enumerate(data):
            for col_index, value in enumerate(row_data[:3]):
                item = QTableWidgetItem(value)
                if col_index in (1, 2):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                table.setItem(row_index, col_index, item)
            # for col_index, value in enumerate(row_data):
            #     item = QTableWidgetItem(value)
            #     if col_index == 1 or col_index == 2:
            #         item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            #     table.setItem(row_index, col_index, item)



        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)

        table.setVerticalScrollMode(table.ScrollMode.ScrollPerPixel)
        table.setHorizontalScrollMode(table.ScrollMode.ScrollPerPixel)

        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        # table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        table.selectionModel().selectionChanged.connect(self.on_selection_changed)

    def on_selection_changed(self, selected, deselected):
        table = self.ui.tableWidget
        rows = selected.indexes()
        if rows:
            row = rows[0].row()
            name = table.item(row, 0).text()
            data = self.get_getsures_data()
            matching_row = (next((item for item in data if item[0] == name), None))
            self.ui.gestName.setText(matching_row[0])
            self.ui.gestHand.setText(matching_row[1])
            self.ui.gestData.setText(matching_row[2])
            self.ui.gestDesc.setPlainText(matching_row[3])
            self.ui.gestDate.setText(matching_row[4])

    def add_gesture(self):
        dialog = AddGestureDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        data = dialog.get_data()
        buffer = self.gesture_buffer_path if data["model"] == 'Right' else self.gesture_buffer_path_left
        with open(buffer, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["@", data["name"], data["model"], data["data"], data["descript"],
                             datetime.now().strftime("%d.%m.%Y")])
        self.start(self.ui.camViewTraining)
        self.collect_gesture_data(data["name"], data["data"], data["model"])

        self.update_gest_buffer()

    def update_gest_buffer(self):
        while self.scroll_widget_layout.count():
            item = self.scroll_widget_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        entries = []
        for path in (self.gesture_buffer_path, self.gesture_buffer_path_left):
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if row and row[0].startswith("@"):
                        entries.append((row, path))

        # entries.sort(key=lambda e: datetime.strptime(e[0][5], "%d.%m.%Y"))
        for row, source_path in entries:
            self.add_gest_row(row)
        """
        while self.scroll_widget_layout.count():
            item = self.scroll_widget_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            else:
                self.scroll_widget_layout.removeItem(item)

        with open(self.gesture_buffer_path, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            lines = list(reader)
            for i, line in enumerate(lines):
                if line[0].startswith("@"):
                    self.add_gest_row(line)
                    # if i + 1 < len(lines) and not lines[i + 1][0].startswith("@"):
                    #     print("123")
        """
    def clearBuffer(self):
        open(self.gesture_buffer_path, "w").close()
        open(self.gesture_buffer_path_left, "w").close()
        self.update_gest_buffer()
        pass

    def add_gest_row(self, file_row):
        row_widget = QWidget()
        row_widget.setFixedHeight(50)
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(5, 5, 5, 5)
        row_layout.setSpacing(10)

        name_label = QLabel(file_row[1])
        name_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        model_label = QLabel(file_row[2])
        model_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        edit_button = QPushButton()
        edit_button.setFixedSize(40, 40)
        edit_button.setIcon(QIcon("icons/install.png"))
        edit_button.setIconSize(edit_button.size())
        delete_button = QPushButton()
        delete_button.setFixedSize(20, 20)
        delete_button.setIcon(QIcon("icons/install.png"))
        delete_button.setIconSize(delete_button.size())

        row_layout.addWidget(name_label)
        row_layout.addStretch()
        row_layout.addWidget(model_label)
        row_layout.addWidget(edit_button)
        row_layout.addWidget(delete_button)
        self.scroll_widget_layout.addWidget(row_widget)

    def init_preset_list(self):
        # container = self.ui.presetScrollWidget
        # layout = self.presetListLayout

        while self.presetListLayout.count():
            item = self.presetListLayout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        for idx, preset in enumerate(self.presets):
            row_widget = QWidget()
            row_widget.setFixedHeight(50)
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(5, 5, 5, 5)
            row_layout.setSpacing(10)


            name_label = QLabel(preset['name'])
            name_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            if idx == self.current_index:
                row_widget.setStyleSheet("background-color: rgb(32, 32, 32);")
            # if idx == self.current_index:
            #     name_label.setStyleSheet("font-weight: bold; color: darkBlue;")

            row_layout.addWidget(name_label)
            row_layout.addStretch()

            btn = QPushButton("Select")
            btn.setFixedSize(80, 30)
            btn.clicked.connect(lambda _, i=idx: self.set_active_preset(i))
            row_layout.addWidget(btn)
            self.presetListLayout.addWidget(row_widget)



    def set_active_preset(self, idx):
        self.current_index = idx
        self.current_preset = self.presets[idx]
        data = {
            "active": self.presets[idx]['name'],
            "presets": self.presets
        }
        with open(self.bindings_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self.init_preset_list()
        self.cbPresets.blockSignals(True)
        self.cbPresets.setCurrentIndex(self.current_index)
        self.cbPresets.blockSignals(False)
        self.on_preset_changed(idx)

    def getBuffer(self):
        return self.gesture_buffer


    def load_bindings(self):
        # with open(self.binding_path, encoding='utf-8') as f:
        #     data = json.load(f)
        # self.presets = data['presets']
        # self.current_preset_index = 0
        # self.current_preset = self.presets[self.current_preset_index]

        with open(self.bindings_path, encoding='utf-8') as f:
            data = json.load(f)
        self.presets = data.get('presets', [])
        # Выбираем активный
        name_to_idx = {p['name']: i for i, p in enumerate(self.presets)}
        # Если в JSON прописан active (имя) — найдём его
        active_name = data.get('active')
        if active_name in name_to_idx:
            self.current_index = name_to_idx[active_name]
        else:
            # fallback: по индексу
            idx = data.get('active_index', 0)
            self.current_index = max(0, min(idx, len(self.presets)-1))
        self.current_preset = self.presets[self.current_index]

    def init_bindings_ui(self):
        self.listActions = self.ui.listActions
        self.listActions.clear()
        for action_name, info in self.current_preset['actions'].items():
            item = QListWidgetItem(action_name)
            if info.get('variants'):
                item.setForeground(QColor('darkGreen'))
                item.setIcon(QIcon('icons/linked.png'))
            else:
                item.setForeground(QColor('red'))
                item.setIcon(QIcon('icons/unlinked.png'))
            item.setData(Qt.ItemDataRole.UserRole, info)
            self.listActions.addItem(item)

        # --- СПИСОК ЖЕСТОВ ПРАВОЙ РУКИ ---
        self.listRight = self.ui.listRightGestures
        self.listLeft = self.ui.listLeftGestures
        for lst, path, hand in (
                (self.listRight, self.gesture_path, 'Right'),
                (self.listLeft, self.gesture_path_left, 'Left'),
        ):
            lst.clear()
            with open(path, encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    name, _, count, desc, date = row
                    item = QListWidgetItem(f"{name} ({hand})")
                    item.setData(Qt.ItemDataRole.UserRole, reader.line_num - 1)
                    item.setData(Qt.ItemDataRole.UserRole + 1, hand)
                    lst.addItem(item)
            lst.setDragEnabled(True)

        self.btnAddVar = self.ui.btnAddVariant
        self.btnRemoveVar = self.ui.btnRemoveVariant

        self.mappingContainer = self.ui.scrollMappingContainer
        self.mappingLayout = self.mappingContainer.layout()

        self.btnAddVar.clicked.connect(self.add_variant)
        self.btnRemoveVar.clicked.connect(self.remove_variant)

        # self.lstSeq = self.ui.lstSeq
        # self.lstOther = self.ui.lstOther
        # self.lstSeq.max_items = 3
        # self.lstOther.max_items = 1
        # self.lstSeq.peer = self.lstOther
        # self.lstOther.peer = self.lstSeq

        first_seq = self.ui.lstSeq
        first_other = self.ui.lstOther
        self.variant_lists.append((first_seq, first_other))

        first_seq.max_items = 3
        first_other.max_items = 1
        first_seq.peer = first_other
        first_other.peer = first_seq

        self.cbPresets = self.ui.presetsBox
        self.cbPresets.clear()
        for p in self.presets:
            self.cbPresets.addItem(p['name'])
        self.cbPresets.setEditable(False)
        self.cbPresets.setCurrentIndex(self.current_index)
        self.cbPresets.currentIndexChanged.connect(self.on_preset_changed)

        self.listActions.itemClicked.connect(self.on_action_selected)

        self.ui.createPresetBtn.clicked.connect(self.on_newcopy_clicked)
        self.ui.renamePresetBtn.clicked.connect(self.on_rename_clicked)
        self.ui.delPresetBtn.clicked.connect(self.on_delete_clicked)
        self.ui.allDefaultBtn.clicked.connect(self.on_apply_default_bindings)

        self.ui.saveActBtn.clicked.connect(self.on_save_action_bind_clicked)
        self.ui.defActBtn.clicked.connect(self.on_default_action_bind_clicked)
        self.ui.clearActBtn.clicked.connect(self.on_clean_action_bind_clicked)
    def on_preset_changed(self, idx):
        self.current_index = idx
        self.current_preset = self.presets[idx]
        self.update_actions()


        # current = self.listActions.currentItem()
        # if current:
        #     self.on_action_selected(current)

    def update_actions(self):
        current = self.listActions.currentItem()
        if current:
            self.on_action_selected(current)

    def on_action_selected(self, item: QListWidgetItem):
        action_name = item.text()
        self.ui.actionNameLbl.setText(f"Action: {action_name}")
        self.removeAll_variants()

        seq0, oth0 = self.variant_lists[0]
        seq0.clear_gestures()
        oth0.clear_gestures()

        variants = self.current_preset['actions'].get(action_name, {}).get('variants', [])

        for idx, var in enumerate(variants):
            if idx >= len(self.variant_lists):
                self.add_variant()
            seq_list, other_list = self.variant_lists[idx]
            seq_list.clear_gestures()
            for gid in var.get('sequence', []):
                hand = var.get('sequence_hand', '')
                name = self.id_to_name(hand, gid) or str(gid)
                seq_list.insertItem(seq_list.count() - 1, QListWidgetItem(f"{name} ({hand})"))

            other_list.clear_gestures()
            oh = var.get('other_hand_gesture')
            if oh:
                gid = oh['id']
                hand2 = oh['hand']
                name2 = self.id_to_name(hand2, gid) or str(gid)
                other_list.insertItem(0, QListWidgetItem(f"{name2} ({hand2})"))
            seq_list.update_placeholders()
            other_list.update_placeholders()
        for seq_list, other_list in self.variant_lists:
            seq_list.peer = other_list
            other_list.peer = seq_list

    def add_variant(self):
        count = sum(
            1 for i in range(self.mappingLayout.count())
            if isinstance(self.mappingLayout.itemAt(i).widget(), QGroupBox)
        )
        new_index = count + 1

        buttons_widget = self.ui.btnAddVariant.parentWidget()
        insert_idx = self.mappingLayout.indexOf(buttons_widget)

        box = QGroupBox(f"Variant {new_index}", self.mappingContainer)
        grid = QGridLayout(box)

        from GestureListWidget import GestureListWidget
        lst_seq = GestureListWidget(box, max_items=3)
        lst_seq.setObjectName(f"lstSeq{new_index}")
        lst_seq.setFixedHeight(100)

        lst_other = GestureListWidget(box, max_items=1)
        lst_other.setObjectName(f"lstOther{new_index}")
        lst_other.setFixedHeight(40)

        lst_seq.peer = lst_other
        lst_other.peer = lst_seq

        lst_seq.setAcceptDrops(True)
        lst_other.setAcceptDrops(True)

        lbl_seq = QLabel("Sequence:", box)
        lbl_other = QLabel("On other hand gesture:", box)
        grid.addWidget(lbl_seq, 0, 0, alignment=Qt.AlignmentFlag.AlignTop)
        grid.addWidget(lst_seq, 0, 1, alignment=Qt.AlignmentFlag.AlignTop)
        grid.addWidget(lbl_other, 1, 0, alignment=Qt.AlignmentFlag.AlignTop)
        grid.addWidget(lst_other, 1, 1, alignment=Qt.AlignmentFlag.AlignTop)

        self.mappingLayout.insertWidget(insert_idx, box)

        self.variant_lists.append((lst_seq, lst_other))

        self.ui.scrollMappingArea.verticalScrollBar().setValue(
            self.ui.scrollMappingArea.verticalScrollBar().maximum()
        )

    def removeAll_variants(self):
        while len(self.variant_lists) > 1:
            seq, oth = self.variant_lists.pop()
            box = seq.parent()
            while not isinstance(box, QGroupBox):
                box = box.parent()
            self.mappingLayout.removeWidget(box)
            box.deleteLater()

    def remove_variant(self):
        groupboxes = [
            (i, self.mappingLayout.itemAt(i).widget())
            for i in range(self.mappingLayout.count())
            if isinstance(self.mappingLayout.itemAt(i).widget(), QGroupBox)
        ]
        if len(groupboxes) <= 1:
            return
        idx, box = groupboxes[-1]
        item = self.mappingLayout.takeAt(idx)
        box.deleteLater()
        self.variant_lists.pop()

    def on_newcopy_clicked(self):
        dlg = NewCopyPresetDialog(self, [p['name'] for p in self.presets])
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        mode, name = dlg.result
        if mode == "new":
            new_p = {"name": name, "actions": {k: {"variants": []} for k in self.presets[0]["actions"]}}
        else:
            new_p = copy.deepcopy(self.presets[self.current_index])
            new_p["name"] = name

        self.presets.append(new_p)
        self.cbPresets.addItem(name)
        self.set_active_preset(len(self.presets) - 1)

    def on_rename_clicked(self):
        old = self.presets[self.current_index]['name']
        dlg = RenamePresetDialog(self, old, [p['name'] for p in self.presets])
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        new_name = dlg.new_name
        self.presets[self.current_index]['name'] = new_name
        self.cbPresets.setItemText(self.current_index, new_name)
        self.init_preset_list()
        self.save_bindings()

    def on_delete_clicked(self):
        name = self.presets[self.current_index]['name']
        if name == "Default":
            QMessageBox.warning(self, "Error", "Cannot delete Default preset")
            return

        reply = QMessageBox.question(
            self,
            "Delete Preset",
            f"Are you sure you want to delete preset '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        del self.presets[self.current_index]
        self.cbPresets.removeItem(self.current_index)
        self.current_index = min(self.current_index, len(self.presets) - 1)
        self.current_preset = self.presets[self.current_index]

        self.init_preset_list()
        self.cbPresets.setCurrentIndex(self.current_index)
        self.on_preset_changed(self.current_index)
        self.save_bindings()
        self.update_actions()

    def on_apply_default_bindings(self):
        reply = QMessageBox.question(
            self,
            "Apply Defaults",
            "Are you sure you want to overwrite the current preset's bindings\n"
            "with the defaults from bindings_default.json?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        default_path = 'model/bindings_default.json'
        try:
            with open(default_path, encoding='utf-8') as f:
                default_data = json.load(f)
            default_actions = default_data['presets'][0]['actions']
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load defaults:\n{e}")
            return

        for action_name, action_info in default_actions.items():
            self.current_preset['actions'][action_name]['variants'] = copy.deepcopy(
                action_info.get('variants', [])
            )

        self.save_bindings()
        self.update_actions()
        QMessageBox.information(self, "Defaults Applied", "Default bindings have been applied.")

    def on_save_action_bind_clicked(self):
        item = self.listActions.currentItem()
        if not item:
            return
        action_name = item.text()

        new_variants = []
        for seq_list, other_list in self.variant_lists:
            seq_ids = []

            for i in range(seq_list.count()):
                itm = seq_list.item(i)
                if isinstance(itm, PlaceholderItem):
                    continue
                text = itm.text()
                name, hand = text.rsplit('(', 1)
                hand = hand.rstrip(')')
                gid = self.name_to_id(hand, name.strip())
                if gid is not None:
                    seq_ids.append(gid)
            if not seq_ids:
                continue

            oth_item = None
            for i in range(other_list.count()):
                itm = other_list.item(i)
                if not isinstance(itm, PlaceholderItem):
                    oth_item = itm
                    break
            if oth_item:
                name2, hand2 = oth_item.text().rsplit('(', 1)
                hand2 = hand2.rstrip(')')
                oid = self.name_to_id(hand2, name2.strip())
                other = {"hand": hand2, "id": oid} if oid is not None else None
            else:
                other = None

            main_hand = seq_list.used_hands().pop() if seq_list.used_hands() else ""

            new_variants.append({
                "sequence_hand": main_hand,
                "sequence": seq_ids,
                "other_hand_gesture": other
            })

        self.current_preset['actions'][action_name]['variants'] = new_variants
        self.save_bindings()
        QMessageBox.information(self, "Saved", f"Bindings for '{action_name}' saved.")

    def on_default_action_bind_clicked(self):
        item = self.listActions.currentItem()
        if not item:
            return
        action_name = item.text()

        default_path = 'model/bindings_default.json'
        try:
            with open(default_path, encoding='utf-8') as f:
                data = json.load(f)
            default_variants = data['presets'][0]['actions'].get(action_name, {}).get('variants', [])
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot load defaults:\n{e}")
            return

        for seq_list, other_list in self.variant_lists:
            seq_list.clear_gestures()
            other_list.clear_gestures()

        for idx, var in enumerate(default_variants):
            if idx >= len(self.variant_lists):
                self.add_variant()
            seq_list, other_list = self.variant_lists[idx]

            seq_list.clear_gestures()
            for gid in var.get('sequence', []):
                name = self.id_to_name(var['sequence_hand'], gid) or str(gid)
                seq_list.insertItem(seq_list.count() - 1, QListWidgetItem(f"{name} ({var['sequence_hand']})"))

            other_list.clear_gestures()
            oh = var.get('other_hand_gesture')
            if oh:
                name2 = self.id_to_name(oh['hand'], oh['id']) or str(oh['id'])
                other_list.insertItem(0, QListWidgetItem(f"{name2} ({oh['hand']})"))

            seq_list.update_placeholders()
            other_list.update_placeholders()

    def on_clean_action_bind_clicked(self):
        for seq_list, other_list in self.variant_lists:
            seq_list.clear_gestures()
            other_list.clear_gestures()
            self.removeAll_variants()

    def id_to_name(self, hand: str, gid: int) -> Optional[str]:
        path = self.gesture_path if hand == 'Right' else self.gesture_path_left
        try:
            with open(path, newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                for idx, row in enumerate(reader):
                    if idx == gid:
                        return row[0]
        except Exception as e:
            print(f"Ошибка чтения {path}: {e}")
        return None

    def name_to_id(self, hand: str, name: str) -> Optional[int]:
        path = self.gesture_path if hand == 'Right' else self.gesture_path_left
        try:
            with open(path, newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                for idx, row in enumerate(reader):
                    if row and row[0] == name:
                        return idx
        except Exception as e:
            print(f"Ошибка чтения {path}: {e}")
        return None

    def save_bindings(self):
        data = {
            "active": self.presets[self.current_index]['name'],
            "presets": self.presets
        }
        with open(self.bindings_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec())

# def toggle_tracking(self):
#     print('toggle')
#     self.tracking = not self.tracking
#
# def update_frame1(self):
#     if self.cap is not None:
#         ret, frame = self.cap.read()
#         if ret:
#             frame = cv2.flip(frame, 1)
#             # frame = cv2.resize(frame, (1280, 720))
#             cv2.rectangle(frame,
#                           (self.control_area_x, self.control_area_y),
#                           (self.control_area_x + self.control_area_width,
#                            self.control_area_y + self.control_area_height),
#                           (0, 255, 0), 2)
#
#             cv2.rectangle(frame,
#                           (self.control_area_x + self.buffer, self.control_area_y + self.buffer),
#                           (self.control_area_x + self.control_area_width - self.buffer,
#                            self.control_area_y + self.control_area_height - self.buffer),
#                           (255, 0, 0), 2)
#
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             result = self.hands.process(rgb_frame)
#
#             if result.multi_hand_landmarks:
#                 for hand_landmarks in result.multi_hand_landmarks:
#                     self.mp_drawing.draw_landmarks(rgb_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
#                     if self.tracking:
#                         x = int(
#                             hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1])
#                         y = int(
#                             hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])
#                         if (self.control_area_x <= x < self.control_area_x + self.control_area_width and
#                                 self.control_area_y <= y < self.control_area_y + self.control_area_height):
#
#                             # norm_x = (x - self.control_area_x) / self.control_area_width
#                             # norm_y = (y - self.control_area_y) / self.control_area_height
#                             norm_x = (x - (self.control_area_x + self.buffer)) / self.effective_area_width
#                             norm_y = (y - (self.control_area_y + self.buffer)) / self.effective_area_height
#                             norm_x = min(max(norm_x, 0.0), 1.0)
#                             norm_y = min(max(norm_y, 0.0), 1.0)
#
#                             # cursor_x = int(x * self.width() / frame.shape[1])
#                             # cursor_y = int(y * self.height() / frame.shape[0])
#
#                             "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
#                             cursor_x = int(norm_x * self.monitor_width)
#                             cursor_y = int(norm_y * self.monitor_height)
#
#                             aspect_ratio_camera = self.camera_width / self.camera_height
#                             aspect_ratio_monitor = self.monitor_width / self.monitor_height
#                             if aspect_ratio_camera != aspect_ratio_monitor:
#                                 scale_x = aspect_ratio_monitor / aspect_ratio_camera
#                                 cursor_x = int(cursor_x * scale_x)
#
#                             if self.prev_cursor_x is not None and self.prev_cursor_y is not None:
#                                 delta_x = abs(cursor_x - self.prev_cursor_x)
#                                 delta_y = abs(cursor_y - self.prev_cursor_y)
#                                 if delta_x < self.movement_threshold and delta_y < self.movement_threshold:
#                                     continue
#                             else:
#                                 self.prev_cursor_x = cursor_x
#                                 self.prev_cursor_y = cursor_y
#
#                             cursor_x = int(
#                                 self.smoothing_factor * cursor_x + (1 - self.smoothing_factor) * self.prev_cursor_x)
#                             cursor_y = int(
#                                 self.smoothing_factor * cursor_y + (1 - self.smoothing_factor) * self.prev_cursor_y)
#                             cursor_pos = QPoint(cursor_x, cursor_y)
#
#                             QCursor.setPos(cursor_pos)
#                             self.prev_cursor_x = cursor_x
#                             self.prev_cursor_y = cursor_y
#
#             h, w, ch = rgb_frame.shape
#             bytes_per_line = ch * w
#             q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
#
#             pixmap = QPixmap.fromImage(q_img)
#             scaled_pixmap = pixmap.scaled(self.camView.size(), Qt.AspectRatioMode.KeepAspectRatio)
#             self.camView.setPixmap(scaled_pixmap)
#
# def update_frame222(self):
#     if self.cap is not None:
#         ret, frame = self.cap.read()
#         if ret:
#             frame = cv2.flip(frame, 1)
#             # frame = cv2.resize(frame, (1280, 720))
#             cv2.rectangle(frame,
#                           (self.control_area_x, self.control_area_y),
#                           (self.control_area_x + self.control_area_width,
#                            self.control_area_y + self.control_area_height),
#                           (0, 255, 0), 2)
#
#             cv2.rectangle(frame,
#                           (self.control_area_x + self.buffer, self.control_area_y + self.buffer),
#                           (self.control_area_x + self.control_area_width - self.buffer,
#                            self.control_area_y + self.control_area_height - self.buffer),
#                           (255, 0, 0), 2)
#
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             result = self.hands.process(rgb_frame)
#
#             if result.multi_hand_landmarks:
#                 for hand_landmarks in result.multi_hand_landmarks:
#                     self.mp_drawing.draw_landmarks(rgb_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
#
#                     index_finger = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
#                     middle_finger = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
#                     x1 = int(index_finger.x * frame.shape[1])
#                     y1 = int(index_finger.y * frame.shape[0])
#                     x2 = int(middle_finger.x * frame.shape[1])
#                     y2 = int(middle_finger.y * frame.shape[0])
#                     distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
#                     print("distance = ", distance)
#                     threshold = 75
#
#                     if distance < threshold:
#                         if not self.tracking:
#                             self.tracking = True
#                             self.prev_cursor_x = QCursor.pos().x()
#                             self.prev_cursor_y = QCursor.pos().y()
#                     else:
#                         self.tracking = False
#
#                     if self.tracking:
#                         x = int(
#                             hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1])
#                         y = int(
#                             hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])
#                         if (self.control_area_x <= x < self.control_area_x + self.control_area_width and
#                                 self.control_area_y <= y < self.control_area_y + self.control_area_height):
#
#                             # norm_x = (x - self.control_area_x) / self.control_area_width
#                             # norm_y = (y - self.control_area_y) / self.control_area_height
#                             norm_x = (x - (self.control_area_x + self.buffer)) / self.effective_area_width
#                             norm_y = (y - (self.control_area_y + self.buffer)) / self.effective_area_height
#                             norm_x = min(max(norm_x, 0.0), 1.0)
#                             norm_y = min(max(norm_y, 0.0), 1.0)
#
#                             # cursor_x = int(x * self.width() / frame.shape[1])
#                             # cursor_y = int(y * self.height() / frame.shape[0])
#
#                             "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
#                             cursor_x = int(norm_x * self.monitor_width)
#                             cursor_y = int(norm_y * self.monitor_height)
#
#                             aspect_ratio_camera = self.camera_width / self.camera_height
#                             aspect_ratio_monitor = self.monitor_width / self.monitor_height
#                             if aspect_ratio_camera != aspect_ratio_monitor:
#                                 scale_x = aspect_ratio_monitor / aspect_ratio_camera
#                                 cursor_x = int(cursor_x * scale_x)
#
#                             if self.prev_cursor_x is not None and self.prev_cursor_y is not None:
#                                 delta_x = abs(cursor_x - self.prev_cursor_x)
#                                 delta_y = abs(cursor_y - self.prev_cursor_y)
#                                 if delta_x < self.movement_threshold and delta_y < self.movement_threshold:
#                                     continue
#                             else:
#                                 self.prev_cursor_x = cursor_x
#                                 self.prev_cursor_y = cursor_y
#
#                             cursor_x = int(
#                                 self.smoothing_factor * cursor_x + (1 - self.smoothing_factor) * self.prev_cursor_x)
#                             cursor_y = int(
#                                 self.smoothing_factor * cursor_y + (1 - self.smoothing_factor) * self.prev_cursor_y)
#                             cursor_pos = QPoint(cursor_x, cursor_y)
#
#                             QCursor.setPos(cursor_pos)
#                             self.prev_cursor_x = cursor_x
#                             self.prev_cursor_y = cursor_y
#
#             h, w, ch = rgb_frame.shape
#             bytes_per_line = ch * w
#             q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
#
#             pixmap = QPixmap.fromImage(q_img)
#             scaled_pixmap = pixmap.scaled(self.camView.size(), Qt.AspectRatioMode.KeepAspectRatio)
#             self.camView.setPixmap(scaled_pixmap)
#
# def update_frame333(self):
#     if self.cap is not None:
#         ret, frame = self.cap.read()
#         if ret:
#             frame = cv2.flip(frame, 1)
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             result = self.hands.process(rgb_frame)
#
#             if result.multi_hand_landmarks:
#                 for hand_landmarks in result.multi_hand_landmarks:
#                     self.mp_drawing.draw_landmarks(rgb_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
#                     index_finger = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
#                     middle_finger = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
#                     thumb_finger = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
#                     ring_finger = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
#                     pinky_finger = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
#
#                     self.mouse_controller.update_tracking(index_finger, middle_finger, frame.shape[1],
#                                                           frame.shape[0])
#                     self.mouse_controller.move_cursor(middle_finger, frame.shape[1], frame.shape[0])
#                     self.mouse_controller.check_for_click(thumb_finger, ring_finger, pinky_finger, frame.shape[1],
#                                                           frame.shape[0])
#
#             h, w, ch = rgb_frame.shape
#             bytes_per_line = ch * w
#             q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
#
#             pixmap = QPixmap.fromImage(q_img)
#             scaled_pixmap = pixmap.scaled(self.camView.size(), Qt.AspectRatioMode.KeepAspectRatio)
#             self.camView.setPixmap(scaled_pixmap)


# def show_cam():
#     mp_hands = mp.solutions.hands
#     hands = mp_hands.Hands()
#     mp_drawing = mp.solutions.drawing_utils
#
#     cap = cv2.VideoCapture(0)
#
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         result = hands.process(rgb_frame)
#         if result.multi_hand_landmarks:
#             for hand_landmarks in result.multi_hand_landmarks:
#                 mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#
#         cv2.imshow('Hand Tracking', frame)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
#
# if __name__ == '__main__':
#
#     # show_cam()
