import csv

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, QMutex
from PyQt6.QtGui import QImage


class FrameProcessor(QThread):
    frame_ready = pyqtSignal(QImage)
    right_hand_data_ready = pyqtSignal(object, object, object, object, object, int, int, str, bool, )
    left_hand_data_ready = pyqtSignal(object, object, object, object, object, int, int, str, bool, )

    def __init__(self, cap, mp_hands, mp_drawing, gesture_buffer,
                 gesture_buffer_left, gesture_right, gesture_left, parent=None):
        super().__init__(parent)
        self.cap = cap
        self.running = True
        self.mp_hands = mp_hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,

        )
        self.mp_drawing = mp_drawing

        self.gesture_buffer = gesture_buffer
        self.gesture_buffer_left = gesture_buffer_left

        self.gesture_right = gesture_right
        self.gesture_left = gesture_left

        self.last_hand_landmarks = {'Right': None, 'Left': None}
        self.last_handedness = {'Right': None, 'Left': None}

        self.hand_present = {"Right": False, "Left": False}
        # self.hand_detected = False
        self.overlay_text = ""

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            resized_frame = cv2.resize(frame, (1920, 1080))
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            result = self.hands.process(rgb_frame)

            detected = {'Right': False, 'Left': False}

            if result.multi_hand_landmarks:
                counts = {'Right': 0, 'Left': 0}
                for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                    label = handedness.classification[0].label  # 'Right' или 'Left'
                    if counts[label] >= 1:
                        continue
                    counts[label] += 1

                    self.mp_drawing.draw_landmarks(rgb_frame,
                                                   hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    pts = {
                        'index': hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP],
                        'middle': hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                        'thumb': hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP],
                        'ring': hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP],
                        'pinky': hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP],
                        'wrist': hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                    }

                    w, h = frame.shape[1], frame.shape[0]
                    self.last_hand_landmarks[label] = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                    self.last_handedness[label] = label
                    detected[label] = True

                    signal = self.right_hand_data_ready if label == 'Right' else self.left_hand_data_ready
                    signal.emit(
                        pts['index'], pts['middle'], pts['thumb'], pts['ring'], pts['pinky'],
                        w, h, label, True
                    )

                    if label == 'Right':
                        buffer = self.gesture_buffer
                        gesture_file = self.gesture_right

                    elif label == 'Left':
                        buffer = self.gesture_buffer_left
                        gesture_file = self.gesture_left
                    else:
                        continue

                    if buffer:
                        gesture_id = buffer[-1]
                        gesture_text = 'None'
                        try:
                            with open(gesture_file, newline='', encoding='utf-8') as f:
                                reader = csv.reader(f)
                                next(reader)
                                for i, row in enumerate(reader, 1):
                                    if i == gesture_id:
                                        gesture_text = row[0]
                                        break
                        except Exception:
                            pass

                        wx = int(pts['wrist'].x * rgb_frame.shape[1])
                        wy = int(pts['wrist'].y * rgb_frame.shape[0])
                        cv2.putText(
                            rgb_frame,
                            gesture_text,
                            (wx - 100, wy + 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (0, 255, 0), 3, cv2.LINE_AA
                        )
                        # if self.gesture_buffer:
                        #     gesture_id = self.gesture_buffer[-1]
                        #     gesture_text = 'None'
                        #     try:
                        #         with open('model/gestures.csv', newline='', encoding='utf-8') as f:
                        #             reader = csv.reader(f)
                        #             next(reader)
                        #             for i, row in enumerate(reader, 1):
                        #                 if i == gesture_id:
                        #                     gesture_text = row[0]
                        #                     break
                        #     except Exception:
                        #         pass
                        #
                        #     wx = int(pts['wrist'].x * rgb_frame.shape[1])
                        #     wy = int(pts['wrist'].y * rgb_frame.shape[0])
                        #     cv2.putText(
                        #         rgb_frame, gesture_text,
                        #         (wx - 100, wy + 50),
                        #         cv2.FONT_HERSHEY_SIMPLEX,
                        #         1.5, (0, 255, 0), 3, cv2.LINE_AA
                        #     )

                        if self.overlay_text:
                            cv2.putText(
                                rgb_frame, self.overlay_text,
                                (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.2, (255, 255, 255), 2, cv2.LINE_AA
                            )
            self.hand_present = detected
            if not detected['Right']:
                self.last_hand_landmarks["Right"] = None
                self.last_handedness["Right"] = None
                self.right_hand_data_ready.emit(None, None, None, None, None, 0, 0, None, False)
            if not detected['Left']:
                self.last_hand_landmarks["Left"] = None
                self.last_handedness["Left"] = None
                self.left_hand_data_ready.emit(None, None, None, None, None, 0, 0, None, False)

            h, w, ch = rgb_frame.shape
            qimg = QImage(rgb_frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
            self.frame_ready.emit(qimg)

    def get_hand_landmarks(self):
        return self.last_hand_landmarks, self.last_handedness

    def get_handedness(self):
        return self.last_handedness

    def stop(self):
        self.running = False
        self.wait()


"""
class FrameProcessor(QThread):
    frame_ready = pyqtSignal(QImage)
    hand_data_ready = pyqtSignal(object, object, object, object, object, int, int, str, bool, )

    def __init__(self, cap, mp_hands, mp_drawing, gesture_buffer, parent=None):
        super().__init__(parent)
        self.cap = cap
        self.running = True
        self.mp_hands = mp_hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,

        )
        self.mp_drawing = mp_drawing
        self.gesture_buffer = gesture_buffer
        self.hand_detected = False
        self.last_hand_landmarks = None
        self.last_handedness = None

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            resized_frame = cv2.resize(frame, (1920, 1080))
            # resized_frame = cv2.resize(frame, (640, 360))
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            result = self.hands.process(rgb_frame)
            wrist_x = 0
            wrist_y = 0

            hand_detected_in_current_frame = False

            if result.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                    label = handedness.classification[0].label

                    if label == 'Right':
                        self.mp_drawing.draw_landmarks(rgb_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                        wrist_x, wrist_y = int(wrist.x * rgb_frame.shape[1]), int(wrist.y * rgb_frame.shape[0])

                        index_finger = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        middle_finger = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                        thumb_finger = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                        ring_finger = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
                        pinky_finger = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]

                        self.hand_detected = True
                        self.last_hand_landmarks = [
                            (landmark.x, landmark.y) for landmark in hand_landmarks.landmark
                        ]
                        self.last_handedness = label

                        self.hand_data_ready.emit(index_finger, middle_finger, thumb_finger, ring_finger, pinky_finger,
                                                  frame.shape[1], frame.shape[0], label, True)
                        hand_detected_in_current_frame = True

                        gesture_id = self.gesture_buffer[-1]
                        gesture_text = ""

                        with open("model/gestures.csv", mode='r', newline='', encoding='utf-8') as file:
                            reader = csv.reader(file)
                            next(reader)
                            if gesture_id == 0 or gesture_id == -1:
                                gesture_text = "None"
                            for i, row in enumerate(reader, 1):
                                if i == gesture_id:
                                    gesture_text = row[0]

                        cv2.putText(
                            rgb_frame,
                            gesture_text,
                            (wrist_x - 100, wrist_y + 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5,
                            (0, 255, 0),
                            3,
                            cv2.LINE_AA
                        )
                        break
                    else:
                        self.last_handedness = None
            if not hand_detected_in_current_frame:
                self.hand_detected = False
                self.last_hand_landmarks = None
                self.last_handedness = None
                self.hand_data_ready.emit(None, None, None, None, None, 0, 0, None, False)

            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.frame_ready.emit(q_img)

    def get_hand_landmarks(self):
        return self.last_hand_landmarks, self.last_handedness

    def get_handedness(self):
        return self.last_handedness

    def stop(self):
        self.running = False
        self.wait()
"""

"""
        def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            resized_frame = cv2.resize(frame, (640, 360))
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

            # Создаем пустой белый фон
            schematic_frame = 255 * np.ones((360, 640, 3), dtype=np.uint8)

            result = self.hands.process(rgb_frame)

            if result.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                    # Собираем точки ладони
                    palm_points = [
                        (int(hand_landmarks.landmark[idx].x * 640), int(hand_landmarks.landmark[idx].y * 360))
                        for idx in [
                            self.mp_hands.HandLandmark.WRIST,
                            self.mp_hands.HandLandmark.THUMB_CMC,
                            self.mp_hands.HandLandmark.THUMB_MCP,
                            self.mp_hands.HandLandmark.INDEX_FINGER_MCP,
                            self.mp_hands.HandLandmark.PINKY_MCP,
                            self.mp_hands.HandLandmark.WRIST
                        ]
                    ]

                    # Заполняем ладонь
                    cv2.fillPoly(schematic_frame, [np.array(palm_points, dtype=np.int32)], color=(200, 200, 200))

                    # Рисуем соединения пальцев
                    for connection in self.mp_hands.HAND_CONNECTIONS:
                        start_idx, end_idx = connection
                        start_point = hand_landmarks.landmark[start_idx]
                        end_point = hand_landmarks.landmark[end_idx]

                        start_coords = (int(start_point.x * 640), int(start_point.y * 360))
                        end_coords = (int(end_point.x * 640), int(end_point.y * 360))

                        cv2.line(schematic_frame, start_coords, end_coords, (0, 0, 0), 2)

                    # Рисуем точки суставов
                    for landmark in hand_landmarks.landmark:
                        x, y = int(landmark.x * 640), int(landmark.y * 360)
                        cv2.circle(schematic_frame, (x, y), 5, (0, 0, 255), -1)

            h, w, ch = schematic_frame.shape
            bytes_per_line = ch * w
            q_img = QImage(schematic_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.frame_ready.emit(q_img)
    """

"""
    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            resized_frame = cv2.resize(frame, (640, 360))
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            result = self.hands.process(rgb_frame)
            if result.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                    label = handedness.classification[0].label
                    
                    self.mp_drawing.draw_landmarks(rgb_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    index_finger = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    middle_finger = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                    thumb_finger = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                    ring_finger = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
                    pinky_finger = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]

                    if label == 'Right':
                        self.hand_detected = True
                        self.hand_data_ready.emit(index_finger, middle_finger, thumb_finger, ring_finger, pinky_finger,
                                                  frame.shape[1], frame.shape[0], label, self.hand_detected)
                        self.last_hand_landmarks = [
                            (landmark.x, landmark.y)
                            for landmark in hand_landmarks.landmark
                        ]
                        self.last_handedness = label

                    elif label == 'Left':
                        self.hand_data_ready.emit(index_finger, middle_finger, thumb_finger, ring_finger, pinky_finger,
                                                  frame.shape[1], frame.shape[0], label, self.hand_detected)
            else:
                self.hand_detected = False
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.frame_ready.emit(q_img)
            """
