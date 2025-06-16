import csv
import os.path
import time
from threading import Thread


class GestureDataCollector:
    def __init__(self, data_file="buffer/gesture_buffer.csv", gestures_names="model/gestures.csv"):
        self.data_file = data_file
        self.gestures_names = gestures_names
        self.collecting = False
        self.collected_data = []
        self.stop_flag = False

        if not os.path.exists(self.data_file):
            with open(self.data_file, mode="w") as file:
                pass

        if not os.path.exists(self.gestures_names):
            with open(self.gestures_names, mode="w") as file:
                pass

    def get_gesture_id(self, gesture_name):
        with open(self.gestures_names, mode="r") as file:
            gestures = file.read().splitlines()
            if gesture_name in gestures:
                return gestures.index(gesture_name)
                # gestures.append(f"{gesture_name}")
                # file.seek(0)
                # file.write("\n".join(gestures) + "\n")
                # file.truncate()
        return -1
        # return gestures.index(gesture_name)

    def normalization(self, landmarks):
        base_x, base_y = landmarks[0]

        raw_landmarks = [
            (x - base_x, y - base_y) for x, y in landmarks
        ]

        max_value = max(
            max(abs(raw_x) for raw_x, _ in raw_landmarks),
            max(abs(raw_y) for _, raw_y in raw_landmarks)
        )

        if max_value == 0:
            return [(0.0, 0.0) for _ in raw_landmarks]

        normalized_landmarks = [
            (raw_x / max_value, raw_y / max_value)
            for raw_x, raw_y in raw_landmarks
        ]
        return normalized_landmarks

#TODO (+) сделать мульти код для двух рук
    def start_collecting(self, processor, gesture_name, data_count, on_finish, hand="Right"):
        if self.collecting:
            return
        self.collecting = True
        self.collected_data = []

        def collect():
            while len(self.collected_data) < data_count:
                landmarks_dict, handedness_dict = processor.get_hand_landmarks()
                if handedness_dict.get(hand) == hand:
                    lm = landmarks_dict.get(hand)
                    if lm:
                        self.collected_data.append((-1, lm))
                time.sleep(0.05)

            normalized_data = []
            for gesture_id, landmarks in self.collected_data:
                norm = self.normalization(landmarks)
                row = [gesture_id] + [c for p in norm for c in p]
                normalized_data.append(row)
            with open(self.data_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(normalized_data)

            self.collecting = False
            print(f"Сбор данных завершён ({hand})")
            if on_finish:
                on_finish()

        Thread(target=collect, daemon=True).start()
        """
        def collect():
            while len(self.collected_data) < data_count:
                landmarks_dict, handedness_dict = processor.get_hand_landmarks()
                if (
                        landmarks_dict.get(hand)
                        and handedness_dict.get(hand) == hand
                ):
                    self.collected_data.append(("-1", landmarks_dict[hand]))
                time.sleep(0.050)

            self.process_and_save_data()
            self.collecting = False
            print(f"Сбор данных завершён ({hand})")
            if on_finish:
                on_finish()

        Thread(target=collect, daemon=True).start()
        """
        # if self.collecting:
        #     return
        # self.collecting = True
        # self.collected_data = []
        #
        # # gesture_id = self.get_gesture_id(gesture_name)
        #
        # def collect():
        #     while len(self.collected_data) < data_count:
        #         landmarks, handedness = processor.get_hand_landmarks()
        #         if landmarks and handedness == "Right":
        #             self.collected_data.append(("-1", landmarks))
        #         time.sleep(0.050)  # 0.033
        #     self.process_and_save_data()
        #     self.collecting = False
        #     print("Сбор данных завершён.")
        #     if on_finish:
        #         on_finish()
        #
        # Thread(target=collect, daemon=True).start()

    def get_current_keypoint(self, processor, hand='Right'):
        all_landmarks, all_handedness = processor.get_hand_landmarks()
        keypoint = None
        if all_landmarks.get(hand) and all_handedness.get(hand) == hand:
            keypoint = self.normalization(all_landmarks[hand])
        return keypoint

        # landmarks, handedness = processor.get_hand_landmarks()
        # keypoint = None
        # if landmarks and handedness == "Right":
        #     keypoint = self.normalization(landmarks)
        # return keypoint

    def process_and_save_data(self):
        normalized_data = []
        for gesture_id, landmarks in self.collected_data:
            normalized_landmarks = self.normalization(landmarks)
            row = [gesture_id] + [coord for point in normalized_landmarks for coord in point]
            normalized_data.append(row)

        with open(self.data_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(normalized_data)

    # def save_data(self, landmarks, gesture_name):
    #     gesture_id = self.get_gesture_id(gesture_name)
    #     normalized_landmarks = self.normalization(landmarks)
    #     row = [gesture_id] + [coord for point in normalized_landmarks for coord in point]
    #     with open(self.data_file, mode="a", newline="") as file:
    #         writer = csv.writer(file)
    #         writer.writerow(row)
