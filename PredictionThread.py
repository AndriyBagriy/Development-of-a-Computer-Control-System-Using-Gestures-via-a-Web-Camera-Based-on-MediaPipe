from PyQt6.QtCore import QThread, pyqtSignal


class PredictionThread(QThread):
    prediction_done = pyqtSignal(str, int)

    def __init__(self, processor, classifier, gesture_collector, hand_label):
        super().__init__()
        self.processor = processor
        self.classifier = classifier
        self.gesture_collector = gesture_collector
        self.hand_label = hand_label

        self.last_keypoint = None
        self.last_gesture_id = None

    def run(self):
        landmarks_dict, _ = self.processor.get_hand_landmarks()
        if not landmarks_dict.get(self.hand_label):
            self.prediction_done.emit(self.hand_label, -1)
            return

        keypoint = self.gesture_collector.get_current_keypoint(self.processor, self.hand_label)
        if keypoint is not None:
            gesture_id = self.classifier.predict(keypoint)
            self.prediction_done.emit(self.hand_label, gesture_id)
        else:
            self.prediction_done.emit(self.hand_label, -1)



        # if not self.processor.hand_detected:
        #     self.prediction_done.emit(-1)
        #     return
        # keypoint = self.gesture_collector.get_current_keypoint(self.processor)
        # gesture_id = 0
        # if keypoint is not None:
        #     gesture_id = self.classifier.predict(keypoint)
        # self.prediction_done.emit(gesture_id)

