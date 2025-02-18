from PyQt6.QtCore import QThread, pyqtSignal


class PredictionThread(QThread):
    prediction_done = pyqtSignal(int)

    def __init__(self, processor, classifier, gesture_collector):
        super().__init__()
        self.processor = processor
        self.classifier = classifier
        self.gesture_collector = gesture_collector
        self.last_keypoint = None
        self.last_gesture_id = None

    def run(self):

        if not self.processor.hand_detected:
            self.prediction_done.emit(-1)
            return
        keypoint = self.gesture_collector.get_current_keypoint(self.processor)
        gesture_id = 0
        if keypoint is not None:
            gesture_id = self.classifier.predict(keypoint)
        self.prediction_done.emit(gesture_id)

