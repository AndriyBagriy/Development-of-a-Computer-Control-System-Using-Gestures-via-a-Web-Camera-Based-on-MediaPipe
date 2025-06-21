from PyQt6.QtCore import QThread, pyqtSignal

import FrameProcessor
import GestureDataCollector
import tensorflow as tf
import numpy as np


class PredictionThread(QThread):
    prediction_done = pyqtSignal(str, int)

    def __init__(self, hand_label: str, processor: FrameProcessor,
                 tflite_model_path: str, gesture_collector: GestureDataCollector):
        super().__init__()
        self.hand_label = hand_label
        self.processor = processor
        self.gesture_collector = gesture_collector
        self.interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def run(self):
        if not self.processor.hand_present.get(self.hand_label, False):
            self.prediction_done.emit(self.hand_label, -1)
            return

        keypoint = self.gesture_collector.get_current_keypoint(self.processor, self.hand_label)
        if keypoint is None:
            self.prediction_done.emit(self.hand_label, -1)
            return

        flat = np.array(keypoint).flatten().astype('float32')
        try:
            self.interpreter.set_tensor(self.input_details[0]['index'], [flat])
            self.interpreter.invoke()
            out = self.interpreter.get_tensor(self.output_details[0]['index'])
            gesture_id = int(np.argmax(out[0]))
        except Exception as e:
            print(f"[{self.hand_label}] tflite inference failed:", e)
            gesture_id = -1

        self.prediction_done.emit(self.hand_label, gesture_id)

    # def __init__(self, processor, classifier, gesture_collector, hand_label):
    #     super().__init__()
    #     self.processor = processor
    #     self.classifier = classifier
    #     self.gesture_collector = gesture_collector
    #     self.hand_label = hand_label
    #     self.last_keypoint = None
    #     self.last_gesture_id = None
    #
    # def run(self):
    #     landmarks_dict, _ = self.processor.get_hand_landmarks()
    #     if not landmarks_dict.get(self.hand_label):
    #         self.prediction_done.emit(self.hand_label, -1)
    #         return
    #
    #     keypoint = self.gesture_collector.get_current_keypoint(self.processor, self.hand_label)
    #     if keypoint is not None:
    #         gesture_id = self.classifier.predict(keypoint)
    #         self.prediction_done.emit(self.hand_label, gesture_id)
    #     else:
    #         self.prediction_done.emit(self.hand_label, -1)
