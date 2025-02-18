import ctypes
import time
from collections import deque

import cv2
import os
import mediapipe as mp
import sys
import pyautogui
from PyQt6 import uic
from pynput.mouse import Button, Controller
from PyQt6.QtCore import QTimer, Qt, QPoint, QThread, QMutex
from PyQt6.QtGui import QImage, QPixmap, QCursor, QGuiApplication
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel
from screeninfo import get_monitors

class MouseController:
    def __init__(self, main_instance, gesture_buffer, monitor_width, monitor_height, control_area, buffer,
                 scaling_factor=1, smoothing_factor=0.5,
                 movement_threshold=10, sensitivity=5, click_delay=0.3, stability_threshold=0.5):
        self.main_instance = main_instance
        self.gesture_buffer = gesture_buffer
        self.buffer = buffer
        self.smoothing_factor = smoothing_factor
        self.movement_threshold = movement_threshold
        self.sensitivity = sensitivity
        self.tracking = False
        self.allow_tracking = False
        self.last_x = None
        self.last_y = None

        self.left_button_pressed = False
        self.right_button_pressed = False

        self.last_click_time = 0
        self.last_stable_time = 0
        self.click_delay = click_delay
        self.stability_threshold = stability_threshold
        self.mouse = Controller()
        # self.monitor_width = monitor_width
        # self.monitor_height = monitor_height
        # self.control_area_x, self.control_area_y, self.control_area_width, self.control_area_height = control_area
        # self.effective_area_width = self.control_area_width - 2 * self.buffer
        # self.effective_area_height = self.control_area_height - 2 * self.buffer
        # self.scaling_factor = scaling_factor
        # self.prev_delta_x = 0
        # self.prev_delta_y = 0
        # self.prev_cursor_x = None
        # self.prev_cursor_y = None
        # self.base_finger_x = None
        # self.base_finger_y = None

    def update_tracking(self, index_finger, middle_finger, frame_width, frame_height):
        self.gesture_buffer = self.main_instance.gesture_buffer
        if self.gesture_buffer[-1] == 3 and self.gesture_buffer[-3] == 3 and not self.allow_tracking:
            self.allow_tracking = True
        elif self.gesture_buffer[-1] == 3 and self.gesture_buffer[-3] == 3 and self.allow_tracking:
            self.allow_tracking = False
            self.tracking = False

        if not self.allow_tracking:
            return

        x1 = int(index_finger.x * frame_width)
        y1 = int(index_finger.y * frame_height)
        x2 = int(middle_finger.x * frame_width)
        y2 = int(middle_finger.y * frame_height)
        distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

        threshold = 60
        if distance < threshold:
            if not self.tracking:
                self.tracking = True
                # self.base_finger_x = x1
                # self.base_finger_y = y1
                # QCursor.pos().x()
                # self.prev_cursor_x = QCursor.pos().x()
                # self.prev_cursor_y = QCursor.pos().y()
                self.last_x = x2
                self.last_y = y2
        else:
            self.tracking = False


    # def move_cursor4444(self, middle_finger, frame_width, frame_height):
    #     if not self.tracking:
    #         return
    #     x = int(middle_finger.x * frame_width)
    #     y = int(middle_finger.y * frame_height)
    #
    #     if self.last_position is not None:
    #         delta_x = abs(x - self.last_position[0])
    #         delta_y = abs(y - self.last_position[1])
    #
    #         if delta_x > self.movement_threshold or delta_y > self.movement_threshold:
    #             current_time = time.time()
    #             if current_time - self.last_stable_time < self.stability_threshold:
    #                 return
    #             else:
    #                 self.last_stable_time = current_time
    #                 self.last_position = (x, y)
    #
    #     cur_x, cur_y = QCursor.pos().x(), QCursor.pos().y()
    #     delta_x = (x - self.last_x) * self.sensitivity
    #     delta_y = (y - self.last_y) * self.sensitivity
    #     new_x = cur_x+delta_x*self.smoothing_factor
    #     new_y = cur_y+delta_y*self.smoothing_factor
    #     if abs(delta_x) > self.movement_threshold or abs(delta_y) > self.movement_threshold:
    #         # self.smooth_move(cur_x, cur_y, new_x, new_y)
    #         QCursor.setPos(int(round(new_x)), int(round(new_y)))
    #
    #
    #         # self.last_x = x
    #         # self.last_y = y

    def move_cursor(self, middle_finger, frame_width, frame_height):
        if not self.tracking:
            return
        x = int(middle_finger.x * frame_width)
        y = int(middle_finger.y * frame_height)
        if self.last_x is None or self.last_y is None:
            return
        delta_x = (x - self.last_x) * self.sensitivity
        delta_y = (y - self.last_y) * self.sensitivity
        cur_x, cur_y = QCursor.pos().x(), QCursor.pos().y()
        new_x = cur_x + delta_x * self.smoothing_factor
        new_y = cur_y + delta_y * self.smoothing_factor
        if abs(delta_x) > self.movement_threshold or abs(delta_y) > self.movement_threshold:
            # self.smooth_move(cur_x, cur_y, new_x, new_y)

            "Управление курсором"
            QCursor.setPos(int(round(new_x)), int(round(new_y)))
        self.last_x = x
        self.last_y = y

    # def check_for_click(self, thumb_finger, index_finger, middle_finger, frame_width, frame_height):
    #     # if not self.tracking:
    #     #     return
    #     x_thumb = int(thumb_finger.x * frame_width)
    #     y_thumb = int(thumb_finger.y * frame_height)
    #     x_index = int(index_finger.x * frame_width)
    #     y_index = int(index_finger.y * frame_height)
    #     x_middle = int(middle_finger.x * frame_width)
    #     y_middle = int(middle_finger.y * frame_height)
    #
    #     distance_thumb_index = ((x_thumb - x_index) ** 2 + (y_thumb - y_index) ** 2) ** 0.5
    #     distance_thumb_middle = ((x_thumb - x_middle) ** 2 + (y_thumb - y_middle) ** 2) ** 0.5
    #
    #     current_time = time.time()
    #     click_threshold = 50
    #     "поотключал пока pyautogui.mouseDown(button='')"
    #     if distance_thumb_index < click_threshold or distance_thumb_middle < click_threshold:
    #         if current_time - self.last_stable_time > self.stability_threshold:
    #             self.last_stable_time = current_time
    #             if not self.left_button_pressed and distance_thumb_index < click_threshold:
    #                 self.left_button_pressed = True
    #                 # pyautogui.mouseDown(button='left')
    #                 print("Left Click Pressed")
    #                 self.last_click_time = current_time
    #
    #             if not self.right_button_pressed and distance_thumb_middle < click_threshold:
    #                 self.right_button_pressed = True
    #                 # pyautogui.mouseDown(button='right')
    #                 print("Right Click Pressed")
    #                 self.last_click_time = current_time
    #     else:
    #         if current_time - self.last_stable_time > self.stability_threshold:
    #             if self.left_button_pressed and distance_thumb_index > click_threshold:
    #                 self.left_button_pressed = False
    #                 # pyautogui.mouseUp(button='left')
    #                 print("Left Click Released")
    #                 self.last_click_time = current_time
    #
    #             if self.right_button_pressed and distance_thumb_middle > click_threshold:
    #                 self.right_button_pressed = False
    #                 # pyautogui.mouseUp(button='right')
    #                 print("Right Click Released")
    #                 self.last_click_time = current_time

    def check_for_click(self, thumb_finger, ring_finger, pinky_finger, frame_width, frame_height):
        if not self.allow_tracking:
            return
        x_thumb = int(thumb_finger.x * frame_width)
        y_thumb = int(thumb_finger.y * frame_height)
        x_ring = int(ring_finger.x * frame_width)
        y_ring = int(ring_finger.y * frame_height)
        x_pinky = int(pinky_finger.x * frame_width)
        y_pinky = int(pinky_finger.y * frame_height)

        distance_thumb_ring = ((x_thumb - x_ring) ** 2 + (y_thumb - y_ring) ** 2) ** 0.5
        distance_thumb_pinky = ((x_thumb - x_pinky) ** 2 + (y_thumb - y_pinky) ** 2) ** 0.5

        current_time = time.time()
        click_threshold = 50
        # print(distance_thumb_ring)
        "поотключал пока pyautogui.mouseDown(button='')"
        if distance_thumb_ring < click_threshold or distance_thumb_pinky < click_threshold:
            if current_time - self.last_stable_time > self.stability_threshold:
                self.last_stable_time = current_time
                if not self.left_button_pressed and distance_thumb_ring < click_threshold:
                    self.left_button_pressed = True
                    self.mouse.press(Button.left)
                    # ctypes.windll.user32.mouse_event(0x0002, 0, 0, 0, 0)
                    self.last_click_time = current_time
                    print("Left Click Pressed")

                if not self.right_button_pressed and distance_thumb_pinky < click_threshold:
                    self.right_button_pressed = True
                    # pyautogui.mouseDown(button='right')
                    self.mouse.press(Button.right)
                    print("Right Click Pressed")
                    self.last_click_time = current_time
        else:
            if current_time - self.last_stable_time > self.stability_threshold:
                if self.left_button_pressed and distance_thumb_ring > click_threshold:
                    self.left_button_pressed = False
                    # pyautogui.mouseUp(button='left')
                    # ctypes.windll.user32.mouse_event(0x0004, 0, 0, 0, 0)
                    self.mouse.release(Button.left)
                    print("Left Click Released")
                    self.last_click_time = current_time

                if self.right_button_pressed and distance_thumb_pinky > click_threshold:
                    self.right_button_pressed = False
                    # pyautogui.mouseUp(button='right')
                    self.mouse.release(Button.right)
                    print("Right Click Released")
                    self.last_click_time = current_time

        # if distance_thumb_ring < click_threshold and not self.left_button_pressed:
        #     if current_time - self.last_click_time >= self.click_delay:
        #         self.left_button_pressed = True
        #         pyautogui.mouseDown(button='left')
        #         print("Left Click Pressed")
        #         self.last_click_time = current_time
        #
        # elif distance_thumb_pinky < click_threshold and not self.right_button_pressed:
        #     if current_time - self.last_click_time >= self.click_delay:
        #         self.right_button_pressed = True
        #         pyautogui.mouseDown(button='right')
        #         print("Right Click Pressed")
        #         self.last_click_time = current_time
        #
        # if distance_thumb_ring > click_threshold and self.left_button_pressed:
        #     if current_time - self.last_click_time >= self.click_delay:
        #         self.left_button_pressed = False
        #         pyautogui.mouseUp(button='left')
        #         print("Left Click Released")
        #         self.last_click_time = current_time
        #
        # if distance_thumb_pinky > click_threshold and self.right_button_pressed:
        #     if current_time - self.last_click_time >= self.click_delay:
        #         self.right_button_pressed = False
        #         pyautogui.mouseUp(button='right')
        #         print("Right Click Released")
        #         self.last_click_time = current_time

    def smooth_move(self, cur_x, cur_y, target_x, target_y, steps=2):
        step_x = (target_x - cur_x) / steps
        step_y = (target_y - cur_y) / steps

        for i in range(steps):
            new_x = int(cur_x + step_x * (i + 1))
            new_y = int(cur_y + step_y * (i + 1))
            QCursor.setPos(new_x, new_y)
            QThread.msleep(2)

    # def move_cursor1(self, index_finger, frame_width, frame_height):
    #     if not self.tracking:
    #         return
    #
    #     if (self.control_area_x <= x < self.control_area_x + self.control_area_width and
    #     x = int(index_finger.x * frame_width)
    #     y = int(index_finger.y * frame_height)
    #
    #             self.control_area_y <= y < self.control_area_y + self.control_area_height):
    #         norm_x = (x - (self.control_area_x + self.buffer)) / self.effective_area_width
    #         norm_y = (y - (self.control_area_y + self.buffer)) / self.effective_area_height
    #         norm_x = min(max(norm_x, 0.0), 1.0)
    #         norm_y = min(max(norm_y, 0.0), 1.0)
    #
    #         cursor_x = int(norm_x * self.monitor_width)
    #         cursor_y = int(norm_y * self.monitor_height)
    #
    #         if self.prev_cursor_x is not None and self.prev_cursor_y is not None:
    #             delta_x = abs(cursor_x - self.prev_cursor_x)
    #             delta_y = abs(cursor_y - self.prev_cursor_y)
    #             if delta_x < self.movement_threshold and delta_y < self.movement_threshold:
    #                 return
    #
    #         cursor_x = int(
    #             self.smoothing_factor * cursor_x + (1 - self.smoothing_factor) * self.prev_cursor_x)
    #         cursor_y = int(
    #             self.smoothing_factor * cursor_y + (1 - self.smoothing_factor) * self.prev_cursor_y)
    #         QCursor.setPos(QPoint(cursor_x, cursor_y))
    #         self.prev_cursor_x = cursor_x
    #         self.prev_cursor_y = cursor_y
    #
    # def move_cursor2(self, index_finger, frame_width, frame_height):
    #     if not self.tracking:
    #         return
    #
    #     current_finger_x = int(index_finger.x * frame_width)
    #     current_finger_y = int(index_finger.y * frame_height)
    #
    #     print("cur finger = ", index_finger.x, ", ", index_finger.y)
    #     print("base finger = ", self.base_finger_x, ", ", self.base_finger_y)
    #
    #     delta_x = (current_finger_x - self.prev_cursor_x)  # * self.scaling_factor
    #     delta_y = (current_finger_y - self.prev_cursor_y)  # * self.scaling_factor
    #
    #     # print("threshold x = ", self.prev_delta_x - delta_x)
    #     # print("threshold x = ", self.prev_delta_y - delta_y)
    #
    #     # if abs(self.prev_delta_x - delta_x) < self.movement_threshold and abs(
    #     #         self.prev_delta_x - delta_x) < self.movement_threshold:
    #     #     self.prev_delta_x = delta_x
    #     #     self.prev_delta_y = delta_y
    #     #     return
    #     #
    #     # self.prev_delta_x = delta_x
    #     # self.prev_delta_y = delta_y
    #
    #     raw_target_x = int(self.prev_cursor_x + delta_x)
    #     raw_target_y = int(self.prev_cursor_y + delta_y)
    #
    #     self.prev_cursor_x = raw_target_x
    #     self.prev_cursor_y = raw_target_y
    #
    #     # target_x = int(self.smoothing_factor * raw_target_x + (1 - self.smoothing_factor) * self.prev_cursor_x)
    #     # target_y = int(self.smoothing_factor * raw_target_y + (1 - self.smoothing_factor) * self.prev_cursor_y)
    #     target_x = raw_target_x
    #     target_y = raw_target_y
    #     QCursor.setPos(QPoint(target_x, target_y))
    #
    # def move_cursor3(self, index_finger, frame_width, frame_height):
    #     if not self.tracking or self.base_finger_x is None or self.base_finger_y is None:
    #         return
    #
    #     current_finger_x = int(index_finger.x * frame_width)
    #     current_finger_y = int(index_finger.y * frame_height)
    #
    #     print("cur finger = ", index_finger.x, ", ", index_finger.y)
    #     print("base finger = ", self.base_finger_x, ", ", self.base_finger_y)
    #
    #     delta_x = (current_finger_x - self.base_finger_x)  # * self.scaling_factor
    #     delta_y = (current_finger_y - self.base_finger_y)  # * self.scaling_factor
    #
    #     # print("threshold x = ", self.prev_delta_x - delta_x)
    #     # print("threshold x = ", self.prev_delta_y - delta_y)
    #
    #     if abs(self.prev_delta_x - delta_x) < self.movement_threshold and abs(
    #             self.prev_delta_x - delta_x) < self.movement_threshold:
    #         self.prev_delta_x = delta_x
    #         self.prev_delta_y = delta_y
    #         return
    #     self.prev_delta_x = delta_x
    #     self.prev_delta_y = delta_y
    #
    #     raw_target_x = int(self.prev_cursor_x + delta_x)
    #     raw_target_y = int(self.prev_cursor_y + delta_y)
    #
    #     # target_x = int(self.smoothing_factor * raw_target_x + (1 - self.smoothing_factor) * self.prev_cursor_x)
    #     # target_y = int(self.smoothing_factor * raw_target_y + (1 - self.smoothing_factor) * self.prev_cursor_y)
    #     target_x = raw_target_x
    #     target_y = raw_target_y
    #     # QCursor.setPos(QPoint(target_x, target_y))
