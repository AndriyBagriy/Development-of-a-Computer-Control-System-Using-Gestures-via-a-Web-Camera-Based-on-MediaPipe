import win32gui
import win32con


class GestureActions:
    @staticmethod
    def action_left_click(gesture_buffer):
        print("Действие: ЛКМ (левая кнопка мыши).")

    @staticmethod
    def action_right_click(gesture_buffer):
        print("Действие: ПКМ (правая кнопка мыши).")

    @staticmethod
    def action_minimize(gesture_buffer):
        print("sdfasdfasdf")
        if gesture_buffer[-2] == 1:
            hwnd = win32gui.GetForegroundWindow()
            # win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)


    @staticmethod
    def action_custom(gesture_buffer):
        print("Пользовательское действие.")

    @staticmethod
    def action_nothing(gesture_buffer):
        print("Нет действия для этого жеста.")


class GestureBinder:
    def __init__(self):
        self.bindings = {}
        self.bind_default_actions()

    def bind_default_actions(self):
        self.bindings = {
            2: GestureActions.action_minimize
        }

    def bind_gesture(self, gesture_id, action):
        self.bindings[gesture_id] = action

    def unbind_gesture(self, gesture_id):
        if gesture_id in self.bindings:
            del self.bindings[gesture_id]

    def execute(self, gesture_buffer):
        gesture_id = gesture_buffer[-1] if gesture_buffer else None
        action = self.bindings.get(gesture_id, GestureActions.action_nothing)
        action(gesture_buffer)
