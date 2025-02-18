import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os


class UiFileHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith("gui.ui") or event.src_path.endswith("add_gesture.ui"):
            print("UI file changed, regenerating...")
            os.system("pyuic6 -x gui.ui -o gui.py")
            os.system("pyuic6 -x add_gesture.ui -o add_gesture.py")
            print("gui.py and add_gesture.ui updated!")


if __name__ == "__main__":
    path = "."
    event_handler = UiFileHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
