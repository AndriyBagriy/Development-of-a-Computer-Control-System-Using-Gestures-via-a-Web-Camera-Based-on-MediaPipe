from PyQt6.QtWidgets import QListWidget, QListWidgetItem, QAbstractItemView
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon


class PlaceholderItem(QListWidgetItem):
    def __init__(self, parent: QListWidget):
        super().__init__("+", parent)
        self.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        # можно задать иконку вместо текста:
        # self.setIcon(QIcon("icons/plus.png"))
        # self.setSizeHint(QSize(40, 40))  # если нужно фиксированная высота


class GestureListWidget(QListWidget):
    def __init__(self, parent=None, max_items: int = 3, peer: 'GestureListWidget' = None):
        super().__init__(parent)
        self.max_items = max_items
        self.setAcceptDrops(True)
        self.peer = peer
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DropOnly)

        self.setDragEnabled(False)
        self.setSpacing(4)
        self.setAlternatingRowColors(True)

        self.itemDoubleClicked.connect(self.on_item_double_clicked)
        self._init_placeholder()

    def _init_placeholder(self):
        if not any(isinstance(self.item(i), PlaceholderItem) for i in range(self.count())):
            self.addItem(PlaceholderItem(self))

    def clear_gestures(self):
        self.clear()
        self._init_placeholder()

    def used_hands(self) -> set[str]:
        hands = set()
        for i in range(self.count()):
            itm = self.item(i)
            if isinstance(itm, PlaceholderItem):
                continue
            txt = itm.text()
            if '(' in txt and txt.endswith(')'):
                hands.add(txt.split('(')[-1][:-1])
        return hands

    def verify_sequence(self):
        texts = [self.item(i).text()
                 for i in range(self.count())
                 if not isinstance(self.item(i), PlaceholderItem)]
        unique = []
        for t in texts:
            if not unique or unique[-1] != t:
                unique.append(t)
        self.clear()
        for t in unique:
            self.addItem(QListWidgetItem(t))
        self._init_placeholder()

    def on_item_double_clicked(self, itm: QListWidgetItem):
        if isinstance(itm, PlaceholderItem):
            return
        idx = self.row(itm)
        self.takeItem(idx)

        if self.max_items > 1:
            self.verify_sequence()

        real_count = sum(
            1 for i in range(self.count())
            if not isinstance(self.item(i), PlaceholderItem)
        )
        has_ph = any(isinstance(self.item(i), PlaceholderItem) for i in range(self.count()))
        if real_count < self.max_items and not has_ph:
            self._init_placeholder()

    def dragEnterEvent(self, e):
        src = e.source()
        from PyQt6.QtWidgets import QListWidget as _QLW
        if not isinstance(src, _QLW) or src is self:
            return e.ignore()

        item = src.currentItem()
        if item is None:
            return e.ignore()

        txt = item.text()
        if '(' in txt and txt.endswith(')'):
            hand = txt.split('(')[-1][:-1]
        else:
            return e.ignore()

        my_hands = self.used_hands()
        peer_hands = self.peer.used_hands() if self.peer else set()

        real_count = len([1 for i in range(self.count())
                          if not isinstance(self.item(i), PlaceholderItem)])
        if real_count >= self.max_items:
            return e.ignore()

        if self.max_items > 1:
            if my_hands and hand not in my_hands:
                return e.ignore()
            if peer_hands and hand in peer_hands:
                return e.ignore()
            last_txt = None
            for i in range(self.count() - 1, -1, -1):
                itm = self.item(i)
                if not isinstance(itm, PlaceholderItem):
                    last_txt = itm.text()
                    break
            if last_txt is not None and last_txt == txt:
                return e.ignore()
        else:
            if peer_hands and hand in peer_hands:
                return e.ignore()

        e.acceptProposedAction()

    def dropEvent(self, e):
        src = e.source()
        if not isinstance(src, QListWidget) or src is self:
            return e.ignore()

        item = src.currentItem()
        if item is None:
            return e.ignore()

        txt = item.text()
        hand = txt.split('(')[-1][:-1]

        if self.max_items == 1:
            super().clear()

        ph_idx = None
        for i in range(self.count()):
            if isinstance(self.item(i), PlaceholderItem):
                ph_idx = i
                break

        new_item = QListWidgetItem(txt)
        self.insertItem(ph_idx if ph_idx is not None else self.count(), new_item)
        self.update_placeholders()
        # real_count = len([1 for i in range(self.count())
        #                   if not isinstance(self.item(i), PlaceholderItem)])
        # if real_count < self.max_items:
        #     self._init_placeholder()
        # else:
        #     for i in reversed(range(self.count())):
        #         if isinstance(self.item(i), PlaceholderItem):
        #             self.takeItem(i)

        e.acceptProposedAction()

    def update_placeholders(self):
        real_count = sum(
            1 for i in range(self.count())
            if not isinstance(self.item(i), PlaceholderItem)
        )
        if real_count < self.max_items:
            for i in reversed(range(self.count())):
                if isinstance(self.item(i), PlaceholderItem):
                    self.takeItem(i)
            self._init_placeholder()
        else:
            for i in reversed(range(self.count())):
                if isinstance(self.item(i), PlaceholderItem):
                    self.takeItem(i)

