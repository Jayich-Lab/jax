from PyQt5 import QtWidgets, QtGui, QtCore
from jax.util.ui.dialog_on_top import DialogOnTop


class CustomListWidget(QtWidgets.QListWidget):
    """QListWidget that can reorder items and hide/show items.

    Do not call self.addItem() or self.addItemWidget() directly.
    Use self.add_item_and_widget() instead.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.visible_items = []
        self.hidden_items = []
        self.all_items = {}

        self.setAcceptDrops(False)
        self.setDragEnabled(False)
        self.setFlow(QtWidgets.QListView.LeftToRight)
        self.setResizeMode(QtWidgets.QListView.Adjust)
        self.setViewMode(QtWidgets.QListView.IconMode)

    def add_item_and_widget(self, name, widget, visible=True, sort_index=None, padding=10):
        """Adds an widget to the ListWidget.

        Combines self.addItem() and self.setItemWidget().

        Args:
            name: str, name of the item.
            widget: QWodget, widget to add to the ListWidget.
            visible: bool, whether the widget should be visible. Default is True.
            sort_index: int, order of the item in the ListWidget. If None, the displayed order
                is the same as the order added. Default is None.
            padding: int, padding between the item and the widget. Default is 10.
        """
        if name in self.all_items:
            raise ValueError(f"{name} already exists in the list widget.")
        size = widget.sizeHint()
        new_size = QtCore.QSize(size.width() + padding, size.height() + padding)
        if self.gridSize().height() > new_size.height():
            new_size.setHeight(self.gridSize().height())
        if self.gridSize().width() > new_size.width():
            new_size.setWidth(self.gridSize().width())
        self.setGridSize(new_size)

        if sort_index is None:
            sort_index = len(self.all_items)
        item = _CustomListWidgetItem(sort_index)
        item.setSizeHint(widget.sizeHint())
        if visible:
            self.visible_items.append(name)
        else:
            self.hidden_items.append(name)
            item.setHidden(True)
        self.addItem(item)
        self.setItemWidget(item, widget)
        self.all_items[name] = item
        self.sortItems()
        self.setDragEnabled(False)

    def update_ui(self):
        for name in self.all_items:
            item = self.all_items[name]
            if name in self.visible_items:
                item.setHidden(False)
                item.sort_index = self.visible_items.index(name)
            else:
                item.setHidden(True)
                item.sort_index = len(self.visible_items) + self.hidden_items.index(name)
        self.sortItems()
        self.setDragEnabled(False)

    def contextMenuEvent(self, event):
        """Override contextMenuEvent to add a custom context menu.

        To add more actions to the context menu, you must copy the code from this method
        and add your own actions.
        """
        menu = QtWidgets.QMenu()
        manage_widgets_action = menu.addAction("Manage Widgets")
        action = menu.exec_(self.mapToGlobal(event.pos()))
        if action == manage_widgets_action:
            self._manage_widgets()

    def _manage_widgets(self):
        dialog = _CustomListWidgetManager(self, self.visible_items, self.hidden_items)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.visible_items = dialog.visible_items
            self.hidden_items = dialog.hidden_items
            self.update_ui()

    def addItem(self, item):
        """Do not call this directly. Use self.add_item_and_widget() instead."""
        if not isinstance(item, _CustomListWidgetItem):
            raise TypeError("item must be a CustomListWidgetItem.")
        super().addItem(item)

    def addItemWidget(self, item, widget):
        """Do not call this directly. Use self.add_item_and_widget() instead."""
        if not isinstance(item, _CustomListWidgetItem):
            raise TypeError("item must be a CustomListWidgetItem.")
        super().addItemWidget(item, widget)


class _CustomListWidgetManager(DialogOnTop):
    """Select widget visibility and orders."""
    def __init__(self, parent, visible_items, hidden_items):
        super().__init__(parent)
        self.setWindowTitle("Select widget visibility and orders")
        self.visible_items = visible_items
        self.hidden_items = hidden_items
        self.init_ui()

    def build_tree_widget_item(self, name, is_header):
        item = QtWidgets.QTreeWidgetItem([name])
        if is_header:
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsSelectable)  # Disable selection
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsDragEnabled)  # Disable drag
        else:
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsDropEnabled)  # Disable drop
        return item

    def init_ui(self):
        self.tree_widget = QtWidgets.QTreeWidget()
        self.tree_widget.setFont(QtGui.QFont("Arial", 12))
        self.tree_widget.setHeaderHidden(True)

        self.tree_widget.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        # Disable drop on the header
        self.tree_widget.invisibleRootItem().setFlags(
            self.tree_widget.invisibleRootItem().flags() & ~QtCore.Qt.ItemIsDropEnabled)

        self.visible_items_header = self.build_tree_widget_item("Visible widgets", True)
        self.tree_widget.insertTopLevelItem(0, self.visible_items_header)
        self.tree_widget.expandItem(self.visible_items_header)
        for item in self.visible_items:
            self.visible_items_header.addChild(self.build_tree_widget_item(item, False))

        self.hidden_items_header = self.build_tree_widget_item("Invisible widgets", True)
        self.tree_widget.insertTopLevelItem(1, self.hidden_items_header)
        self.tree_widget.expandItem(self.hidden_items_header)
        for item in self.hidden_items:
            self.hidden_items_header.addChild(self.build_tree_widget_item(item, False))

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.tree_widget)
        layout.addWidget(self.button_box)
        self.setLayout(layout)

    def accept(self) -> None:
        """Updates the visible and hidden items."""
        self.visible_items = []
        self.hidden_items = []
        for kk in list(range(self.visible_items_header.childCount())):
            self.visible_items.append(self.visible_items_header.child(kk).text(0))
        for kk in list(range(self.hidden_items_header.childCount())):
            self.hidden_items.append(self.hidden_items_header.child(kk).text(0))
        return super().accept()


class _CustomListWidgetItem(QtWidgets.QListWidgetItem):
    """QListWidgetItem with custom sorting. Don't use this class outside of this module."""
    def __init__(self, sort_index):
        super().__init__()
        self.sort_index = sort_index

    def __lt__(self, other):
        return self.sort_index < other.sort_index
