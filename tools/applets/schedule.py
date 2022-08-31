import asyncio
from functools import partial

from artiq.applets.simple import SimpleApplet
from artiq.dashboard import schedule
from artiq.gui.models import ModelSubscriber
from jax import JaxApplet
from PyQt5 import QtCore, QtGui, QtWidgets


class Schedule(QtWidgets.QDockWidget, JaxApplet):
    """Shows scheduled experiments.

    Modified from artiq.dashboard.schedule.ScheduleDock.

    Experiment can be gracifully terminated if the experiment handles it,
    or the experiment can be forcefully deleted. When an experiment is terminated
    or deleted, the preloaded parameters stored in the parameter bank is cleared.

    It also adds context menu options to run the initialization and the background experiments.
    """

    def __init__(self, args, **kwds):
        super().__init__(**kwds)
        self.setObjectName("Schedule")
        self.setDisabled(True)

        self._disconnect_reported = False
        asyncio.get_event_loop().run_until_complete(
            self.connect_subscribers()  # run in the main thread asyncio loop.
        )

        self.initialize_gui()
        self.connect_to_labrad("::1")  # only works on localhost.

    def initialize_gui(self):
        self.table = QtWidgets.QTableView()
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.verticalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeToContents
        )
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionsMovable(True)
        self.setWidget(self.table)

        self._create_context_menu()

        self.table_model = schedule.Model(dict())
        self._schedule_sub.add_setmodel_callback(self.set_model)

        cw = QtGui.QFontMetrics(self.font()).averageCharWidth()
        h = self.table.horizontalHeader()
        h.resizeSection(0, 7 * cw)
        h.resizeSection(1, 12 * cw)
        h.resizeSection(2, 16 * cw)
        h.resizeSection(3, 6 * cw)
        h.resizeSection(4, 16 * cw)
        h.resizeSection(5, 30 * cw)
        h.resizeSection(6, 20 * cw)
        h.resizeSection(7, 20 * cw)

    def _create_context_menu(self):
        self.table.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)
        request_termination_action = QtWidgets.QAction(
            "Request termination", self.table
        )
        request_termination_action.triggered.connect(partial(self.delete_clicked, True))
        request_termination_action.setShortcut("DELETE")
        request_termination_action.setShortcutContext(QtCore.Qt.WidgetShortcut)
        self.table.addAction(request_termination_action)

        delete_action = QtWidgets.QAction("Delete", self.table)
        delete_action.triggered.connect(partial(self.delete_clicked, False))
        delete_action.setShortcut("SHIFT+DELETE")
        delete_action.setShortcutContext(QtCore.Qt.WidgetShortcut)
        self.table.addAction(delete_action)

        terminate_pipeline = QtWidgets.QAction(
            "Gracefully terminate all in pipeline", self.table
        )
        terminate_pipeline.triggered.connect(self.terminate_pipeline_clicked)
        self.table.addAction(terminate_pipeline)

        initialization = QtWidgets.QAction("Run initalization experiment", self.table)
        initialization.triggered.connect(self.run_initialization)
        self.table.addAction(initialization)

        background = QtWidgets.QAction("Run background experiment", self.table)
        background.triggered.connect(self.run_background)
        self.table.addAction(background)

    async def connect_subscribers(self):
        localhost = "::1"
        port_notify = 3250
        self._schedule_sub = ModelSubscriber(
            "schedule", schedule.Model, self._report_disconnect
        )
        await self._schedule_sub.connect(localhost, port_notify)

    def _report_disconnect(self):
        if not self._disconnected_reported:
            print("connection to master lost, restart dashboard to reconnect")
            self.setDisabled(True)
        self._disconnect_reported = True

    async def labrad_connected(self):
        await self.artiq_connected()
        await self.setup_cxn_listeners()

    async def setup_cxn_listeners(self):
        self.cxn.add_on_connect("artiq", self.run_in_labrad_loop(self.artiq_connected))
        self.cxn.add_on_disconnect("artiq", self.artiq_disconnected)

    async def artiq_connected(self):
        self.artiq = self.cxn.get_server("artiq")
        if not self._disconnect_reported:
            self.setDisabled(False)

    def artiq_disconnected(self):
        self.setDisabled(True)

    def set_model(self, model):
        self.table_model = model
        self.table.setModel(self.table_model)

    async def delete(self, rid, graceful):
        """Deletes or terminates an experiment.

        Uses the artiq server functions to delete or to terminate the experiment.
        The functions also clear the experiment parameters cached in the parameter bank.
        """
        if graceful:
            await self.artiq.request_terminate_experiment(rid)
        else:
            await self.artiq.delete_experiment(rid)

    def delete_clicked(self, graceful):
        idx = self.table.selectedIndexes()
        if idx:
            row = idx[0].row()
            rid = self.table_model.row_to_key[row]
            if graceful:
                print(f"Requested termination of RID {rid}", )
            else:
                print(f"Deleted RID {rid}")
            self.run_in_labrad_loop(self.delete)(rid, graceful)

    async def request_term_multiple(self, rids):
        """Terminates multiple experiments."""
        for rid in rids:
            try:
                await self.artiq.request_terminate_experiment(rid)
            except Exception as e:
                # May happen if the experiment has terminated by itself
                # while we were terminating others.
                print(
                    "failed to request termination of RID {rid}"
                )

    def terminate_pipeline_clicked(self):
        idx = self.table.selectedIndexes()
        if idx:
            row = idx[0].row()
            selected_rid = self.table_model.row_to_key[row]
            pipeline = self.table_model.backing_store[selected_rid]["pipeline"]
            print(
                "Requesting termination of all " "experiments in pipeline '{pipeline}'"
            )

            rids = set()
            for rid, info in self.table_model.backing_store.items():
                if info["pipeline"] == pipeline:
                    rids.add(rid)
            self.run_in_labrad_loop(self.request_term_multiple)(rids)

    def run_initialization(self):
        async def worker():
            await self.artiq.run_initialization_experiment()

        self.run_in_labrad_loop(worker)()

    def run_background(self):
        async def worker():
            await self.artiq.run_background_experiment()

        self.run_in_labrad_loop(worker)()

    def save_state(self):
        return bytes(self.table.horizontalHeader().saveState())

    def restore_state(self, state):
        self.table.horizontalHeader().restoreState(QtCore.QByteArray(state))


def main():
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    applet = SimpleApplet(Schedule)
    applet.run()


if __name__ == "__main__":
    main()
