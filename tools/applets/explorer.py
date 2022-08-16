import asyncio
import importlib.util
import logging
import os
import pickle

from PyQt5 import QtGui, QtWidgets, QtCore
from artiq.applets.simple import SimpleApplet
from artiq.dashboard import explorer
from artiq.gui.models import ModelSubscriber
from artiq.master.worker_impl import ExamineDeviceMgr, ExamineDatasetMgr, TraceArgumentManager

from jax import JaxApplet


class StatusUpdater:
    """Stores and updates the status of the experiment explorer.

    Modified from artiq.dashboard.explorer.StatusUpdater.
    """
    def __init__(self, init):
        self.status = init
        self.explorer = None

    def set_explorer(self, explorer):
        self.explorer = explorer
        self.explorer.update_scanning(self.status["scanning"])

    def __setitem__(self, k, v):
        self.status[k] = v
        if self.explorer is not None:
            if k == "scanning":
                self.explorer.update_scanning(v)


class Explorer(QtWidgets.QWidget, JaxApplet):
    """TODO: priority, pipeline_name, log_level, docstrings."""
    def __init__(self, args, **kwds):
        super().__init__(**kwds)
        self.setDisabled(True)

        self._disconnected_reported = False
        asyncio.get_event_loop().run_until_complete(
            self.connect_subscribers()  # run in the main thread asyncio loop.
        )

        self.initialize_gui()
        self.connect_to_labrad("::1")  # only works on localhost.

    async def connect_subscribers(self):
        localhost = "::1"
        port_notify = 3250
        self._explist_sub = ModelSubscriber("explist", explorer.Model, self._report_disconnect)
        await self._explist_sub.connect(localhost, port_notify)
        self._explist_status_sub = ModelSubscriber(
            "explist_status", StatusUpdater, self._report_disconnect
        )
        await self._explist_status_sub.connect(localhost, port_notify)

    def initialize_gui(self):
        font = QtGui.QFont()
        layout = QtWidgets.QGridLayout()
        self.stack = QtWidgets.QStackedWidget()
        layout.addWidget(self.stack, 0, 0)
        self.explorer = QtWidgets.QTreeView()
        self.explorer.setHeaderHidden(True)
        self.explorer.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectItems)
        self.explorer.setFont(font)
        self.stack.addWidget(self.explorer)

        submit = QtWidgets.QPushButton("Submit")
        submit.setIcon(
            QtWidgets.QApplication.style().standardIcon(QtWidgets.QStyle.SP_DialogOkButton)
        )
        submit.setFont(font)
        submit.setToolTip("Schedule the selected experiment")
        layout.addWidget(submit, 1, 0)
        submit.clicked.connect(self.submit)

        self.explist_model = explorer.Model(dict())
        self._explist_sub.add_setmodel_callback(self.set_model)

        self.explorer.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)
        scan_repository_action = QtWidgets.QAction("Scan repository", self.explorer)
        scan_repository_action.triggered.connect(self.scan_repository)
        self.explorer.addAction(scan_repository_action)

        scan_devices_action = QtWidgets.QAction("Scan devices", self.explorer)
        scan_devices_action.triggered.connect(self.scan_devices)
        self.explorer.addAction(scan_devices_action)

        self.waiting_panel = explorer.WaitingPanel()
        self.stack.addWidget(self.waiting_panel)
        self._explist_status_sub.add_setmodel_callback(
            lambda updater: updater.set_explorer(self))

        self.setLayout(layout)

    async def labrad_connected(self):
        await self.artiq_connected()
        await self.setup_cxn_listeners()

    async def setup_cxn_listeners(self):
        self.cxn.add_on_connect("artiq", self.run_in_labrad_loop(self.artiq_connected))
        self.cxn.add_on_disconnect("artiq", self.artiq_disconnected)

    async def artiq_connected(self):
        self.artiq = self.cxn.get_server("artiq")
        self.repo_path = await self.artiq.get_repository_path()
        device_db = await self.artiq.get_device_db()
        self.device_db = pickle.loads(device_db)
        ExamineDeviceMgr.get_device_db = lambda: self.device_db
        self.setDisabled(False)

    def artiq_disconnected(self):
        self.setDisabled(True)

    def _report_disconnect(self):
        if not self._disconnected_reported:
            logging.error("connection to master lost, restart dashboard to reconnect")
            self.setDisabled(True)
        self._disconnect_reported = True

    def scan_repository(self):
        async def worker():
            await self.artiq.scan_experiment_repository(False)

        self.run_in_labrad_loop(worker)()

    def scan_devices(self):
        async def worker():
            await self.artiq.scan_device_db()
            device_db = await self.artiq.get_device_db()
            self.device_db = pickle.loads(device_db)
            ExamineDeviceMgr.get_device_db = lambda: self.device_db

        self.run_in_labrad_loop(worker)()

    def set_model(self, model):
        self.explist_model = model
        self.explorer.setModel(model)

    def _get_selected_expname(self):
        selection = self.explorer.selectedIndexes()
        if selection:
            return self.explist_model.index_to_key(selection[0])
        else:
            return None

    def _resolve_expurl(self, expurl):
        expinfo = self.explist_model.backing_store[expurl]
        return expinfo

    async def _submit_experiment(self, expurl):
        expinfo = self._resolve_expurl(expurl)
        file = expinfo["file"]
        class_name = expinfo["class_name"]
        priority = expinfo["scheduler_defaults"].get("priority", 0)
        pipeline = expinfo["scheduler_defaults"].get("pipeline_name", "main")
        required_params = self._get_experiment_parameters(
            os.path.join(self.repo_path, file), class_name
        )
        parameter_override_list = []
        log_level = 20
        await self.artiq.schedule_experiment_with_parameters(
            file,
            class_name,
            required_params,
            parameter_override_list,
            priority,
            pipeline,
            log_level,
        )

    def submit(self):
        expname = self._get_selected_expname()
        if expname is not None:
            self.run_in_labrad_loop(self._submit_experiment)(expname)

    def update_scanning(self, scanning):
        if scanning:
            self.stack.setCurrentWidget(self.waiting_panel)
            self.waiting_panel.start()
        else:
            self.stack.setCurrentWidget(self.explorer)
            self.waiting_panel.stop()

    def _get_experiment_parameters(self, file, class_name):
        module_name = os.path.basename(file).split(".")[0]
        file = os.path.join("C:\\Users\\scientist\\code\\spock", file)
        spec = importlib.util.spec_from_file_location(module_name, file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls = getattr(module, class_name)
        exp = cls((ExamineDeviceMgr, ExamineDatasetMgr, TraceArgumentManager(), {}))
        try:
            parameter_paths = exp.parameter_paths
        except AttributeError as e:
            parameter_paths = []  # allows experiments without parameter_paths to run.
        return parameter_paths


def main():
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    applet = SimpleApplet(Explorer)
    applet.run()


if __name__ == "__main__":
    main()


