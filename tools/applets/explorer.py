import asyncio
import importlib.util
import logging
import os
import pickle
from functools import partial

from PyQt5 import QtGui, QtWidgets, QtCore
from artiq.applets.simple import SimpleApplet
from artiq.dashboard import explorer
from artiq.gui.models import ModelSubscriber
from artiq.master.worker_impl import ExamineDeviceMgr, ExamineDatasetMgr, TraceArgumentManager

from jax import JaxApplet


class Explorer(QtWidgets.QWidget, JaxApplet):
    def __init__(self, args, **kwds):
        super().__init__(**kwds)
        self.setDisabled(True)

        self._disconnected_reported = False
        self._explist_sub = ModelSubscriber("explist", explorer.Model)
        self._explist_status_sub = ModelSubscriber("explist_status", explorer.StatusUpdater)

        self.initialize_gui()
        self.connect_to_labrad(args.ip)
        asyncio.run_coroutine_threadsafe(
            self.start_subscribers(),
            asyncio.get_event_loop()
        )

    async def start_subscribers(self):
        port_notify = 3250
        await self._explist_sub.connect("::1", port_notify)
        await self._explist_status_sub.connect("::1", port_notify)

    def initialize_gui(self):
        layout = QtWidgets.QGridLayout()
        self.stack = QtWidgets.QStackedWidget()
        layout.addWidget(self.stack, 0, 0)
        self.explorer = QtWidgets.QTreeView()
        self.explorer.setHeaderHidden(True)
        self.explorer.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectItems)
        self.stack.addWidget(self.explorer)

        submit = QtWidgets.QPushButton("Submit")
        submit.setIcon(
            QtWidgets.QApplication.style().standardIcon(QtWidgets.QStyle.SP_DialogOkButton)
        )
        submit.setToolTip("Schedule the selected experiment")
        layout.addWidget(submit, 1, 0)
        submit.clicked.connect(partial(self.expname_action, "submit"))

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
        return (expinfo["file"], expinfo["class_name"])

    async def _submit_experiment(self, expurl):
        """TODO: Handle experiment parameters, priority, pipeline, and log_level correctly."""
        file, class_name = self._resolve_expurl(expurl)
        await self.artiq.schedule_experiment_with_parameters(file, class_name)

    def expname_action(self, action):
        expname = self._get_selected_expname()
        if expname is not None:
            if action == "submit":
                self.run_in_labrad_loop(self._submit_experiment)(expname)

    def update_scanning(self, scanning):
        """TODO: This is not working."""
        if scanning:
            self.stack.setCurrentWidget(self.waiting_panel)
            self.waiting_panel.start()
        else:
            self.stack.setCurrentWidget(self.el_buttons)
            self.waiting_panel.stop()

    def update_cur_rev(self, cur_rev):
        """This function is not implemented.

        We do not show the current revision of the repository in the GUI
        as we do not use the git backend of the artiq experiment manager.
        This function is for compatibility with `artiq.dashboard.explorer.StatusUpdater`.
        """
        pass

    def get_experiment_parameters(self, file, class_name):
        module_name = os.path.basename(file).split(".")[0]
        spec = importlib.util.spec_from_file_location(module_name, file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls = getattr(module, class_name)
        exp = cls(ExamineDeviceMgr, ExamineDatasetMgr, TraceArgumentManager(), {})
        return exp.parameter_paths


def main():
    applet = SimpleApplet(Explorer)
    Explorer.add_labrad_ip_argument(applet)
    applet.run()


if __name__ == "__main__":
    main()


