import asyncio
import importlib.util
import logging
import os
import pickle

from artiq.applets.simple import SimpleApplet
from artiq.dashboard import explorer
from artiq.gui.models import ModelSubscriber
from artiq.master.worker_impl import (ExamineDatasetMgr, ExamineDeviceMgr,
                                      TraceArgumentManager)
from jax import JaxApplet
from jax.util.ui.dialog_on_top import DialogOnTop
from PyQt5 import QtCore, QtGui, QtWidgets


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


class ExperimentDetails(DialogOnTop):
    """A dialog showing details of an experiment."""

    def __init__(self, expurl, parent):
        super().__init__(parent)
        self.expurl = expurl
        self._expinfo = self.parent()._resolve_expurl(expurl)
        self.setWindowTitle(self._expinfo["class_name"])
        self.initialize_gui()

    def initialize_gui(self):
        grid = QtWidgets.QGridLayout()
        self.setLayout(grid)

        label_priority = QtWidgets.QLabel("Priority")
        label_priority.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        grid.addWidget(label_priority, 0, 0)

        self.spinbox_priority = QtWidgets.QSpinBox()
        self.spinbox_priority.setMinimum(-100)
        self.spinbox_priority.setMaximum(100)
        default_priority = self._expinfo["scheduler_defaults"].get("priority", 0)
        self.spinbox_priority.setValue(default_priority)
        grid.addWidget(self.spinbox_priority, 0, 1)

        label_pipeline = QtWidgets.QLabel("Pipeline")
        label_pipeline.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        grid.addWidget(label_pipeline, 1, 0)

        self.textbox_pipeline = QtWidgets.QLineEdit()
        default_pipeline = self._expinfo["scheduler_defaults"].get(
            "pipeline_name", "main"
        )
        self.textbox_pipeline.setText(default_pipeline)
        grid.addWidget(self.textbox_pipeline, 1, 1)

        label_log_level = QtWidgets.QLabel("Log level")
        label_log_level.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        grid.addWidget(label_log_level, 2, 0)

        self.combobox_log_level = QtWidgets.QComboBox()
        self.combobox_log_level.addItems(
            ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        )
        self.combobox_log_level.setCurrentIndex(1)
        grid.addWidget(self.combobox_log_level, 2, 1)

        self.checkbox_preload = QtWidgets.QCheckBox("Preload parameters when scheduled")
        self.checkbox_preload.setChecked(True)
        self.checkbox_preload.setToolTip(
            "Uses the parameters when the experiment is scheduled."
        )
        grid.addWidget(self.checkbox_preload, 3, 0, 1, 2)

        submit = QtWidgets.QPushButton("Submit")
        submit.setIcon(
            QtWidgets.QApplication.style().standardIcon(
                QtWidgets.QStyle.SP_DialogOkButton
            )
        )
        submit.clicked.connect(self.submit_clicked)
        submit.setToolTip("Schedule the selected experiment")
        grid.addWidget(submit, 4, 0, 1, 2)

    def submit_clicked(self):
        self.parent().submit(
            self.checkbox_preload.isChecked(),
            self.spinbox_priority.value(),
            self.textbox_pipeline.text(),
            getattr(logging, self.combobox_log_level.currentText()),
            self.expurl,
        )
        self.close()


class Explorer(QtWidgets.QDockWidget, JaxApplet):
    """Experiment explorer.

    Modified from artiq.dashboard.explorer.ExplorerDock.
    It supports running the experiment with parameter preloaded when the experiment
    is scheduled. It can be connected with the parameter bank GUI to show only
    relevant parameters for the experiment chosen.
    This applet needs to be restarted when the ARTIQ master restarts.
    """

    # subscribe to this method to update when an experiment is selected / deselected.
    # TODO: add a blank experiment to show all parameters.
    parameters_updated = QtCore.pyqtSignal(object)

    def __init__(self, args, **kwds):
        super().__init__(**kwds)
        self.setObjectName("Explorer")
        self.setDisabled(True)
        self.initialize_gui()

        self._disconnect_reported = False
        asyncio.get_event_loop().run_until_complete(
            self.connect_subscribers()  # run in the main thread asyncio loop.
        )

        self.connect_to_labrad("::1")

    def initialize_gui(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QGridLayout()

        self.stack = (
            QtWidgets.QStackedWidget()
        )  # either shows the explorer or the waiting panel.
        layout.addWidget(self.stack, 0, 0)

        self.explorer = QtWidgets.QTreeView()
        self.explorer.setHeaderHidden(True)
        self.explorer.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectItems)
        self.stack.addWidget(self.explorer)

        submit = QtWidgets.QPushButton("Submit")
        submit.setIcon(
            QtWidgets.QApplication.style().standardIcon(
                QtWidgets.QStyle.SP_DialogOkButton
            )
        )
        submit.setToolTip("Schedule the selected experiment")
        layout.addWidget(submit, 1, 0)
        submit.clicked.connect(self.submit)

        self._create_context_menu()

        self.waiting_panel = explorer.WaitingPanel()
        self.stack.addWidget(self.waiting_panel)

        self.repo_path = None
        self._parameters_initialized = True
        self._experiment_parameters = {}
        widget.setLayout(layout)
        self.setWidget(widget)

    def _create_context_menu(self):
        self.explorer.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)

        submit_action = QtWidgets.QAction("Submit", self.explorer)
        submit_action.triggered.connect(self.submit)
        self.explorer.addAction(submit_action)

        detail_action = QtWidgets.QAction("Details", self.explorer)
        detail_action.triggered.connect(self.show_details)
        self.explorer.addAction(detail_action)

        scan_repository_action = QtWidgets.QAction("Scan repository", self.explorer)
        scan_repository_action.triggered.connect(self.scan_repository)
        self.explorer.addAction(scan_repository_action)

        scan_devices_action = QtWidgets.QAction("Scan devices", self.explorer)
        scan_devices_action.triggered.connect(self.scan_devices)
        self.explorer.addAction(scan_devices_action)

    async def connect_subscribers(self):
        """Connect subscribers to ARTIQ notification publishers.

        The subscribers and publishers are similar to LabRAD signals. We need to get
        updates of the experiment management using them. They only work on localhost.
        """
        localhost = "::1"
        port_notify = 3250
        self._explist_sub = ModelSubscriber(
            "explist", explorer.Model, self._report_disconnect
        )
        await self._explist_sub.connect(localhost, port_notify)
        self._explist_status_sub = ModelSubscriber(
            "explist_status", StatusUpdater, self._report_disconnect
        )
        await self._explist_status_sub.connect(localhost, port_notify)

        self.explist_model = explorer.Model(dict())
        self.explorer.setModel(self.explist_model)
        self.explorer.selectionModel().selectionChanged.connect(self.selection_changed)
        self._explist_sub.add_setmodel_callback(self.set_model)
        self._explist_status_sub.add_setmodel_callback(
            lambda updater: updater.set_explorer(self)
        )

    def _report_disconnect(self):
        """Handles subscriber disconnections.

        Currently the applet must be restarted after a subscriber disconnection which
        is typically due to ARTIQ master stopping. We can reconnect when the ARTIQ server
        restarts, or when the ARTIQ master is restarted using the ARTIQ server is restarted.
        It is not implemented now due to the complexity with communication between LabRAD
        and ARTIQ event loops. This can be implemented easier with rockdove.
        """
        if not self._disconnect_reported:
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
        device_db = await self.artiq.get_device_db()
        self.device_db = pickle.loads(device_db)
        # patch the `ExamineDeviceMgr.get_device_db` method.
        ExamineDeviceMgr.get_device_db = lambda: self.device_db
        self.repo_path = await self.artiq.get_repository_path()

        if not self._parameters_initialized:
            self._get_all_experiment_parameters()
            self._parameters_initialized = True

        if not self._disconnect_reported:
            self.setDisabled(False)

    def artiq_disconnected(self):
        self.setDisabled(True)

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
        """Called when the experiment list subscriber receives an update."""
        self.explist_model = model
        self.explorer.setModel(model)

    def _get_selected_expurl(self):
        selection = self.explorer.selectedIndexes()
        if selection:
            return self.explist_model.index_to_key(selection[0])
        else:
            return None

    def _resolve_expurl(self, expurl):
        expinfo = self.explist_model.backing_store[expurl]
        return expinfo

    def show_details(self):
        expurl = self._get_selected_expurl()
        details = ExperimentDetails(expurl, self)
        details.exec_()

    async def _submit_experiment(
        self,
        expurl,
        preload_parameters=True,
        priority=None,
        pipeline=None,
        log_level=None,
    ):
        expinfo = self._resolve_expurl(expurl)
        filename = expinfo["file"]
        class_name = expinfo["class_name"]
        if priority is None:
            priority = expinfo["scheduler_defaults"].get("priority", 0)
        if pipeline is None:
            pipeline = expinfo["scheduler_defaults"].get("pipeline_name", "main")
        if log_level is None:
            log_level = 20

        if preload_parameters:
            # If the experiment is changed after last repository scan, the parameters
            # may have been changed and the cache saved in `self._experiment_parameters`
            # should not be used.
            required_params = self._get_experiment_parameters(filename, class_name)
        else:
            required_params = []
        parameter_override_list = []  # TODO: implement parameter scanning / overriding.

        rid = await self.artiq.schedule_experiment_with_parameters(
            filename,
            class_name,
            required_params,
            parameter_override_list,
            priority,
            pipeline,
            log_level,
        )
        print(f"{class_name} of RID {rid} is scheduled")

    def submit(
        self,
        preload_parameters=True,
        priority=None,
        pipeline=None,
        log_level=None,
        expurl=None,
    ):
        if expurl is None:
            expurl = self._get_selected_expurl()
        if expurl is not None:
            self.run_in_labrad_loop(self._submit_experiment)(
                expurl=expurl,
                preload_parameters=preload_parameters,
                priority=priority,
                pipeline=pipeline,
                log_level=log_level,
            )

    def update_scanning(self, scanning):
        """When the experiment list status subscriber is updated.

        After a scanning, updates the experiment parameters cache and triggers
        the parameters_updated signal.
        """
        if scanning:
            self.stack.setCurrentWidget(self.waiting_panel)
            self.waiting_panel.start()
        else:
            if self.repo_path is not None:
                self._get_all_experiment_parameters()
            else:
                self._parameters_initialized = False
            self.stack.setCurrentWidget(self.explorer)
            self.selection_changed(None, None)
            self.waiting_panel.stop()

    def selection_changed(self, selected, deselected):
        """Triggers the parameter_updated signal."""
        expurl = self._get_selected_expurl()
        if expurl is None and expurl in self._experiment_parameters:
            self.parameters_updated.emit(self._experiment_parameters[expurl])
        else:
            self.parameters_updated.emit([])

    def _get_all_experiment_parameters(self):
        self._experiment_parameters = {}
        for expurl in self.explist_model.backing_store:
            expinfo = self._resolve_expurl(expurl)
            filename = expinfo["file"]
            class_name = expinfo["class_name"]
            try:
                self._experiment_parameters[expurl] = self._get_experiment_parameters(
                    filename, class_name
                )
            except Exception:
                print(f"Cannot load the parameters in {class_name} of {filename}.")

    def _get_experiment_parameters(self, filename, class_name):
        """Creates the experiment class and tries to get the required parameters of the experiment.

        It imports the experiment module, builds the class, and tries to read the `parameter_paths`
        attribute of the experiment.

        Args:
            filename: str, file path from the repository root.
            class_name: str, name of the experiment class.

        Returns:
            list of 2-tuples of strs, list of required parameters in
            (collection_name, parameter_name). If the experiment does not have required parmeters,
            it returns an empty list.
        """
        module_name = os.path.basename(filename).split(".")[0]
        filename = os.path.join(self.repo_path, filename)
        spec = importlib.util.spec_from_file_location(module_name, filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls = getattr(module, class_name)
        exp = cls((ExamineDeviceMgr, ExamineDatasetMgr, TraceArgumentManager(), {}))
        try:
            parameter_paths = exp.parameter_paths
        except AttributeError as e:
            parameter_paths = []  # allows experiments without parameter_paths to run.
        return parameter_paths

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        """Closes subscribers when the applet is closed."""

        async def close_subscribers():
            try:
                await self._explist_sub.close()
            except Exception as e:
                pass
            try:
                await self._explist_status_sub.close()
            except Exception as e:
                pass

        asyncio.run_coroutine_threadsafe(close_subscribers(), asyncio.get_event_loop())
        return super().closeEvent(a0)

    def save_state(self):
        return {}

    def restore_state(self, state):
        pass


def main():
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    applet = SimpleApplet(Explorer)
    applet.run()


if __name__ == "__main__":
    main()
