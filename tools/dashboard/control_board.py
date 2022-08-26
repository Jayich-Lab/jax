#!/usr/bin/env python3

import argparse
import asyncio
import atexit
import logging
import os

from artiq import __artiq_dir__ as artiq_dir
from artiq import __version__ as artiq_version
from artiq.dashboard import applets_ccb, datasets
from artiq.gui import log, state
from artiq.gui.models import ModelSubscriber
from artiq.tools import get_user_config_dir
from jax.tools.applets.dds import DDS
from jax.tools.applets.explorer import Explorer
from jax.tools.applets.schedule import Schedule
from PyQt5 import QtCore, QtGui, QtWidgets
from qasync import QEventLoop
from sipyco import common_args
from sipyco.asyncio_tools import atexit_register_coroutine
from sipyco.broadcast import Receiver
from sipyco.pc_rpc import Client


def get_argparser():
    parser = argparse.ArgumentParser(description="ARTIQ Control Dashboard")
    parser.add_argument(
        "--version",
        action="version",
        version="ARTIQ v{}".format(artiq_version),
        help="print the ARTIQ version number",
    )
    parser.add_argument(
        "-s",
        "--server",
        default="::1",
        help="hostname or IP of the master to connect to",
    )
    parser.add_argument(
        "--port-notify",
        default=3250,
        type=int,
        help="TCP port to connect to for notifications",
    )
    parser.add_argument(
        "--port-control",
        default=3251,
        type=int,
        help="TCP port to connect to for control",
    )
    parser.add_argument(
        "--port-broadcast",
        default=1067,
        type=int,
        help="TCP port to connect to for broadcasts",
    )
    parser.add_argument(
        "--db-file", default=None, help="database file for local GUI settings"
    )
    common_args.verbosity_args(parser)
    return parser


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, server):
        QtWidgets.QMainWindow.__init__(self)

        # icon = QtGui.QIcon(os.path.join(artiq_dir, "gui", "logo.svg"))
        # self.setWindowIcon(icon)
        self.setWindowTitle("ARTIQ Control Dashboard - {}".format(server))

        qfm = QtGui.QFontMetrics(self.font())
        self.resize(140 * qfm.averageCharWidth(), 38 * qfm.lineSpacing())

        self.exit_request = asyncio.Event()

    def closeEvent(self, event):
        event.ignore()
        self.exit_request.set()

    def save_state(self):
        return {
            "state": bytes(self.saveState()),
            "geometry": bytes(self.saveGeometry()),
        }

    def restore_state(self, state):
        self.restoreGeometry(QtCore.QByteArray(state["geometry"]))
        self.restoreState(QtCore.QByteArray(state["state"]))


class MdiArea(QtWidgets.QMdiArea):
    def __init__(self):
        QtWidgets.QMdiArea.__init__(self)


def main():
    # initialize application
    args = get_argparser().parse_args()
    widget_log_handler = log.init_log(args, "dashboard")

    if args.db_file is None:
        args.db_file = os.path.join(
            get_user_config_dir(),
            "artiq_control_dashboard_{server}_{port}.pyon".format(
                server=args.server.replace(":", "."), port=args.port_notify
            ),
        )

    app = QtWidgets.QApplication(["ARTIQ Control Dashboard"])
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    atexit.register(loop.close)
    smgr = state.StateManager(args.db_file)

    config = Client(args.server, args.port_control, "master_config")
    try:
        server_name = config.get_name()
    finally:
        config.close_rpc()

    disconnect_reported = False

    def report_disconnect():
        nonlocal disconnect_reported
        if not disconnect_reported:
            logging.error(
                "connection to master lost, " "restart dashboard to reconnect"
            )
        disconnect_reported = True

    sub_clients = dict()
    for notifier_name, modelf in (("datasets", datasets.Model),):
        subscriber = ModelSubscriber(notifier_name, modelf, report_disconnect)
        loop.run_until_complete(subscriber.connect(args.server, args.port_notify))
        atexit_register_coroutine(subscriber.close)
        sub_clients[notifier_name] = subscriber

    broadcast_clients = dict()
    for target in "log", "ccb":
        client = Receiver(target, [], report_disconnect)
        loop.run_until_complete(client.connect(args.server, args.port_broadcast))
        atexit_register_coroutine(client.close)
        broadcast_clients[target] = client

    # initialize main window
    main_window = MainWindow(args.server if server_name is None else server_name)
    smgr.register(main_window)
    mdi_area = MdiArea()
    mdi_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
    mdi_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
    main_window.setCentralWidget(mdi_area)

    # create UI components

    d_applets = applets_ccb.AppletsCCBDock(main_window, sub_clients["datasets"])
    atexit_register_coroutine(d_applets.stop)
    smgr.register(d_applets)
    broadcast_clients["ccb"].notify_cbs.append(d_applets.ccb_notify)

    logmgr = log.LogDockManager(main_window)
    smgr.register(logmgr)
    broadcast_clients["log"].notify_cbs.append(logmgr.append_message)
    widget_log_handler.callback = logmgr.append_message

    # lay out docks
    right_docks = [d_applets]
    main_window.addDockWidget(QtCore.Qt.RightDockWidgetArea, right_docks[0])
    for d1, d2 in zip(right_docks, right_docks[1:]):
        main_window.tabifyDockWidget(d1, d2)

    # load/initialize state
    if os.name == "nt":
        # HACK: show the main window before creating applets.
        # Otherwise, the windows of those applets that are in detached
        # QDockWidgets fail to be embedded.
        main_window.show()
    smgr.load()
    smgr.start()
    atexit_register_coroutine(smgr.stop)

    # create first log dock if not already in state
    d_log0 = logmgr.first_log_dock()
    if d_log0 is not None:
        main_window.tabifyDockWidget(d_log0)

    if server_name is not None:
        server_description = server_name + " ({})".format(args.server)
    else:
        server_description = args.server
    logging.info(
        "ARTIQ dashboard %s connected to %s", artiq_version, server_description
    )

    # run
    main_window.show()
    loop.run_until_complete(main_window.exit_request.wait())


if __name__ == "__main__":
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    main()
