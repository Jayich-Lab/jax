import pickle
import threading

import numpy as _np
from artiq.experiment import *
from jax.util.drift_tracker import DriftTracker
from jax.util.parameter_group import ParameterGroup

__all__ = ["JaxEnvironment"]


class JaxEnvironment(HasEnvironment):
    """Environment for Jayich lab experiments.

    Set USE_LABRAD to False in experiments that do not need labrad.

    Attributes:
        scheduler: ARTIQ scheduler device.
        cxn: labrad connection.
        dv: labrad vault server.
        parameter_paths: list of (collection, parameter), paths of parameters.
            Populated when calling self.get_parameter_paths().
        p: ParameterGroup, experiment parameters. Populated when calling self.save_parameters().
        drift_trackers: a dict of {name: DriftTracker}, drift trackers used by the experiment.
    """

    USE_LABRAD = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.drift_trackers = {}
        self._is_dataset_open = False

    def _connect_labrad(self):
        import labrad

        self.cxn = labrad.connect()
        try:
            self.dv = self.cxn.vault
        except Exception as e:
            print("Data vault is not connected.")

    def _get_experiment_info(self):
        """Gets the current experiment information from the scheduler.

        Returns:
            (rid, pipeline_name, priority, expid)
        """
        rid = self.scheduler.rid
        pipeline_name = self.scheduler.pipeline_name
        priority = self.scheduler.priority
        expid = self.scheduler.expid
        return (rid, pipeline_name, priority, expid)

    def build(self):
        """Building the experiment.

        Called when the experiment is loaded in the scheduler, and before the experiment prepares.
        Can be overriden by derived classes.
        """
        self.setattr_device("scheduler")
        self._dv_lock = threading.Lock()

    def prepare(self):
        """Prepare for the experiment.

        Called when the experiment is next in the queue to run.
        Can be overriden by derived classes.
        """
        if self.USE_LABRAD:
            self._connect_labrad()

    @host_only
    def disconnect_labrad(self):
        self.cxn.disconnect()

    @host_only
    def check_stop_or_do_pause(self):
        """Checks if the experiment should stop or pause in the host code.

        To check in the kernel if the experiment should stop or pause,
        use self.scheduler.check_pause(). The experiment should then exit from kernel and call
        this function in the host code.

        If a higher priority experiment is queued, it closes the core and yield control.
        The function returns when this experiment is back at the top of the queue.
        If the user has stopped the experiment, it returns True.

        Returns:
            should_stop: bool, whether the user has stopped the experiment. If True,
                the experiment should be closed gracefully.
        """
        status = self.scheduler.get_status()
        same_pipeline_higher_priority = False

        for exp in status:
            same_pipeline = status[exp]["pipeline"] == self.scheduler.pipeline_name
            high_priority = status[exp]["priority"] > self.scheduler.priority
            if same_pipeline and high_priority:
                same_pipeline_higher_priority = True
                break
        if not self.scheduler.check_pause():
            return False
        elif same_pipeline_higher_priority:
            self.core.close()  # close the connection so other experiment can use it
            self.scheduler.pause()  # yield control to the scheduler
            return False
        return True

    @rpc
    def open_file(self):
        """Opens a data file to write to."""
        if self._is_dataset_open:
            return
        rid, pipeline_name, priority, expid = self._get_experiment_info()
        self.dv.open(expid["class_name"], True, rid)
        self.dv.add_attribute("rid", rid, "scheduler")
        self.dv.add_attribute("expid", self.serialize(expid), "scheduler")
        self.dv.add_attribute("pipeline_name", pipeline_name, "scheduler")
        self.dv.add_attribute("priority", priority, "scheduler")
        self._is_dataset_open = True

    @rpc
    def close_file(self):
        """Saves and closes a data file to write to."""
        if not self._is_dataset_open:
            return
        self.dv.close()
        self._is_dataset_open = False

    @rpc
    def add_attribute(self, name, value, group_path="/") -> TStr:
        """Adds an attribute.

        Args:
            name: str, name of the attribute.
            value: value of the attribute.
            group_path: str, path to the group to save the attribute at. Default "/", file root.

        Returns:
            str, key of the dataset. Use this key to call set_dataset() or append_dataset().
        """
        if not self._is_dataset_open:
            self.open_file()
        return self.dv.add_attribute(name, value, group_path)

    @rpc
    def add_dataset(self, name, value, group_path="/datasets", shared=False) -> TStr:
        """Adds a dataset.

        Args:
            name: str, dataset name.
            value: value of the dataset.
            group_path: str, path to the group to save the dataset at. Default "/datasets".
            shared: bool, make the dataset accessible to other labrad connections. Default False.

        Returns:
            str, key of the dataset. Use this key to call set_dataset() or append_dataset().
        """
        if not self._is_dataset_open:
            self.open_file()
        return self.dv.add_dataset(name, value, group_path, shared)

    @rpc
    def add_streaming_dataset(
        self, name, value, maxshape, rows_stream=1, group_path="/datasets", shared=False
    ) -> TStr:
        """Adds a streaming dataset that is automatically saved into the file.

        Args:
            name: str, dataset name.
            value: value of the dataset.
            maxshape: tuple, maximum shape of the data. Use None for axis that is unlimited.
            rows_stream: int, rows of data in the cache when saving to the file. Default 1.
            group_path: str, path to the group to save the dataset at. Default "/datasets".
            shared: bool, make the dataset accessible to other labrad connections. Default False.

        Returns:
            str, key of the dataset. Use this key to call set_dataset() or append_dataset().
        """
        if not self._is_dataset_open:
            self.open_file()
        return self.dv.add_streaming_dataset(
            name, value, maxshape, rows_stream, group_path, shared
        )

    @rpc(flags={"async"})
    def set_dataset(self, dataset_path, value):
        """Sets the value of the dataset or a streaming dataset.

        Args:
            dataset_path: str, path to the dataset.
            value: value of the dataset.
        """
        with self._dv_lock:
            self.dv.set_dataset(dataset_path, value)

    @rpc(flags={"async"})
    def append_dataset(self, dataset_path, value):
        """Appends to the value of the dataset or a streaming dataset.

        Args:
            dataset_path: str, path to the dataset.
            value: value to append to the dataset.
        """
        with self._dv_lock:
            self.dv.append_dataset(dataset_path, value)

    @rpc(flags={"async"})
    def set_shared_dataset(self, name, value):
        """Sets the value of a shared dataset.

        Args:
            name: str, shared dataset name. This is the name used in add_dataset()
                or add_streaming_dataset() as the argument, but not the dataset path.
            value: value of the shared dataset.
        """
        with self._dv_lock:
            self.dv.set_shared_dataset(name, value)

    @rpc(flags={"async"})
    def append_shared_dataset(self, name, value):
        """Appends to the value of a shared dataset.

        Args:
            name: str, shared dataset name. This is the name used in add_dataset()
                or add_streaming_dataset() as the argument, but not the dataset path.
            value: value to append to the shared dataset.
        """
        with self._dv_lock:
            self.dv.append_shared_dataset(name, value)

    @rpc(flags={"async"})
    def delete_shared_dataset(self, name):
        """Deletes the shared dataset.

        Args:
            name: str, shared dataset name. This is the name used in add_dataset()
                or add_streaming_dataset() as the argument, but not the dataset path.
        """
        with self._dv_lock:
            self.dv.delete_shared_dataset(name)

    @host_only
    def get_shared_dataset(self, name):
        """Gets the shared dataset.

        This function can only be called from host, as the return type is unknown at compile time.
        All shared dataset it gets are saved in "/shared" group in the data file to archive.

        Args:
            name: str, shared dataset name. This is the name used in add_dataset()
                or add_streaming_dataset() as the argument, but not the dataset path.

        Returns:
            dataset value.
        """
        if not self._is_dataset_open:
            self.open_file()
        with self._dv_lock:
            value = self.dv.get_shared_dataset(name)
            self.add_dataset(name, value, "/shared", False)
        return value

    @host_only
    def get_parameter_paths(self, experiment_parameters=[], pulse_sequence_classes=[]):
        """Populates self.parameter_paths with required parameters.

        Args:
            experiment_parameters: list of tuples, parameter needed for the experiment.
            pulse_sequence_classes: list of classes, pulse sequences classes needed.
        """
        self.parameter_paths = experiment_parameters
        for pulse_sequence_class in pulse_sequence_classes:
            self.parameter_paths.extend(pulse_sequence_class.all_required_parameters())
        self.parameter_paths = list(set(self.parameter_paths))

    @host_only
    def serialize(self, object):
        """Serializes an object to be saved in h5py."""
        return pickle.dumps(object, protocol=4)

    @host_only
    def save_parameters(self):
        """Loads all parameters into self.p.

        The function first tries to load cached parameters from the parameter bank. The cached
        parameters are either preloaded or overridden when the experiment is scheduled.
        Then it loads parameters not saved as cached parameters in the experiment.

        Also saves all parameters to the data file.
        The experiment parameter values are saved under the "parameters" key of the data file.
        The full form of the experiment parameters (including the parameter type, range, etc.)
        are saved under the "parameter_full" key of the data file.
        """
        from jax.util.labrad import remove_labrad_units

        pb = self.cxn.parameter_bank
        params_serialized, params_full_serialized = pb.get_cached_parameters(
            self.scheduler.rid
        )
        params = pickle.loads(params_serialized)
        params_full = pickle.loads(params_full_serialized)
        for collection, name in self.parameter_paths:
            if collection not in params:
                params[collection] = {}
                params_full[collection] = {}
            if name not in params[collection]:
                value = pb.get_parsed_value(collection, name)
            else:
                value = params[collection][name]
            params[collection][name] = remove_labrad_units(value)
            if name not in params_full[collection]:
                value_full = pb.get_raw_form(collection, name)
            else:
                value_full = params_full[collection][name]
            params_full[collection][name] = remove_labrad_units(value_full)
        self.p = ParameterGroup(params)
        self.add_attribute("parameters", self.serialize(params))
        self.add_attribute("parameters_full", self.serialize(params_full))

    @host_only
    def get_drift_tracker(self, name):
        if not self._is_dataset_open:
            self.open_file()
        if not hasattr(self, "drift_trackers"):
            self.drift_trackers = {}
        if name not in self.drift_trackers:
            value = self.cxn.drift_tracker.get_drift_tracker(name)
            self.drift_trackers[name] = DriftTracker(pickle.loads(value))
            self.add_attribute(name, value, "drift_trackers")
        return self.drift_trackers[name]
