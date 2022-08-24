__all__ = ["Devices"]


class Devices:
    """Sinara device manager.

    It gets lists of device names from device_db.py.

    It saves the devices requested by an experiment in lists.
    These lists can be used to reset used devices at the end of the experiment.
    Device types that are saved include urukuls, ad9910s, and ttl_outs. To access the devices
    used by an experiment in the kernel code, there must be at least one of such device in the
    sinara system, so the device can be used as a placeholder in the list.

    If the sinara system does not contain those devices or other device types need to be saved,
    this class can be inherited. Check jax.experiments.SinaraEnvironment about how to use a
    inherited device manager class.

    Attributes in this class are not guarenteed to have non-zero lengths, so they cannot be used
    reliably in ARTIQ python directly. Check whether they are empty in host code before using them
    in the kernel.

    Attributes:
        urukuls: a list of urukul CPLD names.
        ad9910s: a list of AD9910 DDS names.
        ttl_ins: a list of TTL input channel names.
        ttl_outs: a list of TTL output channel names.
        ttl_in_outs: a list of TTL input/output channel names. These channels are usually inputs.
            They cannot be used for both inputs and outputs without toggling a switch
            on the TTL board.
        urukuls_used: a list of tuple (name, object) of urukuls used in the experiment.
        ad9910s_used: a list of tuple (name, object) AD9910s used in the experiment.
        ttl_ins_used: a list of tuple (name, object) TTL inputs used in the experiment.
        ttl_outs_used: a list of tuple (name, object) TTL outputs used in the experiment.
        ttl_in_outs_used: a list of tuple (name, object) of TTL in/outs used in the experiment.

    Args:
        device_db: device_db dict from device_db.py.
    """

    kernel_invariants = {
        "urukuls",
        "ad9910s",
        "ttl_ins",
        "ttl_outs",
        "ttl_in_outs",
        "urukuls_used",
        "ad9910s_used",
        "ttl_ins_used",
        "ttl_outs_used",
        "ttl_in_outs_used",
    }

    def __init__(self, device_db):
        self._parse_device_db(device_db)

    def use_device(self, key, device):
        """Saves a device to the used device lists.

        Args:
            key: str, device name.
            device, device object.
        """
        if key in self.urukuls:
            if (key, device) not in self.urukuls_used:
                self.urukuls_used.append((key, device))
        if key in self.ad9910s:
            if (key, device) not in self.ad9910s_used:
                self.ad9910s_used.append((key, device))
        if key in self.ttl_ins:
            if (key, device) not in self.ttl_ins_used:
                self.ttl_ins_used.append((key, device))
        if key in self.ttl_outs:
            if (key, device) not in self.ttl_outs_used:
                self.ttl_outs_used.append((key, device))
        if key in self.ttl_in_outs:
            if (key, device) not in self.ttl_in_outs_used:
                self.ttl_in_outs_used.append((key, device))

    def _parse_device_db(self, device_db):
        self.urukuls = []
        self._urukul_io_updates = (
            []
        )  # We don't want to manually control these TTL outputs.
        self.ad9910s = []
        self._ad9910_sws = []  # We don't want to manually control these TTL outputs.
        self.ttl_ins = []
        self.ttl_outs = []
        self.ttl_in_outs = []

        self.urukuls_used = []
        self.ad9910s_used = []
        self.ttl_ins_used = []
        self.ttl_outs_used = []
        self.ttl_in_outs_used = []

        for kk in device_db:
            if "class" not in device_db[kk]:
                continue
            urukul_module = "artiq.coredevice.urukul"
            if (
                device_db[kk]["class"] == "CPLD"
                and device_db[kk]["module"] == urukul_module
            ):
                self.urukuls.append(kk)
                self._urukul_io_updates.append(
                    device_db[kk]["arguments"]["io_update_device"]
                )
            if device_db[kk]["class"] == "AD9910":
                self.ad9910s.append(kk)
                self._ad9910_sws.append(device_db[kk]["arguments"]["sw_device"])
        for kk in device_db:
            if "class" not in device_db[kk]:
                continue
            if device_db[kk]["class"] == "TTLIn":
                self.ttl_ins.append(kk)
            if device_db[kk]["class"] == "TTLOut":
                if kk not in self._ad9910_sws and kk not in self._urukul_io_updates:
                    self.ttl_outs.append(kk)
            if device_db[kk]["class"] == "TTLInOut":
                self.ttl_in_outs.append(kk)
