import importlib.util
import sys
import tempfile
from triton.backends.driver import DriverBase
from triton.backends.compiler import GPUTarget

# This is no-op.
class ExampleLauncher(object):
    def __init__(self, src, metadata):
        pass

    def __call__(
        self,
        gridX, gridY, gridZ, stream, module,
        kernel_metadata, launch_metadata,
        launch_enter_hook, launch_exit_hook, *args
    ):
        module(*args)

class ExampleUtils(object):
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(ExampleUtils, cls).__new__(cls)
        return cls.instance

    # Dummy.
    @staticmethod
    def get_device_properties(device):
        return {
          "max_shared_mem": 2 ** 20,
          "multiprocessor_count": None,
          "sm_clock_rate": None,
          "mem_clock_rate": None,
          "mem_bus_width": None
        }

    @staticmethod
    def load_binary(name, kernel_asm, shared, device):
        mod_name = "fack"
        def fake_wrapped (*args, **kwargs):
            print ("hello world!")
        return (
          mod_name,   # module
          fake_wrapped,    # function
          None,       # n_regs
          None,       # n_spills
          1024        # n_max_threads
        )

class TVMDriver(DriverBase):

    def __init__(self):
        super().__init__()

class ExampleDriver(DriverBase):

    def __init__(self):
        super().__init__()
        self.launcher_cls = ExampleLauncher
        self.utils = ExampleUtils()

    # Remember to use triton.runtime.driver.set_active(TVMDriver())
    @staticmethod
    def is_active():
        return False

    def get_device_capability(self):
        return ("example", 0)

    def get_current_stream(self, device):
        return None

    def get_current_device(self):
        return "example"

    def set_current_device(self, device):
        assert device == "example"
        return

    def get_current_target(self):
        return GPUTarget("example", 0, 0)

    def assemble_tensormap_to_arg(self, tensormaps_info, args):
        return args

    def get_active_torch_device():
        pass

    def get_benchmarker():
        pass

    def map_python_to_cpp_type():
        pass

