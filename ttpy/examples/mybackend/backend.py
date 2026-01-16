from dataclasses import dataclass
import functools
import hashlib
import re
import tempfile
import triton
from triton.backends.compiler import BaseBackend, GPUTarget
from triton._C.libtriton import ir, passes, nvidia
from ttpy.triton_plugin.xdsl_base_backend import XdslBaseBackEnd

@dataclass(frozen=True)
class ExampleOptions:
    debug: bool = False
    arch: str | int = 0
    sanitize_overflow: bool= True
    cluster_dims: tuple = (1, 1, 1)
    num_warps: int = 4
    num_stages: int = 1
    num_ctas: int = 1
    shared: bool = False
    name: str = None
  
    def hash(self):
        key = '_'.join([f'{name}-{val}' for name, val in self.__dict__.items()])
        return hashlib.md5(key.encode("utf-8")).hexdigest() 

class ExampleBackend(XdslBaseBackEnd):
    binary_ext = 'exampleir'

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'example'

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)

    def get_codegen_implementation(self, options):
        return {"min_dot_size": lambda lhsType, rhsType: (1, 1, 1)}

    def pack_metadata(self, metadata):
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
            metadata.cluster_dims[0],
            metadata.cluster_dims[1],
            metadata.cluster_dims[2],
            metadata.name
        )

    def load_dialects(self, ctx):
        nvidia.load_dialects(ctx)
    
    def parse_options(self, opts):
        args = {'arch': self.target.arch}
        args.update({k: opts[k] for k in ExampleOptions.__dataclass_fields__.keys() if k in opts})
        return ExampleOptions(**args)

    def get_module_map(self):
        return {}

    @staticmethod
    def make_exampleir(mod, metadata, opt):

        return None

    def add_stages(self, stages, options, language):
        stages["ttir"] = lambda src, metadata: XdslBaseBackEnd.make_xdsl_ttir(src, metadata, options)
        stages["exampleir"] = lambda src, metadata: self.make_exampleir(src, metadata, options)

    @functools.lru_cache()
    def hash(self):
        return self.target

