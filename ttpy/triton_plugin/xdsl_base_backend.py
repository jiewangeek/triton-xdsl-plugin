from dataclasses import dataclass
import functools
import hashlib
import re
import tempfile
import triton
from triton.backends.compiler import BaseBackend, GPUTarget
from triton._C.libtriton import ir, passes, nvidia
from triton.backends import backends, Backend
from xdsl.dialects import builtin, func, arith, scf

from io import StringIO
from xdsl.context import Context
from xdsl.parser import Parser
from xdsl.printer import Printer

class XdslBaseBackEnd(BaseBackend):
    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.ctx = Context()
        self.ctx.load_dialect(builtin.Builtin)
        self.ctx.load_dialect(func.Func)
        self.ctx.load_dialect(arith.Arith)
        self.ctx.load_dialect(scf.Scf)

        #TODO: it seems there are problems in loading the IR
        #from ttpy.dialects import ttir
        #self.ctx.load_dialect(ttir.Triton_dialect)

    @staticmethod    
    def convert_to_xdsl_ir(mod):
        mlir_str = str(mod)
        print('===== MLIR before converting: ', dir(mod), mlir_str)
        # Create MLIR context and register dialects

        # Parse the MLIR string
        parser = Parser(self.ctx, mlir_str)
        module = parser.parse_module()

        # Print the parsed IR
        printer = Printer()
        printer.print(module)
        print()  # newline

        return module

    @staticmethod
    def make_xdsl_ttir(mod, metadata, opt):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_combine(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_licm(pm)
        passes.common.add_symbol_dce(pm)
        pm.run(mod)
        cluster_info = nvidia.ClusterInfo()
        if opt.cluster_dims is not None:
            cluster_info.clusterDimX = opt.cluster_dims[0]
            cluster_info.clusterDimY = opt.cluster_dims[1]
            cluster_info.clusterDimZ = opt.cluster_dims[2]
        metadata["cluster_dims"] = (cluster_info.clusterDimX, cluster_info.clusterDimY, cluster_info.clusterDimZ)
        metadata["name"] = "mykernel"
        return XdslBaseBacked.convert_to_xdsl_ir(mod)


def register_xdsl_backend(name:str, backEnd, driver):
    backends[name] = Backend(backEnd,driver)
    triton.runtime.driver.set_active(driver())

