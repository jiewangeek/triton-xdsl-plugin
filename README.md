# A light-weight and pure python framework to add new triton backend and use pure python MLIR compiler (xDsl) 

Researchers and developers may want to add new backend to triton, for adapting new hardward or improving the current compiling pipeline. Common problems met can be:

* heavy-weight: need to fork the whole triton project, add or modify the MLIR related code, recompile the project 

* pure python: MLIR is processed in C++ and writing C++ is painful

This framework makes possible to:

* light-weight: no need to fork triton. You just need to define your own backend and driver, then register to triton. 

* Pure python: The framework intercepts the triton TTIR module. Then you can use MLIR python bingding or xDSL (a python-native MLIR compiler) to define your customized ir, make transformations and compilation. 


# Usage

see the example in ttpy/examples

```
git clone https://github.com/jiewangeek/triton-xdsl-plugin.git

cd triton-xdsl-plugin

python ttpy/examples/add.py
```

## What happens

* Define your backend

```
# ttpy/examples/mybackend/backend.py
class ExampleBackend(XdslBaseBackEnd):
   def add_stages(self, stages, options, language):
        # Invoking make_xdsl_ttir to get triton ttir or xDSL ttir
        # (xDsl ttir not works well yet)
        stages["ttir"] = lambda src, metadata: self.make_xdsl_ttir(src, metadata, options, False)

        # add customized stages
        stages["exampleir"] = lambda src, metadata: self.make_exampleir(src, metadata, options)

   ...


# ttpy/examples/mybackend/backend.py
class ExampleDriver(DriverBase):
   ...
```

* Registering to triton
```
import triton
from ttpy.examples.mybackend.driver import ExampleDriver
from ttpy.examples.mybackend.backend import ExampleBackend
from ttpy.triton_plugin.xdsl_base_backend import register_xdsl_backend

register_xdsl_backend('example', ExampleBackend, ExampleDriver)

@triton.jit
def your_kernel():
    ...


```



