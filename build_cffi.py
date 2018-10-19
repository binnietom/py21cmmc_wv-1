from cffi import FFI
import os

ffi = FFI()
LOCATION = os.path.dirname(os.path.abspath(__file__))
CLOC = os.path.join(LOCATION, 'py21cmmc_wv')
include_dirs = [CLOC]

# This is the overall C code.
ffi.set_source(
    "py21cmmc_wv.transforms",  # Name/Location of shared library module
    '''
    #include "transforms.c"
    ''',
    include_dirs = include_dirs,
    libraries=['m'],
)

# This is the Header file
with open(os.path.join(CLOC, "transforms.h")) as f:
    ffi.cdef(f.read())

if __name__ == "__main__":
    ffi.compile()
