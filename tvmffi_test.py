import mxnet as mx
import time
import ctypes
from mxnet.base import OpHandle, check_call, _LIB, c_str, NDArrayHandle
from mxnet._cy3.zeros import _imperative_invoke_zeros, _imperative_invoke_zeros_dummy

mx.npx.set_np()

name = '_npi_zeros'
zeros_op_handle = OpHandle()
check_call(_LIB.NNGetOpHandle(c_str(name), ctypes.byref(zeros_op_handle)))
# zeros_op_c_handle = Handle()
# zeros_op_c_handle.chandle = zeros_op_handle.value


def cython_zeros(shape):
    out_ndarray_handle = _imperative_invoke_zeros([zeros_op_handle, shape])
    return mx.np.ndarray(handle=ctypes.cast(out_ndarray_handle, NDArrayHandle))


def cython_zeros_dummy(shape):
    out_ndarray_handle = _imperative_invoke_zeros_dummy([zeros_op_handle, shape])

# out = cython_zeros((2, 2))
# print(out)
num_repeats = 10000
start = time.time()
for _ in range(num_repeats):
    # cython_zeros((2, 2))
    cython_zeros_dummy((2, 2))
elapse = time.time() - start

print('Time: {} us'.format(elapse * 1e6 / num_repeats))
