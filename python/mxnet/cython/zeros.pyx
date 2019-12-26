# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License

from libc.stdint cimport int64_t
import ctypes

cdef extern from "mxnet/c_api.h":
    size_t _npi_zeros(size_t op_handle, size_t shape)
    size_t _npi_zeros_dummy(size_t op_handle, size_t shape)


cdef extern from "mxnet/tuple.h" namespace "mxnet":
    cdef cppclass Tuple[T]:
        inline T& operator[](int i)
        # inline const T& operator[](int i) const

    cdef cppclass TShape(Tuple[int64_t]):
        TShape()
        inline TShape(const int ndim, const int64_t value)


class Int64Array(ctypes.Structure):
    """Temp data structure for byte array."""
    _fields_ = [("data", ctypes.POINTER(ctypes.c_int64)),
                ("size", ctypes.c_size_t)]


def convert_tuple(src_tuple, temp_objs):
    arr = Int64Array()
    size = len(src_tuple)
    arr.data = ctypes.cast((ctypes.c_int64 * size)(*src_tuple),
                           ctypes.POINTER(ctypes.c_int64))
    arr.size = size
    temp_objs.append(arr)
    return <size_t>(ctypes.addressof(arr))


def _imperative_invoke_zeros(op_handle, shape):
    temp_objs = []
    address = convert_tuple(shape, temp_objs)
    out_ndarray_handle = _npi_zeros(op_handle, address)
    return out_ndarray_handle


def _imperative_invoke_zeros_dummy(op_handle, shape):
    temp_objs = []
    address = convert_tuple(shape, temp_objs)
    out_ndarray_handle = _npi_zeros_dummy(op_handle, address)
    return out_ndarray_handle
