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


cdef inline size_t convert_tuple(tuple src_tuple) except +:
    cdef size_t size = len(src_tuple)
    cdef TShape ret = TShape(size, 0)
    for i in range(size):
        ret[i] = <int>(src_tuple[i])
    return <size_t>(&ret)


def _imperative_invoke_zeros(op_handle, shape):
    out_ndarray_handle = _npi_zeros(op_handle, convert_tuple(shape))
    return out_ndarray_handle


def _imperative_invoke_zeros_dummy(op_handle, shape):
    out_ndarray_handle = _npi_zeros_dummy(op_handle, convert_tuple(shape))
    return out_ndarray_handle
