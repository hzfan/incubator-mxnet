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
    ctypedef struct Int64Array:
        int64_t* data
        size_t size


cdef extern from "mxnet/c_api_runtime.h":
    cdef cppclass ArrayBuilder[T]:
        ArrayBuilder()
        inline ArrayBuilder(size_t size)
        inline T& operator[](int i)
        inline void resize(size_t size)
    cdef cppclass Int64ArrayPtr:
        Int64ArrayPtr()
        inline Int64ArrayPtr(const ArrayBuilder[int64_t]& arr)
        inline Int64Array* get()
        inline void reset(const ArrayBuilder[int64_t]& arr)


cdef extern from "mxnet/tuple.h" namespace "mxnet":
    cdef cppclass Tuple[T]:
        inline T& operator[](int i)
        # inline const T& operator[](int i) const

    cdef cppclass TShape(Tuple[int64_t]):
        TShape()
        inline TShape(const int ndim, const int64_t value)


cdef extern from "<utility>" namespace "std":
    T move[T](T t)


cdef inline void convert_tuple(tuple src_tuple,
                               ArrayBuilder[int64_t]* arr,
                               Int64ArrayPtr* arrp) except +:
    cdef size_t size = len(src_tuple)
    arr[0].resize(size)
    for i in range(size):
        arr[0][i] = src_tuple[i]
    arrp[0].reset(arr[0])


def _imperative_invoke_zeros(op_handle, shape):
    cdef ArrayBuilder[int64_t] temp_obj1
    cdef Int64ArrayPtr temp_obj2
    convert_tuple(shape, &temp_obj1, &temp_obj2)
    out_ndarray_handle = _npi_zeros(op_handle, <size_t>temp_obj2.get())
    return out_ndarray_handle


def _imperative_invoke_zeros_dummy(op_handle, shape):
    cdef ArrayBuilder[int64_t] temp_obj1
    cdef Int64ArrayPtr temp_obj2
    convert_tuple(shape, &temp_obj1, &temp_obj2)
    out_ndarray_handle = _npi_zeros_dummy(op_handle, <size_t>temp_obj2.get())
    return out_ndarray_handle
