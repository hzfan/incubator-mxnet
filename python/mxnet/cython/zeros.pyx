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

import ctypes
from libc.stdint cimport int64_t

# cdef class Handle:
#     cdef size_t chandle
#     property chandle:
#         def __set__(self, value):
#             if value == None:
#                 self.chandle = <size_t>NULL
#             else:
#                 self.chandle = <size_t>value


cdef extern from "mxnet/c_api_runtime.h":
    ctypedef enum TypeCode:
        kInt = 0,
        kUInt = 1,
        kFloat = 2,
        kHandle = 3,
        kNull = 4
    ctypedef union Value:
        int64_t v_int64
        double v_float64
        size_t v_handle
        const char* v_str
    size_t _npi_zeros(Value* arg_values, TypeCode* type_codes, int num_args)
    size_t _npi_zeros_dummy(Value* arg_values, TypeCode* type_codes, int num_args)
    ctypedef struct Int64Array:
        int64_t* data
        size_t size
    # cdef cppclass ArrayBuilder[T]:
    #     ArrayBuilder()
    #     inline ArrayBuilder(size_t size)
    #     inline T& operator[](int i)
    #     inline void resize(size_t size)
    cdef cppclass Int64ArrayCtor:
        inline void operator() (Int64Array& obj, size_t size)
    cdef cppclass Int64ArrayWrapper:
        inline Int64ArrayWrapper() except +
        inline Int64ArrayWrapper(size_t size) except +
        inline Int64ArrayWrapper(Int64ArrayWrapper&& other) except +
        inline Int64ArrayWrapper&& move()
        Int64Array obj
    # cdef cppclass Int64ArrayPtr:
    #     Int64ArrayPtr()
    #     inline Int64ArrayPtr(const ArrayBuilder[int64_t]& arr)
    #     inline Int64Array* get()
    #     inline void reset(const ArrayBuilder[int64_t]& arr)


cdef extern from "mxnet/tuple.h" namespace "mxnet":
    cdef cppclass Tuple[T]:
        inline T& operator[](int i)
        # inline const T& operator[](int i) const

    cdef cppclass TShape(Tuple[int64_t]):
        TShape()
        inline TShape(const int ndim, const int64_t value)


cdef extern from "dmlc/any.h" namespace "dmlc":
    inline T& get[T](any& src)
    cdef cppclass any:
        inline any& operator=[T](T&& other)


# cdef inline void convert_tuple(tuple src_tuple,
#                                ArrayBuilder[int64_t]* arr,
#                                Int64ArrayPtr* arrp) except +:
#     cdef size_t size = len(src_tuple)
#     arr[0].resize(size)
#     for i in range(size):
#         arr[0][i] = src_tuple[i]
#     arrp[0].reset(arr[0])


cdef inline size_t c_handle(object handle):
    """Cast C types handle to c handle."""
    cdef unsigned long long v_ptr
    v_ptr = handle.value
    return <size_t>(v_ptr)


cdef inline void convert_to_tuple(tuple x, any* temp_args, size_t* value):
    cdef size_t size = len(x)
    cdef Int64ArrayWrapper arr
    cdef Int64ArrayCtor func
    func(arr.obj, size)
    for i in range(size):
        arr.obj.data[i] = x[i]
    temp_args[0] = arr.move()
    value[0] = <size_t>&(get[Int64ArrayWrapper](temp_args[0]).obj)


cdef inline int make_arg(object arg, Value* value, TypeCode* tcode, any* temp_objs) except -1:
    if isinstance(arg, ctypes.c_void_p):
        value[0].v_handle = c_handle(arg)
        tcode[0] = kHandle
    # elif isinstance(arg, Handle):
    #     value[0].v_handle = (<Handle>arg).chandle
    #     tcode[0] = kHandle
    elif isinstance(arg, tuple):
        convert_to_tuple(arg, temp_objs, &value[0].v_handle)
        tcode[0] = kHandle
    else:
        raise TypeError("Don't know how to handle type %s" % type(arg))


def _imperative_invoke_zeros(args):
    cdef any[2] temp_obj
    cdef Value[2] values
    cdef TypeCode[2] tcodes
    for i in range(2):
        make_arg(args[i], &values[i], &tcodes[i], &temp_obj[i])
    out_ndarray_handle = _npi_zeros(values, tcodes, 2)
    return out_ndarray_handle


def _imperative_invoke_zeros_dummy(args):
    cdef any[2] temp_obj
    cdef Value[2] values
    cdef TypeCode[2] tcodes
    for i in range(2):
        make_arg(args[i], &values[i], &tcodes[i], &temp_obj[i])
    out_ndarray_handle = _npi_zeros_dummy(values, tcodes, 2)
    return out_ndarray_handle
