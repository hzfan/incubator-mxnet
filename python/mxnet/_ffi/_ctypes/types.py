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
# under the License.
"""The C Types used in API."""
# pylint: disable=invalid-name
from __future__ import absolute_import as _abs

import ctypes
import struct
from ..base import check_call, _LIB
from ..runtime_ctypes import TypeCode
from ...base import NDArrayHandle
from ...numpy import ndarray

class TypeCode(object):
    """Type code used in API calls"""
    INT = 0
    UINT = 1
    FLOAT = 2
    HANDLE = 3
    NULL = 4
    TVM_TYPE = 5
    TVM_CONTEXT = 6
    ARRAY_HANDLE = 7
    OBJECT_HANDLE = 8
    MODULE_HANDLE = 9
    FUNC_HANDLE = 10
    STR = 11
    BYTES = 12
    NDARRAY_CONTAINER = 13
    NDARRAYHANDLE = 14
    EXT_BEGIN = 15


class MXNetValue(ctypes.Union):
    """MXNetValue in C API"""
    _fields_ = [("v_int64", ctypes.c_int64),
                ("v_float64", ctypes.c_double),
                ("v_handle", ctypes.c_void_p),
                ("v_str", ctypes.c_char_p)]

RETURN_SWITCH = {
    TypeCode.INT: lambda x: x.v_int64,
    TypeCode.FLOAT: lambda x: x.v_float64,
    TypeCode.NULL: lambda x: None,
    TypeCode.NDARRAYHANDLE: lambda x: ndarray(handle=NDArrayHandle(x.v_handle))
}
