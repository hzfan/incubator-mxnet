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

import tvm
from .. import defop, AllTypes
from .. import assign_by_req, reduce_axes

_bin_logic_op_map = {
    'equal': lambda a, b, *idx: a[idx] == b[idx],
    'not_equal': lambda a, b, *idx: a[idx] != b[idx],
    'greater': lambda a, b, *idx: a[idx] > b[idx],
    'less': lambda a, b, *idx: a[idx] < b[idx],
    'greater_equal': lambda a, b, *idx: a[idx] >= b[idx],
    'less_equal': lambda a, b, *idx: a[idx] <= b[idx],
}


def _compute_binary_logic(op, dtype, ndim):
    a = tvm.placeholder([tvm.var() for _ in range(ndim)], dtype=dtype, name='a')
    b = tvm.placeholder([tvm.var() for _ in range(ndim)], dtype=dtype, name='b')
    c = tvm.compute([tvm.var() for _ in range(ndim)],
                    lambda *idx: _bin_logic_op_map[op](a, b, *idx), name='c')
    s = tvm.create_schedule(c.op)
    return s, a, b, c


_bin_logic_cpu_attrs = {
    'compute_func': _compute_binary_logic,
    'target': 'cpu',
    'auto_broadcast': True,
    'itype': AllTypes + ['bool'],
    'ndim': list(range(6))
}

_bin_logic_gpu_attrs = {
    'compute_func': _compute_binary_logic,
    'target': 'gpu',
    'auto_broadcast': True,
    'itype': AllTypes + ['bool'],
    'ndim': list(range(6))
}


def _binary_logic_cpu(compute_func, op, itype, ndim):
    s, a, b, c = compute_func(op, itype, ndim)
    axes = [axis for axis in c.op.axis]
    fused = s[c].fuse(*axes)
    s[c].parallel(fused)
    return s, [a, b, c]


def _binary_logic_gpu(compute_func, op, itype, ndim):
    s, a, b, c = compute_func(op, itype, ndim)
    axes = [axis for axis in c.op.axis]
    fused = s[c].fuse(*axes)
    bx, tx = s[c].split(fused, factor=64)
    s[c].bind(bx, tvm.thread_axis('blockIdx.x'))
    s[c].bind(tx, tvm.thread_axis('threadIdx.x'))
    return s, [a, b, c]


# register binary element-wise logic ops with broadcasting supported
for op_name in _bin_logic_op_map.keys():
    defop(name='{}_cpu'.format(op_name), op=op_name, **_bin_logic_cpu_attrs)(_binary_logic_cpu)
    defop(name='{}_gpu'.format(op_name), op=op_name, **_bin_logic_gpu_attrs)(_binary_logic_gpu)


# Note that `b.dtype` is hard-coded as 'float64'.
# We should always promote `a`'s elements to `b.dtype`.
_bin_scalar_logic_op_map = {
    'equal_scalar': lambda a, b, *idx: a[idx].astype(b.dtype) == b,
    'not_equal_scalar': lambda a, b, *idx: a[idx].astype(b.dtype) != b,
    'greater_scalar': lambda a, b, *idx: a[idx].astype(b.dtype) > b,
    'less_scalar': lambda a, b, *idx: a[idx].astype(b.dtype) < b,
    'greater_equal_scalar': lambda a, b, *idx: a[idx].astype(b.dtype) >= b,
    'less_equal_scalar': lambda a, b, *idx: a[idx].astype(b.dtype) <= b,
}


def _compute_binary_scalar_logic(op, dtype, ndim):
    a = tvm.placeholder([tvm.var() for _ in range(ndim)], name='a', dtype=dtype)
    b = tvm.var('b', dtype='float64')
    c = tvm.compute([tvm.var() for _ in range(ndim)],
                    lambda *idx: _bin_scalar_logic_op_map[op](a, b, *idx), name='c')
    s = tvm.create_schedule(c.op)
    return s, a, b, c


_bin_scalar_logic_cpu_attrs = {
    'compute_func': _compute_binary_scalar_logic,
    'target': 'cpu',
    'itype': AllTypes + ['bool'],
    'ndim': list(range(6))
}

_bin_scalar_logic_gpu_attrs = {
    'compute_func': _compute_binary_scalar_logic,
    'target': 'gpu',
    'itype': AllTypes + ['bool'],
    'ndim': list(range(6))
}


# register binary element-wise scalar logic ops
for op_name in _bin_scalar_logic_op_map.keys():
    defop(name='{}_cpu'.format(op_name), op=op_name,
          **_bin_scalar_logic_cpu_attrs)(_binary_logic_cpu)
    defop(name='{}_gpu'.format(op_name), op=op_name,
          **_bin_scalar_logic_gpu_attrs)(_binary_logic_gpu)


_bin_op_map = {
    'multiply': lambda a, b, *idx: a[idx] * b[idx],
}

def _compute_binary(op, dtype, ndim):
    a = tvm.placeholder([tvm.var() for _ in range(ndim)], dtype=dtype, name='a')
    b = tvm.placeholder([tvm.var() for _ in range(ndim)], dtype=dtype, name='b')
    c = tvm.compute([tvm.var() for _ in range(ndim)],
                    lambda *idx: _bin_op_map[op](a, b, *idx), name='c')
    s = tvm.create_schedule(c.op)
    return s, a, b, c

_bin_cpu_attrs = {
    'compute_func': _compute_binary,
    'target': 'cpu',
    'auto_broadcast': True,
    'itype': AllTypes,
    'ndim': list(range(6))
}

_bin_gpu_attrs = {
    'compute_func': _compute_binary,
    'target': 'gpu',
    'auto_broadcast': True,
    'itype': ["float32", "float64", "uint8", "int8", "int32", "int64"],
    'ndim': list(range(6))
}

def _binary_cpu(compute_func, op, itype, ndim):
    s, a, b, c = compute_func(op, itype, ndim)
    axes = [axis for axis in c.op.axis]
    fused = s[c].fuse(*axes)
    s[c].parallel(fused)
    return s, [a, b, c]

def _binary_gpu(compute_func, op, itype, ndim):
    s, a, b, c = compute_func(op, itype, ndim)
    axes = [axis for axis in c.op.axis]
    fused = s[c].fuse(*axes)
    bx, tx = s[c].split(fused, factor=64)
    s[c].bind(bx, tvm.thread_axis('blockIdx.x'))
    s[c].bind(tx, tvm.thread_axis('threadIdx.x'))
    return s, [a, b, c]

for op_name in _bin_op_map.keys():
    defop(name='{}_cpu'.format(op_name), op=op_name, **_bin_cpu_attrs)(_binary_cpu)
    defop(name='{}_gpu'.format(op_name), op=op_name, **_bin_gpu_attrs)(_binary_gpu)

_bin_backward_op_map = {
    'backward_multiply': (lambda a, b, *idx: b[idx],
                          lambda a, b, *idx: b[idx]),
}

def _compute_binary_backward(op, dtype, ndim, reduce1st, req):
    axes = ([reduce1st, 1 - reduce1st] * ndim)[:ndim]
    oshape = [tvm.var() for _ in range(ndim)]
    ograd = tvm.placeholder(oshape, name='X', dtype=dtype)
    a = tvm.placeholder([tvm.var() for _ in range(ndim)], dtype=dtype, name='a')
    b = tvm.placeholder([tvm.var() for _ in range(ndim)], dtype=dtype, name='b')
    grad = tvm.compute(oshape, lambda *idx: op(a, b, *idx) * ograd[idx])
    ret = reduce_axes(grad, axes, tvm.sum)
    igrad_old, igrad = assign_by_req(ret, req)
    s = tvm.create_schedule(igrad.op)
    s[grad].compute_inline()
    return s, ograd, a, b, igrad_old, igrad, [ret, igrad]

_bin_backward_cpu_attrs = {
    'compute_func': _compute_binary_backward,
    'itype': AllTypes,
    'ndim': [9],
    'output': [0, 1],
    'reduce1st': [0, 1],
    'req': ['kWriteTo', 'kAddTo'],
    'attrs': ["output", "reduce1st", "req"],
    'target': 'cpu',
    'auto_broadcast': True,
}

_bin_backward_gpu_attrs = {
    'compute_func': _compute_binary_backward,
    'itype': ["float32", "float64", "uint8", "int8", "int32", "int64"],
    'ndim': [9],
    'output': [0, 1],
    'reduce1st': [0, 1],
    'req': ['kWriteTo', 'kAddTo'],
    'attrs': ["output", "reduce1st", "req"],
    'target': 'gpu',
    'auto_broadcast': True,
}

def _binary_backward_cpu(compute_func, op, itype, ndim, output, reduce1st, req):
    op = _bin_backward_op_map[op][output]
    s, ograd, a, b, igrad_old, igrad, c_list = compute_func(op, itype, ndim, reduce1st, req)
    for t in c_list:
        axes = [axis for axis in t.op.axis]
        fused = s[t].fuse(*axes)
        s[t].parallel(fused)
    return s, [ograd, a, b, igrad_old, igrad]


def _binary_backward_gpu(compute_func, op, itype, ndim, output, reduce1st, req):
    op = _bin_backward_op_map[op][output]
    s, ograd, a, b, igrad_old, igrad, c_list = compute_func(op, itype, ndim, reduce1st, req)
    num_thread = 64
    for t in c_list:
        block_x = tvm.thread_axis("blockIdx.x")
        thread_x = tvm.thread_axis("threadIdx.x")
        axes = [axis for axis in t.op.axis]
        fused = s[t].fuse(*axes)
        bx, tx = s[t].split(fused, factor=num_thread)
        s[t].bind(bx, block_x)
        s[t].bind(tx, thread_x)
    return s, [ograd, a, b, igrad_old, igrad]

# register binary element-wise ops backward with broadcasting supported
for op_name in _bin_backward_op_map.keys():
    defop(name='{}_cpu'.format(op_name), op=op_name, **_bin_backward_cpu_attrs)(_binary_backward_cpu)
    defop(name='{}_gpu'.format(op_name), op=op_name, **_bin_backward_gpu_attrs)(_binary_backward_gpu)
