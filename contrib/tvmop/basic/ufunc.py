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

# coding: utf-8
import tvm
from tvm import autotvm
from .. import defop, AllTypes
from .. import assign_by_req, reduce_axes

def compute_add(A, B):
    oshape = [tvm.var() for _ in range(len(A.shape))]
    C = tvm.compute(oshape,
                    lambda *index: A[index] + B[index], name='C')
    s = tvm.create_schedule(C.op)
    return s, C


@defop(name="vadd", target="cpu", auto_broadcast=True,
       dtype=AllTypes, ndim=[1, 2, 3, 4, 5])
def vadd(dtype, ndim):
    # cfg = autotvm.get_config()
    # cfg.define_knob("ifactor", [4] if fallback else [4])
    # ishape = [tvm.var("i{}".format(i)) for i in range(ndim - 1)] \
    #          + [tvm.var("i{}".format(ndim - 1)) if fallback \
    #          else cfg["ifactor"].val * tvm.var("i{}".format(ndim - 1))]
    ashape = [tvm.var() for _ in range(ndim)]
    bshape = [tvm.var() for _ in range(ndim)]
    A = tvm.placeholder(ashape, name='A', dtype=dtype)
    B = tvm.placeholder(bshape, name='B', dtype=dtype)
    s, C = compute_add(A, B)
    s[C].parallel(C.op.axis[0])
    # iaxis = C.op.axis[-1]
    # bx, tx = s[C].split(iaxis, factor=cfg["ifactor"].val)
    # s[C].vectorize(tx)
    return s, [A, B, C]


@defop(name="cuda_vadd", target="cuda", auto_broadcast=True,
       dtype=["float32", "float64"], ndim=[1, 2, 3, 4, 5])
def vadd_gpu(dtype, ndim):
    ashape = [tvm.var() for _ in range(ndim)]
    bshape = [tvm.var() for _ in range(ndim)]
    A = tvm.placeholder(ashape, name='A', dtype=dtype)
    B = tvm.placeholder(bshape, name='B', dtype=dtype)
    s, C = compute_add(A, B)
    axes = [axis for axis in C.op.axis]
    fused = s[C].fuse(*axes)
    bx, tx = s[C].split(fused, factor=256)
    s[C].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[C].bind(tx, tvm.thread_axis("threadIdx.x"))
    return s, [A, B, C]


def compute_backward_vadd(dtype, ndim, reduce1st, req):
    # The backward of broadcast op is basically a reduction on broadcast axes.
    # We label the reduce axes as 1 and other axes as 0, and they form a bit string.
    # Each bit string correponds to a kernel, so the number of kernels is as many as `2^n`
    # To reduce it, the bit string is compressed by combining consecutive 0s or 1s.
    # In this way, the number of bit string (the number of kernels) is reduced to `2 * n`
    # They compressed bit string is stored in `axes`. And `reduce1st` represents the first bit
    # of the compressed bit string. Credit to @junrushao1994 and @yzhliu.
    axes = ([reduce1st, 1 - reduce1st] * ndim)[:ndim]
    X = tvm.placeholder([tvm.var() for _ in range(ndim)], name='X', dtype=dtype)
    reducer = tvm.comm_reducer(lambda x, y: x + y,
        lambda t: tvm.const(0, dtype=t), name="sum")
    ret = reduce_axes(X, axes, reducer)
    old, new = assign_by_req(ret, req)
    s = tvm.create_schedule(new.op)
    return s, X, old, new, [ret, new]


@defop(name="backward_vadd", target="cpu", dtype=AllTypes, 
       ndim=[1, 2, 3, 4, 5], reduce1st=[0, 1],
       req=["kWriteTo", "kAddTo"], attrs=["reduce1st", "req"])
def backward_vadd(dtype, ndim, reduce1st, req):
    s, X, old, new, c_list = compute_backward_vadd(dtype, ndim, reduce1st, req)
    if len(new.op.axis) > 0:
        s[new].parallel(new.op.axis[0])
    return s, [X, old, new]


@defop(name="cuda_backward_vadd", target="gpu", dtype=["float32", "float64"],
       ndim=[1, 2, 3, 4 ,5], reduce1st=[0, 1],
       req=["kWriteTo", "kAddTo"], attrs=["reduce1st", "req"])
def backward_vadd_gpu(dtype, ndim, reduce1st, req):
    s, X, in_grad_a, in_grad, c_list = compute_backward_vadd(dtype, ndim, reduce1st, req)
    num_thread = 64
    for t in c_list:
        block_x = tvm.thread_axis("blockIdx.x")
        thread_x = tvm.thread_axis("threadIdx.x")
        axes = [axis for axis in t.op.axis]
        fused = s[t].fuse(*axes)
        bx, tx = s[t].split(fused, factor=num_thread)
        s[t].bind(bx, block_x)
        s[t].bind(tx, thread_x)
    return s, [X, in_grad_a, in_grad]
