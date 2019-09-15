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
	
import time
import timeit
import mxnet as mx
import numpy as _np
from mxnet import np, npx

def measure_mx_cost(repeat, func_name, *args, **kwargs):
    """Measure time cost of running a function
    """
    mx.nd.waitall()
    start = time.time()
    for _ in range(repeat):
        func_name(*args, **kwargs)
    mx.nd.waitall()
    end = time.time()
    diff = end - start
    return diff / repeat


def measure_np_cost(repeat, func_name, *args, **kwargs):
    start = time.time()
    for _ in range(repeat):
        func_name(*args, **kwargs)
    end = time.time()
    diff = end - start
    return diff / repeat


def stabalize(x, nremoved):
    # warm up
    x = x[1:]
    # remove the min and max
    # for _ in range(nremoved):
    #     x = _np.delete(x, _np.amax(x))
    # for _ in range(nremoved):
    #     x = _np.delete(x, _np.amin(x))
    return x


def test_add():
    shapes = [32, 64, 128, 256, 512]
    repeat = 1
    times = 100
    nremoved = 2
    for size in shapes:
        n = size
        m = size
        k = size
        dtype = 'float32'
        a_np = _np.array(_np.random.uniform(-2.0, 2.0, size=(n, m, k)), dtype=dtype)
        b_np = _np.array(_np.random.uniform(-2.0, 2.0, size=(n, m, k)), dtype=dtype)

        cost_tvm = []
        cost_mx = []
        cost_np = []
        print("===========================size = {}======================================".format(size))
        for i in range(times):
            a = mx.nd.array(a_np, dtype=dtype)
            b = mx.nd.array(b_np, dtype=dtype)
            cost = measure_mx_cost(repeat, mx.nd.contrib.tvm_vadd, a, b)
            cost_tvm.append(cost)

        for i in range(times):
            a = np.array(a_np, dtype=dtype)
            b = np.array(b_np, dtype=dtype)
            cost = measure_mx_cost(repeat, np.add, a, b)
            cost_mx.append(cost)

        for i in range(times):
            a = a_np
            b = b_np
            cost = measure_np_cost(repeat, _np.add, a, b)
            cost_np.append(cost)

        cost_tvm = stabalize(cost_tvm, nremoved)
        cost_mx = stabalize(cost_mx, nremoved)
        cost_np = stabalize(cost_np, nremoved)
        print("tvm:")
        print("mean: {}".format(_np.mean(cost_tvm)))
        print("std:  {}".format(_np.std(cost_tvm) / _np.mean(cost_tvm)))
        print("mx:")
        print("mean: {}".format(_np.mean(cost_mx)))
        print("std:  {}".format(_np.std(cost_mx) / _np.mean(cost_mx)))
        print("np:")
        print("mean: {}".format(_np.mean(cost_np)))
        print("std:  {}".format(_np.std(cost_np) / _np.mean(cost_np)))


if __name__ == "__main__":
    npx.set_np()
    test_add() 