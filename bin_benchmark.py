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
import jax
from mxnet import np, npx


def measure_jax_cost(repeat, func_name, *args, **kwargs):
    assert repeat == 1
    start = time.time()
    for _ in range(repeat):
        func_name(*args, **kwargs).block_until_ready()
    end = time.time()
    diff = end - start
    return diff / repeat


def measure_jax_backward(repeat, func_name, *args, **kwargs):
    def get_grad(*args):
        return jax.np.sum(func_name(*args))

    assert repeat == 1
    grad_func = jax.jit(jax.grad(get_grad, (0, 1)))
    start = time.time()
    for _ in range(repeat):
        ret = grad_func(*args, **kwargs)
        for i in ret:
            i.block_until_ready()
    end = time.time()
    diff = end - start
    return diff / repeat
    

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


def measure_mx_backward(repeat, func_name, *args, **kwargs):
    assert repeat == 1
    for arg in args:
        arg.attach_grad()
    with mx.autograd.record():
        mx_out = func_name(*args, **kwargs)
    mx.nd.waitall()
    start = time.time()
    mx_out.backward()
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


def stabalize(x):
    # warm up
    x = x[1:]
    return x


def test_add():
    configs = [
        ((32, 32, 32), (32, 32, 32)),
        ((64, 64, 64), (64, 64, 64)),
        ((128, 128, 128), (128, 128, 128)),
        ((256, 256, 256), (256, 256, 256)),
        ((32, 1, 32), (32, 32, 32)),
        ((64, 1, 64), (64, 64, 64)),
        ((128, 1, 128), (128, 128, 128)),
        ((256, 1, 256), (256, 256, 256)),
    ]
    forward_repeat = 1
    backward_repeat = 1
    times = 100
    nremoved = 0
    enable_gpu = False
    ctx = mx.gpu(0) if enable_gpu else mx.cpu()
    for config in configs:
        dtype = 'float32'
        dsize = 4
        a_np = _np.array(_np.random.uniform(-2.0, 2.0, size=config[0]), dtype=dtype)
        b_np = _np.array(_np.random.uniform(-2.0, 2.0, size=config[1]), dtype=dtype)
        c_np = _np.add(a_np, b_np)

        cost_tvm = []
        cost_mx = []
        cost_np = []
        cost_jax = []
        cost_tvm_backward = []
        cost_mx_backward = []
        cost_jax_backward = []
        print("================================================================================")
        print("config: {}".format(config))
        for i in range(times):
            a = mx.nd.array(a_np, dtype=dtype, ctx=ctx)
            b = mx.nd.array(b_np, dtype=dtype, ctx=ctx)
            cost = measure_mx_cost(forward_repeat, mx.nd.contrib.tvm_vadd, a, b)
            cost_tvm.append(cost)

        for i in range(times):
            a = np.array(a_np, dtype=dtype, ctx=ctx)
            b = np.array(b_np, dtype=dtype, ctx=ctx)
            cost = measure_mx_cost(forward_repeat, np.add, a, b)
            cost_mx.append(cost)

        for i in range(times):
            a = a_np
            b = b_np
            cost = measure_np_cost(forward_repeat, _np.add, a, b)
            cost_np.append(cost)
        
        for i in range(times):
            a = jax.np.array(a_np).block_until_ready()
            b = jax.np.array(b_np).block_until_ready()
            func = jax.jit(jax.np.add)
            cost = measure_jax_cost(forward_repeat, func, a, b)
            cost_jax.append(cost)

        for i in range(times):
            a = mx.nd.array(a_np, dtype=dtype, ctx=ctx)
            b = mx.nd.array(b_np, dtype=dtype, ctx=ctx)
            cost = measure_mx_backward(backward_repeat, mx.nd.contrib.tvm_vadd, a, b)
            cost_tvm_backward.append(cost)
        
        for i in range(times):
            a = np.array(a_np, dtype=dtype, ctx=ctx)
            b = np.array(b_np, dtype=dtype, ctx=ctx)
            cost = measure_mx_backward(backward_repeat, np.add, a, b)
            cost_mx_backward.append(cost)   
        
        for i in range(times):
            a = jax.np.array(a_np, dtype=dtype).block_until_ready()
            b = jax.np.array(b_np, dtype=dtype).block_until_ready()
            cost = measure_jax_backward(backward_repeat, jax.np.add, a, b)
            cost_jax_backward.append(cost)

        cost_tvm = stabalize(cost_tvm)
        cost_mx = stabalize(cost_mx)
        cost_np = stabalize(cost_np)
        cost_jax = stabalize(cost_jax)
        cost_tvm_backward = stabalize(cost_tvm_backward)
        cost_mx_backward = stabalize(cost_mx_backward)
        cost_jax_backward = stabalize(cost_jax_backward)
        print("tvm:")
        mean = _np.mean(cost_tvm)
        print("mean(s):         {}".format(mean))
        print("std/mean:        {}".format(_np.std(cost_tvm) / mean))
        print("bandwidth(GBps): {}".format((a_np.size + b_np.size + c_np.size) * dsize/ mean / 2 ** 30))
        print("mx:")
        mean = _np.mean(cost_mx)
        print("mean(s):         {}".format(mean))
        print("std/mean:        {}".format(_np.std(cost_mx) / mean))
        print("bandwidth(GBps): {}".format((a_np.size + b_np.size + c_np.size) * dsize/ mean / 2 ** 30))
        print("np:")
        mean = _np.mean(cost_np)
        print("mean(s):         {}".format(mean))
        print("std/mean:        {}".format(_np.std(cost_np) / mean))
        print("bandwidth(GBps): {}".format((a_np.size + b_np.size + c_np.size) * dsize/ mean / 2 ** 30))
        print("jax:")
        mean = _np.mean(cost_jax)
        print("mean(s):         {}".format(mean))
        print("std/mean:        {}".format(_np.std(cost_jax) / mean))
        print("bandwidth(GBps): {}".format((a_np.size + b_np.size + c_np.size) * dsize/ mean / 2 ** 30))
        print("tvm_backward:")
        mean = _np.mean(cost_tvm_backward)
        print("mean(s):         {}".format(mean))
        print("std/mean:        {}".format(_np.std(cost_tvm_backward) / mean))
        print("bandwidth(GBps): {}".format((c_np.size + c_np.size) * dsize/ mean / 2 ** 30))
        print("mx_backward:")
        mean = _np.mean(cost_mx_backward)
        print("mean(s):         {}".format(mean))
        print("std/mean:        {}".format(_np.std(cost_mx_backward) / mean))
        print("bandwidth(GBps): {}".format((c_np.size + c_np.size) * dsize/ mean / 2 ** 30))
        print("jax_backward:")
        mean = _np.mean(cost_jax_backward)
        print("mean(s):         {}".format(mean))
        print("std/mean:        {}".format(_np.std(cost_jax_backward) / mean))
        print("bandwidth(GBps): {}".format((c_np.size + c_np.size) * dsize/ mean / 2 ** 30))


if __name__ == "__main__":
    npx.set_np()
    test_add() 