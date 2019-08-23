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
import mxnet as mx
from mxnet import np, npx

def measure_cost(repeat, func_name, *args, **kwargs):
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


def test_add():
    # tvm add
    n = 128
    m = 128
    k = 128
    print("tvm 1024 add:")
    a = mx.nd.random.uniform(shape=(n, m, k), dtype='float32')
    b = mx.nd.random.uniform(shape=(n, m, k), dtype='float32')
    cost = measure_cost(50, mx.nd.contrib.tvm_vadd, a, b)
    print("cost: {} ms".format(cost * 1000))
    print("tvm 1024 add:")
    a = mx.nd.random.uniform(shape=(n, m, k), dtype='float32')
    b = mx.nd.random.uniform(shape=(n, m, k), dtype='float32')
    cost = measure_cost(50, mx.nd.contrib.tvm_vadd, a, b)
    print("cost: {} ms".format(cost * 1000))
    print("tvm 1024 add_1024:")
    a = mx.nd.random.uniform(shape=(n, m, k), dtype='float32')
    b = mx.nd.random.uniform(shape=(n, m, k), dtype='float32')
    cost = measure_cost(50, mx.nd.contrib.tvm_vadd_1024, a, b)
    print("cost: {} ms".format(cost * 1000))
    print("tvm 1023 add:")
    n = 128
    m = 128
    k = 128
    a = mx.nd.random.uniform(shape=(n, m, k), dtype='float32')
    b = mx.nd.random.uniform(shape=(n, m, k), dtype='float32')
    cost = measure_cost(50, mx.nd.contrib.tvm_vadd, a, b)
    print("cost: {} ms".format(cost * 1000))
    print("tvm 1025 add:")
    n = 128
    m = 128
    k = 128
    a = mx.nd.random.uniform(shape=(n, m, k), dtype='float32')
    b = mx.nd.random.uniform(shape=(n, m, k), dtype='float32')
    cost = measure_cost(50, mx.nd.contrib.tvm_vadd, a, b)
    print("cost: {} ms".format(cost * 1000))
    # np add
    # print("np add:")
    # a = mx.nd.random.uniform(shape=(n, m), dtype='float32')
    # b = np.array(mx.nd.random.uniform(shape=(n, m)), dtype='float32')
    # print("generate done")
    # cost = measure_cost(1, np.add, a, b)
    # print("cost: {} ms".format(cost * 1000))


if __name__ == "__main__":
    npx.set_np()
    test_add() 