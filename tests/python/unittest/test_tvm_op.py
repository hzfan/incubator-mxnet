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

import itertools
import mxnet as mx
import numpy as _np
from mxnet import np
from mxnet.test_utils import same, rand_shape_nd, assert_almost_equal, use_np
from mxnet.runtime import Features
from mxnet.gluon import HybridBlock
from common import with_seed

_features = Features()

@with_seed()
def test_tvm_broadcast_add():
    if _features.is_enabled("TVM_OP"):
        a_shape = rand_shape_nd(4)
        b_shape = (1,) + a_shape[1:2] + (1, 1)
        a = mx.nd.normal(shape=a_shape)
        b = mx.nd.normal(shape=b_shape)
        c = mx.nd.contrib.tvm_vadd(a, b)
        c_np = a.asnumpy() + b.asnumpy()
        assert same(c.asnumpy(), c_np)


@with_seed()
@use_np
def test_tvm_vcumprod():
    if _features.is_enabled("TVM_OP"):
        class TestCumprod(HybridBlock):
            def __init__(self, axis):
                super(TestCumprod, self).__init__()
                self.axis = axis

            def hybrid_forward(self, F, data):
                return F.np.tvm_vcumprod(data, axis=self.axis)
        
        def get_grad(data, axis):
            isNone = False
            print(" - flatten...")
            if axis == None:
                isNone = True
                ishape = data.shape
                data = _np.reshape(data, (-1,))
                axis = 0
            data = _np.swapaxes(data, -1, axis)
            jacobi = _np.empty(data.shape + (data.shape[-1],), dtype=data.dtype)
            print(" - product...")
            r = [range(ndim) for ndim in jacobi.shape]
            r = itertools.product(*r)
            print(" - iter...")
            for idx in r:
                j = idx[-2]
                i = idx[-1]
                if i < j:
                    jacobi[idx] = 0
                else:
                    jacobi[idx] = 1
                    for k in range(i + 1):
                        if k != j:
                            jacobi[idx] *= data[idx[:-2] + (k,)]
            print(" - sum...")
            grad = _np.sum(jacobi, axis=-1)
            print(" - swapaxes...")
            grad = _np.swapaxes(grad, -1, axis)
            if isNone:
                grad = _np.reshape(grad, ishape)
            return grad

        dtypes = ['int32', 'float32', 'float64']
        for hybridize in [False, True]:
            for dtype in dtypes:
                for ndim in range(0, 6):
                    axes = list(range(ndim)) + [None]
                    for axis in axes:
                        print("===================================")
                        print("hybridize = {}".format(hybridize))
                        print("dtype = {}".format(dtype))
                        print("ndim = {}".format(ndim))
                        print("axis = {}".format(axis))
                        rtol = 1e-2
                        atol = 1e-2
                        test_cumprod = TestCumprod(axis)
                        if hybridize:
                            test_cumprod.hybridize()
                        dim = 3 if axis is None else 10
                        data_shape = rand_shape_nd(ndim, dim=dim, allow_zero_size=True)
                        print("data_shape = {}".format(data_shape))
                        data_np = _np.array(_np.random.uniform(-1.0, 1.0, data_shape), dtype=dtype)
                        data = np.array(data_np, dtype=dtype)
                        data.attach_grad()
                        expected_np = _np.cumprod(data_np, axis=axis)
                        print("forward...")
                        with mx.autograd.record():
                            out_mx = test_cumprod(data)
                        assert out_mx.shape == expected_np.shape
                        assert_almost_equal(out_mx.asnumpy(), expected_np, rtol=rtol, atol=atol)
                        print("backward...")
                        print(" - mxnet backward...")
                        out_mx.backward()
                        print(" - numpy backward...")
                        backward_expected = get_grad(data_np, axis=axis)
                        assert_almost_equal(data.grad.asnumpy(), backward_expected, rtol=rtol, atol=atol)

                        # Test imperative once again
                        print("imperative...")
                        data = np.array(data_np, dtype=dtype)
                        out_mx = np.tvm_vcumprod(data, axis=axis)
                        assert_almost_equal(out_mx.asnumpy(), expected_np, rtol=rtol, atol=atol)

                        # Test AddTo Request
                        print("addto request...")
                        data = np.array(data_np, dtype=dtype)
                        data.attach_grad()
                        with mx.autograd.record():
                            a = test_cumprod(data)
                            b = test_cumprod(data)
                        mx.autograd.backward([a, b])
                        backward_expected = 2 * get_grad(data_np, axis=axis)
                        assert_almost_equal(data.grad.asnumpy(), backward_expected, rtol=rtol, atol=atol)


if __name__ == '__main__':
    import nose
    nose.runmodule()
