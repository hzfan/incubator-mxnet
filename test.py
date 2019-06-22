from __future__ import absolute_import
import numpy as _np
import mxnet as mx
from mxnet import np, npx
from mxnet.base import MXNetError
from mxnet.gluon import HybridBlock
from mxnet.test_utils import same, assert_almost_equal, rand_shape_nd, rand_ndarray
from mxnet.test_utils import check_numeric_gradient
from common import assertRaises, with_seed
import random

@with_seed()
@npx.use_np_shape
def test_np_vstack():
    class TestVstack(HybridBlock):
        def __init__(self):
            super(TestVstack, self).__init__()
        
        def hybrid_forward(self, F, a, *args):
            return F.np.vstack([a] + list(args))
    
    def g(data):
        return _np.ones_like(data)


    configs = [
        # ((), (), ()),
        # ((2), (2), (2)),
        # ((2), (0), (1)),
        ((2, 2), (3, 2), (0, 2)),
        ((2, 3), (1, 3), (4, 3)),
        ((2, 2, 2), (3, 2, 2), (1, 2, 2)),
        ((0, 1, 1), (4, 1, 1), (5, 1, 1))
    ]
    types = ['float16', 'float32', 'float64', 'int8', 'int32', 'int64']
    for config in configs:
        for hybridize in [True, False]:
            for dtype in types:
                test_vstack = TestVstack()
                if hybridize:
                    test_vstack.hybridize()
                print('config is {}'.format(config))
                rtol = 1e-3
                atol = 1e-5
                v = []
                v_np = []
                for i in range(3):
                    v_np.append(_np.array(_np.random.uniform(-10.0, 10.0, config[i]), dtype=dtype))
                    v.append(mx.nd.array(v_np[i]).as_np_ndarray())
                    v[i].attach_grad()
                expected_np = _np.vstack(v_np)
                with mx.autograd.record():
                    mx_out = test_vstack(*v)
                assert mx_out.shape == expected_np.shape
                assert_almost_equal(mx_out.asnumpy(), expected_np, rtol=rtol, atol=atol)

                # Test gradient
                mx_out.backward()
                for i in range(3):
                    expected_grad = g(v_np[i])
                    assert_almost_equal(v[i].grad.asnumpy(), expected_grad, rtol=rtol, atol=atol)

                # Test imperative once again
                mx_out = np.vstack(v)
                expected_np = _np.vstack(v_np)
                assert_almost_equal(mx_out.asnumpy(), expected_np, rtol=rtol, atol=atol)


if __name__ == '__main__':
    test_np_vstack()