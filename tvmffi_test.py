import os
os.environ["MXNET_ENGINE_TYPE"] = "NaiveEngine"

import mxnet as mx
import time
from mxnet.ndarray import np

# print("tvm ffi...")
# a = np.zeros1((3, 4), ctx="cpu(0)", dtype='float64')
# print(a)
# print("legacy ffi...")
# a = np.zeros((3, 4), ctx=mx.cpu())
# print(a)

# print("tvm ffi dummy...")
# repeat = 10000
# start = time.time()
# for i in range(repeat):
#     a = np.zeros0((3, 4), ctx="cpu(0)", dtype='float64')
# end = time.time()
# print("time = {}".format((end - start) / repeat))

# print("tvm ffi...")
# repeat = 10000
# start = time.time()
# for i in range(repeat):
#     a = np.zeros1((3, 4), ctx="cpu(0)", dtype='float64')
# end = time.time()
# print("time = {}".format((end - start) / repeat))

print("scalar axis...")
a = np.ones((2, 2))
b = np.ones((2, 2))
c = np.tensordot1(a, b)
print(c)

print("tuple axes...")
a = np.ones((2, 2))
b = np.ones((2, 2))
c = np.tensordot1(a, b, (1, 0))
print(c)

print("tvm ffi...")
repeat = 10000
a = np.ones((2, 2))
b = np.ones((2, 2))
start = time.time()
for i in range(repeat):
    c = np.tensordot1(a, b, (1, 0))
end = time.time()
print("time = {}".format((end - start) / repeat))

print("legacy ffi...")
repeat = 10000
a = np.ones((2, 2))
b = np.ones((2, 2))
start = time.time()
for i in range(repeat):
    c = np.tensordot(a, b, (1, 0))
end = time.time()
print("time = {}".format((end - start) / repeat))

print("tvm dummy ffi...")
repeat = 10000
a = np.ones((2, 2))
b = np.ones((2, 2))
start = time.time()
for i in range(repeat):
    c = np.tensordot0(a, b, (1, 0))
end = time.time()
print("time = {}".format((end - start) / repeat))
