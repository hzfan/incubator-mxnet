import os
os.environ["MXNET_ENGINE_TYPE"] = "NaiveEngine"

import mxnet as mx
import time
from mxnet.ndarray import np

#########################zeros##################################
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

#########################tensordot##################################
# print("scalar axis...")
# a = np.ones((2, 2))
# b = np.ones((2, 2))
# c = np.tensordot1(a, b)
# print(c)

# print("tuple axes...")
# a = np.ones((2, 3))
# b = np.ones((3, 2))
# c = np.tensordot1(a, b, ((1, 0), (0, 1)))
# print(c)

# print("new tvm...")
# a = np.ones((2, 3))
# b = np.ones((3, 2))
# c = np.tensordot3(a, b, ((1, 0), (0, 1)))
# print(c)

# print("tvm ffi...")
# repeat = 10000
# a = np.ones((2, 2))
# b = np.ones((2, 2))
# start = time.time()
# for i in range(repeat):
#     c = np.tensordot1(a, b, ((1, 0), (0, 1)))
# end = time.time()
# print("time = {}".format((end - start) / repeat))

# print("legacy ffi...")
# repeat = 10000
# a = np.ones((2, 2))
# b = np.ones((2, 2))
# start = time.time()
# for i in range(repeat):
#     c = np.tensordot(a, b, ((1, 0), (0, 1)))
# end = time.time()
# print("time = {}".format((end - start) / repeat))

# print("tvm new ffi...")
# repeat = 10000
# a = np.ones((2, 2))
# b = np.ones((2, 2))
# start = time.time()
# for i in range(repeat):
#     c = np.tensordot3(a, b, ((1, 0), (0, 1)))
# end = time.time()
# print("time = {}".format((end - start) / repeat))

# print("tvm dummy ffi...")
# repeat = 10000
# a = np.ones((2, 2))
# b = np.ones((2, 2))
# start = time.time()
# for i in range(repeat):
#     c = np.tensordot0(a, b, ((1, 0), (0, 1)))
# end = time.time()
# print("time = {}".format((end - start) / repeat))

# print("tvm new dummy ffi...")
# repeat = 10000
# a = np.ones((2, 2))
# b = np.ones((2, 2))
# start = time.time()
# for i in range(repeat):
#     c = np.tensordot2(a, b, ((1, 0), (0, 1)))
# end = time.time()
# print("time = {}".format((end - start) / repeat))

#########################index##################################
print("tvm ffi...")
a = np.ones((2, 2))
b = a[0]
print(b)

print("legacy ffi...")
a = np.ones((2, 2))
b = np.mxnet_get_item(a, 0)
print(b)

print("benchmark tvm ffi...")
repeat = 10000
a = np.ones((2, 2))
start = time.time()
for i in range(repeat):
   b = np.mxnet_get_item(a, 0)
end = time.time()
print("time = {}".format((end - start) / repeat))

print("benchmark legacy ffi...")
repeat = 10000
a = np.ones((2, 2))
start = time.time()
for i in range(repeat):
   b = a[0: 2, 0: 2]
end = time.time()
print("time = {}".format((end - start) / repeat))
