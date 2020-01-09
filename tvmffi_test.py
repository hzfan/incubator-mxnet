import mxnet as mx
import time
from mxnet.ndarray import np

print("tvm ffi...")
a = np.zeros1((3, 4), ctx="cpu(0)")
print(a)
print("legacy ffi...")
a = np.zeros((3, 4), ctx=mx.cpu())
print(a)

print("tvm ffi dummy...")
repeat = 10000
start = time.time()
for i in range(repeat):
    a = np.zeros0((3, 4), ctx="cpu(0)")
end = time.time()
print("time = {}".format((end - start) / repeat))

print("tvm ffi...")
repeat = 10000
start = time.time()
for i in range(repeat):
    a = np.zeros1((3, 4), ctx="cpu(0)")
end = time.time()
print("time = {}".format((end - start) / repeat))
