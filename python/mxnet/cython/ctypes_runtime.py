import ctypes

class Int64Array(ctypes.Structure):
    """Temp data structure for byte array."""
    _fields_ = [("data", ctypes.POINTER(ctypes.c_int64)),
                ("size", ctypes.c_size_t)]
