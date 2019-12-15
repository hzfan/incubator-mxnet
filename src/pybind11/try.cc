#include <pybind11/pybind11.h>

namespace py = pybind11;

int add(int x, int y) {
  return x + y;
}

PYBIND11_MODULE(libmxnet, m)
{
  m.def("add", &add);
}

