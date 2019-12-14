#include <mxnet/c_api.h>
#include <mxnet/tuple.h>
#include <mxnet/pybind11.h>

void MXImperativeInvokeExZeros(size_t creator_s,
                               std::vector<size_t> inputs_s,
                               std::vector<size_t> outputs_s,
                               py::list param_keys,
                               py::list param_vals) {
  const nnvm::Op* op = reinterpret_cast<nnvm::Op*>(creator_s);
  size_t num_inputs = inputs_s.size();
  ParseAttrsZeros(op, num_inputs, param_keys, param_vals);
}


PYBIND11_MODULE(libmxnet, m)
{
  m.def("MXImperativeInvokeExZeros", &MXImperativeInvokeExPybind);
}
