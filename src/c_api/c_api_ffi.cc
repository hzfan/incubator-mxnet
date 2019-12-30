#include <inttypes.h>
#include <mxnet/base.h>
#include <mxnet/c_api_runtime.h>
#include "../operator/tensor/init_op.h"
#include "../imperative/imperative_utils.h"

// size_t _npi_zeros(size_t op_handle, size_t shape) {
//   const nnvm::Op* op = static_cast<nnvm::Op*>(reinterpret_cast<void*>(op_handle));

//   mxnet::op::InitOpParam param;
//   param.shape = *reinterpret_cast<TShape*>(shape);
//   param.dtype = 0;
//   param.ctx = "cpu";
//   nnvm::NodeAttrs attrs;
//   attrs.parsed = std::move(param);
//   attrs.op = op;

//   int num_inputs = 0;
//   int infered_num_outputs;
//   int num_visible_outputs;
//   mxnet::imperative::SetNumOutputs(op, attrs, num_inputs, &infered_num_outputs, &num_visible_outputs);

//   std::vector<mxnet::NDArray*> ndoutputs(1, nullptr), ndinputs;
//   ndoutputs[0] = new mxnet::NDArray();
//   auto state = mxnet::Imperative::Get()->Invoke(Context::CPU(), attrs, ndinputs, ndoutputs);

//   return reinterpret_cast<size_t>(ndoutputs[0]);
// }

void _npi_zeros(Value* arg_values, TypeCode* type_codes, int num_args, Value* ret_val, TypeCode* ret_type_code) {
  const nnvm::Op* op = reinterpret_cast<nnvm::Op*>(arg_values[0].v_handle);
  Int64Array* arr = reinterpret_cast<Int64Array*>(arg_values[1].v_handle);
  mxnet::op::InitOpParam param;
  param.shape = TShape(arr->size, 0);
  for (size_t i = 0; i < arr->size; ++i) {
    param.shape[i] = arr->data[i];
  }
  param.dtype = 0;
  param.ctx = "cpu";
  nnvm::NodeAttrs attrs;
  attrs.parsed = std::move(param);
  attrs.op = op;

  int num_inputs = 0;
  int infered_num_outputs;
  int num_visible_outputs;
  mxnet::imperative::SetNumOutputs(op, attrs, num_inputs, &infered_num_outputs, &num_visible_outputs);

  std::vector<mxnet::NDArray*> ndoutputs(1, nullptr), ndinputs;
  ndoutputs[0] = new mxnet::NDArray();
  auto state = mxnet::Imperative::Get()->Invoke(Context::CPU(), attrs, ndinputs, ndoutputs);

  ret_val->v_handle = reinterpret_cast<size_t>(ndoutputs[0]);
  (*ret_type_code) = kArrayHandle;
}

void _npi_zeros_dummy(Value* arg_values, TypeCode* type_codes, int num_args, Value* ret_val, TypeCode* ret_type_code) {
  ret_val->v_handle = reinterpret_cast<size_t>(new mxnet::NDArray());
  (*ret_type_code) = kArrayHandle;
}

//#endif  // MXNET_USE_PYBIND11