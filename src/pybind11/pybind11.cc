#include <tuple>

#include <mxnet/c_api.h>
#include <mxnet/base.h>
#include <mxnet/imperative.h>
#include <mxnet/pybind11.h>
#include <nnvm/pass_functions.h>
#include <dmlc/parameter.h>
#include <dmlc/optional.h>
#include <pybind11/stl.h>
#include "../operator/mxnet_op.h"
#include "../operator/tensor/init_op.h"
#include "../imperative/imperative_utils.h"

using namespace mxnet;

// struct InitOpParam : public dmlc::Parameter<InitOpParam> {
//   mxnet::TShape shape;
//   std::string ctx;
//   int dtype;
//   DMLC_DECLARE_PARAMETER(InitOpParam) {
//     DMLC_DECLARE_FIELD(shape)
//     .set_default(mxnet::TShape(0, 1))
//     .describe("The shape of the output");
//     DMLC_DECLARE_FIELD(ctx)
//     .set_default("")
//     .describe("Context of output, in format [cpu|gpu|cpu_pinned](n)."
//               "Only used for imperative calls.");
//     DMLC_DECLARE_FIELD(dtype).set_default(mshadow::kFloat32)
//     MXNET_ADD_ALL_TYPES_WITH_BOOL
//     .describe("Target data type.");
//   }
// };

/*!
 * \brief Parse parameter attributes into a nnvm::NodeAttrs structure
 * \param op Pointer to the nnvm Operator object
 * \param num_inputs Number of operator inputs
 * \param num_params Number of parameters
 * \param param_keys Array of string pointers representing the parameter keys
 * \param param_vals Array of string pointers representing the associated values
 * \return nnvm::NodeAttrs structure representing the parsed attributes
 */
inline nnvm::NodeAttrs ParseAttrsZeros(const nnvm::Op *op,
                                       const int num_inputs,
                                       py::list param_keys,
                                       py::list param_vals) {
  nnvm::NodeAttrs attrs;
  mxnet::op::InitOpParam param;
  attrs.op = op;
  // InitOpParam::shape
  size_t num_params = py::len(param_keys);
  param.shape = param_vals[0].cast<mxnet::TShape>();
  // for (size_t i = 0; i < num_params; ++i) {
  //   if (param_keys[i].cast<std::string>() == "shape") {
  //     param.shape = param_vals[i].cast<mxnet::TShape>();
  //   } else if (param_keys[i].cast<std::string>() == "ctx") {
  //     param.ctx = param_vals[i].cast<std::string>();
  //   } else if (param_keys[i].cast<std::string>() == "dtype") {
  //     const std::string dtype = param_vals[i].cast<std::string>();
  //     if (dtype == "float32") {
  //       param.dtype = mshadow::kFloat32;
  //     } else if (dtype == "float64") {
  //       param.dtype = mshadow::kFloat64;
  //     } else if (dtype == "float16") {
  //       param.dtype = mshadow::kFloat16;
  //     } else if (dtype == "uint8") {
  //       param.dtype = mshadow::kUint8;
  //     } else if (dtype == "int8") {
  //       param.dtype = mshadow::kInt8;
  //     } else if (dtype == "int32") {
  //       param.dtype = mshadow::kInt32;
  //     } else if (dtype == "int64") {
  //       param.dtype = mshadow::kInt64;
  //     }
  //   }
  // }
  // std::cout << "shape: " << param.shape << std::endl;
  // std::cout << "ctx: " << param.ctx << std::endl;
  // std::cout << "dtype: " << param.dtype << std::endl;
  attrs.parsed = std::move(param);
  // attrs.dict.reserve(num_params+1);
  // for (int i = 0; i < num_params; ++i) {
  //   attrs.dict.emplace(param_keys[i], param_vals[i]);
  // }
  // if (num_args.count(op)) {
  //   attrs.dict.emplace(num_args[op], std::to_string(num_inputs));
  // }
  return attrs;
}

void SetNDInputsOutputsZeros(const nnvm::Op* op,
                        std::vector<NDArray*>* ndinputs,
                        std::vector<NDArray*>* ndoutputs,
                        int num_inputs,
                        const NDArrayHandle *inputs,
                        int *num_outputs,
                        int infered_num_outputs,
                        int num_visible_outputs,
                        NDArrayHandle **outputs) {
  NDArray** out_array = *reinterpret_cast<NDArray***>(outputs);

  ndinputs->clear();
  ndinputs->reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    NDArray* inp = reinterpret_cast<NDArray*>(inputs[i]);
    if (!features::is_enabled(features::INT64_TENSOR_SIZE)) {
      CHECK_LT(inp->shape().Size(), (int64_t{1} << 31) - 1) <<
                "[SetNDInputsOutputs] Size of tensor you are trying to allocate is larger than "
                "2^31 elements. Please build with flag USE_INT64_TENSOR_SIZE=1";
    }
    ndinputs->emplace_back(inp);
  }

  ndoutputs->clear();
  ndoutputs->reserve(infered_num_outputs);
  if (out_array == nullptr) {
    for (int i = 0; i < infered_num_outputs; ++i) {
      ndoutputs->emplace_back(new NDArray());
    }
    *num_outputs = num_visible_outputs;
  } else {
    CHECK(*num_outputs == infered_num_outputs || *num_outputs == num_visible_outputs)
      << "Operator expects " << infered_num_outputs << " (all) or "
      << num_visible_outputs << " (visible only) outputs, but got "
      << *num_outputs << " instead.";
    for (int i = 0; i < *num_outputs; ++i) {
      ndoutputs->emplace_back(out_array[i]);
    }
    for (int i = *num_outputs; i < infered_num_outputs; ++i) {
      ndoutputs->emplace_back(new NDArray());
    }
  }
}

void MXImperativeInvokeExZeros(size_t creator_s,
                               std::vector<size_t> inputs_s,
                               std::vector<size_t> outputs_s,
                               py::list param_keys,
                               py::list param_vals) {
  const nnvm::Op* op = reinterpret_cast<nnvm::Op*>(creator_s);
  int num_inputs = inputs_s.size();
  int num_outputs = outputs_s.size();
  nnvm::NodeAttrs attrs = ParseAttrsZeros(op, num_inputs,
                                          param_keys, param_vals);
  // int infered_num_outputs;
  // int num_visible_outputs;
  // imperative::SetNumOutputs(op, attrs, num_inputs, &infered_num_outputs, &num_visible_outputs);
  
  // std::vector<NDArray*> ndinputs, ndoutputs;
  // NDArrayHandle* inputs = num_inputs > 0 ? reinterpret_cast<NDArrayHandle*>(&inputs_s[0]) : nullptr;
  // NDArrayHandle* outputs = num_outputs > 0 ? reinterpret_cast<NDArrayHandle*>(&outputs_s[0]) : nullptr;
  // SetNDInputsOutputsZeros(op, &ndinputs, &ndoutputs, num_inputs, inputs,
  //     &num_outputs, infered_num_outputs, num_visible_outputs, &outputs);

  // auto state = Imperative::Get()->Invoke(Context::CPU(), attrs, ndinputs, ndoutputs);
  // if (Imperative::Get()->is_recording()) {
  //   Imperative::Get()->RecordOp(std::move(attrs), ndinputs, ndoutputs, state);
  // }
  // for (int i = num_outputs; i < infered_num_outputs; ++i) delete ndoutputs[i];
  // ndoutputs.resize(num_outputs);
  // std::vector<size_t>* ndoutputs_s = reinterpret_cast<std::vector<size_t>*>(&ndoutputs);
  // std::vector<int> out_stypes;
  // out_stypes.reserve(num_outputs);
  // for (const auto& i: ndoutputs) {
  //   out_stypes.emplace_back(i->storage_type());
  // }
}



size_t _npi_zeros(size_t op_handle, const std::vector<int>& shape) {
  const nnvm::Op* op = static_cast<nnvm::Op*>(reinterpret_cast<void*>(op_handle));

  mxnet::op::InitOpParam param;
  param.shape = TShape(shape.begin(), shape.end());
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

  return reinterpret_cast<size_t>(ndoutputs[0]);
}

size_t _npi_zeros_dummy(size_t op_handle, const std::vector<int>& shape, const std::string& ctx, const std::string& dtype) {
  return 0;
}

size_t _npi_zeros_tshape_dummy(size_t op_handle, const TShape& shape, const std::string& ctx, const std::string& dtype) {
  return 0;
}

PYBIND11_MODULE(libmxnet, m)
{
  m.def("MXImperativeInvokeExZeros", &MXImperativeInvokeExZeros);
  m.def("_npi_zeros", &_npi_zeros, "Creating zeros in shape");
  m.def("_npi_zeros_dummy", &_npi_zeros_dummy, "Creating zeros in shape");
  m.def("_npi_zeros_tshape_dummy", &_npi_zeros_tshape_dummy, "Creating zeros in shape");
}
