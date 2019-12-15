#include <mxnet/c_api.h>
#include <mxnet/base.h>
#include <mxnet/imperative.h>
#include <mxnet/pybind11.h>
#include <nnvm/pass_functions.h>
#include <dmlc/parameter.h>
#include <dmlc/optional.h>
#include "../operator/mxnet_op.h"

struct InitOpParam : public dmlc::Parameter<InitOpParam> {
  mxnet::TShape shape;
  std::string ctx;
  int dtype;
  DMLC_DECLARE_PARAMETER(InitOpParam) {
    DMLC_DECLARE_FIELD(shape)
    .set_default(mxnet::TShape(0, 1))
    .describe("The shape of the output");
    DMLC_DECLARE_FIELD(ctx)
    .set_default("")
    .describe("Context of output, in format [cpu|gpu|cpu_pinned](n)."
              "Only used for imperative calls.");
    DMLC_DECLARE_FIELD(dtype).set_default(mshadow::kFloat32)
    MXNET_ADD_ALL_TYPES_WITH_BOOL
    .describe("Target data type.");
  }
};

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
  InitOpParam param;
  attrs.op = op;
  // InitOpParam::shape
  size_t num_params = py::len(param_keys);
  for (size_t i = 0; i < num_params; ++i) {
    if (param_keys[i].cast<std::string>() == "shape") {
      param.shape = param_vals[i].cast<mxnet::TShape>();
    } else if (param_keys[i].cast<std::string>() == "ctx") {
      param.ctx = param_vals[i].cast<std::string>();
    } else if (param_keys[i].cast<std::string>() == "dtype") {
      if (param_vals[i].cast<std::string>() == "float32") {
        param.dtype = mshadow::kFloat32;
      } else if (param_vals[i].cast<std::string>() == "float64") {
        param.dtype = mshadow::kFloat64;
      } else if (param_vals[i].cast<std::string>() == "float16") {
        param.dtype = mshadow::kFloat16;
      } else if (param_vals[i].cast<std::string>() == "uint8") {
        param.dtype = mshadow::kUint8;
      } else if (param_vals[i].cast<std::string>() == "int8") {
        param.dtype = mshadow::kInt8;
      } else if (param_vals[i].cast<std::string>() == "int32") {
        param.dtype = mshadow::kInt32;
      } else if (param_vals[i].cast<std::string>() == "int64") {
        param.dtype = mshadow::kInt64;
      }
    }
  }
  std::cout << "shape: " << param.shape << std::endl;
  std::cout << "ctx: " << param.ctx << std::endl;
  std::cout << "dtype: " << param.dtype << std::endl;
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

void MXImperativeInvokeExZeros(size_t creator_s,
                               std::vector<size_t> inputs_s,
                               std::vector<size_t> outputs_s,
                               py::list param_keys,
                               py::list param_vals) {
  const nnvm::Op* op = reinterpret_cast<nnvm::Op*>(creator_s);
  size_t num_inputs = inputs_s.size();
  nnvm::NodeAttrs attrs = ParseAttrsZeros(op, num_inputs, 
                                                      param_keys, param_vals);
}


PYBIND11_MODULE(libmxnet, m)
{
  m.def("MXImperativeInvokeExZeros", &MXImperativeInvokeExZeros);
}
