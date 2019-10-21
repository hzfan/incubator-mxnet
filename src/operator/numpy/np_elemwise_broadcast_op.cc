/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2019 by Contributors
 * \file np_elemwise_binary_op.cc
 * \brief CPU Implementation of basic functions for elementwise numpy binary broadcast operator.
 */

#include "./np_elemwise_broadcast_op.h"
#include "../tensor/elemwise_binary_broadcast_op.h"
#include "../tensor/elemwise_binary_scalar_op.h"

namespace mxnet {
namespace op {

bool NumpyBinaryScalarType(const nnvm::NodeAttrs& attrs,
                           std::vector<int>* in_attrs,
                           std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  TYPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
  return in_attrs->at(0) != -1;
}

#define MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(name)              \
  NNVM_REGISTER_OP(name)                                            \
  .set_num_inputs(1)                                                \
  .set_num_outputs(1)                                               \
  .set_attr_parser([](NodeAttrs* attrs) {                           \
      attrs->parsed = std::stod(attrs->dict["scalar"]);             \
    })                                                              \
  .set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>) \
  .set_attr<nnvm::FInferType>("FInferType", NumpyBinaryScalarType)  \
  .set_attr<nnvm::FInplaceOption>("FInplaceOption",                 \
    [](const NodeAttrs& attrs){                                     \
      return std::vector<std::pair<int, int> >{{0, 0}};             \
    })                                                              \
  .add_argument("data", "NDArray-or-Symbol", "source input")        \
  .add_argument("scalar", "float", "scalar input")

bool NumpyBinaryMixedPrecisionType(const nnvm::NodeAttrs& attrs,
                                   std::vector<int>* in_attrs,
                                   std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  const int ltype = in_attrs->at(0);
  const int rtype = in_attrs->at(1);
  if (ltype != -1 && rtype != -1 && (ltype != rtype)) {
    // Only when both input types are known and not the same, we enter the mixed-precision mode
    TYPE_ASSIGN_CHECK(*out_attrs, 0, common::np_binary_out_infer_type(ltype, rtype));
  } else {
    return ElemwiseType<2, 1>(attrs, in_attrs, out_attrs);
  }
  return true;
}

#ifndef _WIN32
#define MXNET_OPERATOR_REGISTER_NP_BINARY_MIXED_PRECISION(name)                \
  NNVM_REGISTER_OP(name)                                                       \
  .set_num_inputs(2)                                                           \
  .set_num_outputs(1)                                                          \
  .set_attr<nnvm::FListInputNames>("FListInputNames",                          \
    [](const NodeAttrs& attrs) {                                               \
      return std::vector<std::string>{"lhs", "rhs"};                           \
    })                                                                         \
  .set_attr<mxnet::FInferShape>("FInferShape", BinaryBroadcastShape)           \
  .set_attr<nnvm::FInferType>("FInferType", NumpyBinaryMixedPrecisionType)     \
  .set_attr<nnvm::FInplaceOption>("FInplaceOption",                            \
    [](const NodeAttrs& attrs){                                                \
      return std::vector<std::pair<int, int> >{{0, 0}, {1, 0}};                \
    })                                                                         \
  .add_argument("lhs", "NDArray-or-Symbol", "First input to the function")     \
  .add_argument("rhs", "NDArray-or-Symbol", "Second input to the function")
#else
#define MXNET_OPERATOR_REGISTER_NP_BINARY_MIXED_PRECISION(name)                \
  NNVM_REGISTER_OP(name)                                                       \
  .set_num_inputs(2)                                                           \
  .set_num_outputs(1)                                                          \
  .set_attr<nnvm::FListInputNames>("FListInputNames",                          \
    [](const NodeAttrs& attrs) {                                               \
      return std::vector<std::string>{"lhs", "rhs"};                           \
    })                                                                         \
  .set_attr<mxnet::FInferShape>("FInferShape", BinaryBroadcastShape)           \
  .set_attr<nnvm::FInferType>("FInferType", NumpyBinaryMixedPrecisionType)     \
  .set_attr<nnvm::FInplaceOption>("FInplaceOption",                            \
    [](const NodeAttrs& attrs){                                                \
      return std::vector<std::pair<int, int> >{{0, 0}, {1, 0}};                \
    })                                                                         \
  .set_attr<FResourceRequest>("FResourceRequest",                              \
  [](const NodeAttrs& attrs) {                                                 \
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};          \
  })                                                                           \
  .add_argument("lhs", "NDArray-or-Symbol", "First input to the function")     \
  .add_argument("rhs", "NDArray-or-Symbol", "Second input to the function")
#endif

MXNET_OPERATOR_REGISTER_NP_BINARY_MIXED_PRECISION(_npi_add)
#ifndef _WIN32
.set_attr<FCompute>(
  "FCompute<cpu>",
  NumpyBinaryBroadcastComputeWithBool<cpu, op::mshadow_op::plus, op::mshadow_op::mixed_plus,
                                      op::mshadow_op::mixed_plus>)
#else
.set_attr<FCompute>(
  "FCompute<cpu>",
  NumpyBinaryBroadcastComputeWithBool<cpu, op::mshadow_op::plus>)
#endif
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_broadcast_add"});

MXNET_OPERATOR_REGISTER_NP_BINARY_MIXED_PRECISION(_npi_subtract)
#ifndef _WIN32
.set_attr<FCompute>(
  "FCompute<cpu>",
  NumpyBinaryBroadcastCompute<cpu, op::mshadow_op::minus, op::mshadow_op::mixed_minus,
                              op::mshadow_op::mixed_rminus>)
#else
.set_attr<FCompute>(
  "FCompute<cpu>",
  NumpyBinaryBroadcastCompute<cpu, op::mshadow_op::minus>)
#endif
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_broadcast_sub"});

MXNET_OPERATOR_REGISTER_NP_BINARY_MIXED_PRECISION(_npi_multiply)
#ifndef _WIN32
.set_attr<FCompute>(
  "FCompute<cpu>",
  NumpyBinaryBroadcastComputeWithBool<cpu, op::mshadow_op::mul, op::mshadow_op::mixed_mul,
                                      op::mshadow_op::mixed_mul>)
#else
.set_attr<FCompute>(
  "FCompute<cpu>",
  NumpyBinaryBroadcastComputeWithBool<cpu, op::mshadow_op::mul>)
#endif
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_broadcast_mul"});

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(_npi_mod)
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, mshadow_op::mod>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_broadcast_mod"});

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(_npi_power)
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, mshadow_op::power>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_broadcast_power"});

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(_npi_copysign)
.describe(R"code()code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, mshadow_op::copysign>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_npi_copysign"});

NNVM_REGISTER_OP(_backward_npi_copysign)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 1}};
  })
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastBackwardUseIn<cpu, mshadow_op::copysign_grad,
                                                                  mshadow_op::copysign_rgrad>);

NNVM_REGISTER_OP(_npi_lcm)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
[](const NodeAttrs& attrs) {
     return std::vector<std::string>{"lhs", "rhs"};
})
.set_attr<mxnet::FInferShape>("FInferShape", BinaryBroadcastShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseIntType<2, 1>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
[](const NodeAttrs& attrs){
     return std::vector<std::pair<int, int> >{{0, 0}, {1, 0}};
})
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, mshadow_op::lcm>)
.add_argument("lhs", "NDArray-or-Symbol", "First input to the function")
.add_argument("rhs", "NDArray-or-Symbol", "Second input to the function");

NNVM_REGISTER_OP(_npi_lcm_scalar)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser([](NodeAttrs* attrs) {
    attrs->parsed = std::stod(attrs->dict["scalar"]);
  })
.set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseIntType<1, 1>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("data", "NDArray-or-Symbol", "source input")
.add_argument("scalar", "int", "scalar input")
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, mshadow_op::lcm>);

// MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(_npi_add_scalar)
// .set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, op::mshadow_op::plus>)
// .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_copy"});

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(_npi_subtract_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, op::mshadow_op::minus>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_copy"});

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(_npi_rsubtract_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, mshadow_op::rminus>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"negative"});

// MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(_npi_multiply_scalar)
// .set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, op::mshadow_op::mul>)
// .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_mul_scalar"});

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(_npi_mod_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, mshadow_op::mod>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_mod_scalar"});

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(_npi_rmod_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, mshadow_op::rmod>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_rmod_scalar"});

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(_npi_power_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, mshadow_op::power>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_power_scalar"});

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(_npi_rpower_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, mshadow_op::rpower>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{"_backward_rpower_scalar"});

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(_npi_copysign_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, mshadow_op::copysign>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_npi_copysign_scalar"});

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(_npi_rcopysign_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, mshadow_op::rcopysign>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_npi_rcopysign_scalar"});

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(_backward_npi_copysign_scalar)
.set_attr<FCompute>("FCompute<cpu>",
                    BinaryScalarOp::Backward<cpu, mshadow_op::copysign_grad>);

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(_backward_npi_rcopysign_scalar)
.set_attr<FCompute>("FCompute<cpu>",
                    BinaryScalarOp::Backward<cpu, mshadow_op::rcopysign_grad>);

inline bool IsFloatType(const int dtype) {
  return (dtype == mshadow::kFloat16 ||
          dtype == mshadow::kFloat32 ||
          dtype == mshadow::kFloat64);
}

inline bool Arctan2OpType(const nnvm::NodeAttrs& attrs,
                          std::vector<int>* in_attrs,
                          std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);

  TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(1));
  TYPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
  TYPE_ASSIGN_CHECK(*in_attrs, 1, out_attrs->at(0));
  // check if it is float16, float32 or float64. If not, raise error.
  CHECK(IsFloatType(in_attrs->at(0))) << "Do not support `int` as input.\n";
  return out_attrs->at(0) != -1;
}

NNVM_REGISTER_OP(_npi_arctan2)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"x1", "x2"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", BinaryBroadcastShape)
.set_attr<nnvm::FInferType>("FInferType", Arctan2OpType)
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, mshadow_op::arctan2>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_npi_arctan2"})
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.add_argument("x1", "NDArray-or-Symbol", "The input array")
.add_argument("x2", "NDArray-or-Symbol", "The input array");

NNVM_REGISTER_OP(_backward_npi_arctan2)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastBackwardUseIn<cpu, mshadow_op::arctan2_grad,
                                                                  mshadow_op::arctan2_rgrad>);

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(_npi_arctan2_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, mshadow_op::arctan2>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_npi_arctan2_scalar"});

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(_npi_rarctan2_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, mshadow_op::rarctan2>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_npi_rarctan2_scalar"});

MXNET_OPERATOR_REGISTER_BINARY(_backward_npi_arctan2_scalar)
.add_argument("scalar", "float", "scalar value")
.set_attr_parser([](NodeAttrs *attrs) { attrs->parsed = std::stod(attrs->dict["scalar"]); })
.set_attr<FCompute>("FCompute<cpu>",
                    BinaryScalarOp::Backward<cpu, mshadow_op::arctan2_grad>);

MXNET_OPERATOR_REGISTER_BINARY(_backward_npi_rarctan2_scalar)
.add_argument("scalar", "float", "scalar value")
.set_attr_parser([](NodeAttrs *attrs) { attrs->parsed = std::stod(attrs->dict["scalar"]); })
.set_attr<FCompute>("FCompute<cpu>",
                    BinaryScalarOp::Backward<cpu, mshadow_op::arctan2_rgrad>);

bool HypotOpType(const nnvm::NodeAttrs& attrs,
                 std::vector<int>* in_attrs,
                 std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);

  TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(1));
  TYPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
  TYPE_ASSIGN_CHECK(*in_attrs, 1, out_attrs->at(0));

  CHECK(IsFloatType(in_attrs->at(0))) << "Do not support `int` as input.\n";
  return out_attrs->at(0) != -1;
}

// rigister hypot that do not support int here
NNVM_REGISTER_OP(_npi_hypot)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"x1", "x2"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", BinaryBroadcastShape)
.set_attr<nnvm::FInferType>("FInferType", HypotOpType)
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, mshadow_op::hypot>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_npi_hypot"})
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int> >{{0, 0}, {1, 0}};
  })
.add_argument("x1", "NDArray-or-Symbol", "The input array")
.add_argument("x2", "NDArray-or-Symbol", "The input array");

NNVM_REGISTER_OP(_backward_npi_hypot)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int> > {{0, 1}};
  })
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastBackwardUseIn<cpu, mshadow_op::hypot_grad_left,
                                                                  mshadow_op::hypot_grad_right>);

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(_npi_ldexp)
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, mshadow_op::ldexp>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_npi_ldexp"});

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(_npi_ldexp_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, mshadow_op::ldexp>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_npi_ldexp_scalar"});

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(_npi_rldexp_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, mshadow_op::rldexp>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_npi_rldexp_scalar"});

NNVM_REGISTER_OP(_backward_npi_ldexp)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 1}};
  })
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastBackwardUseIn<cpu, mshadow_op::ldexp_grad,
                                                                  mshadow_op::ldexp_rgrad>);

MXNET_OPERATOR_REGISTER_BINARY(_backward_npi_ldexp_scalar)
.add_argument("scalar", "float", "scalar value")
.set_attr_parser([](NodeAttrs *attrs) { attrs->parsed = std::stod(attrs->dict["scalar"]); })
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Backward<cpu, mshadow_op::ldexp_grad>);

MXNET_OPERATOR_REGISTER_BINARY(_backward_npi_rldexp_scalar)
.add_argument("scalar", "float", "scalar value")
.set_attr_parser([](NodeAttrs *attrs) { attrs->parsed = std::stod(attrs->dict["scalar"]); })
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Backward<cpu, mshadow_op::rldexp_grad>);

#if MXNET_USE_TVM_OP
TBlob PrependAxes(const TBlob& src, const int dst_ndim);

inline std::string set_attr(const std::string& name,
                            const std::string& val) {
  return name + '_' + val;
}

inline std::string set_req(OpReqType req) {
  if (req == kWriteTo)
    return "req_kWriteTo";
  return "req_kAddTo";
}

static constexpr int maxdim = 5;
static constexpr char func_multiply_cpu[] = "multiply_cpu";
static constexpr char func_multiply_gpu[] = "multiply_gpu";
static constexpr char func_backward_multiply_cpu[] = "backward_multiply_cpu";
static constexpr char func_backward_multiply_gpu[] = "backward_multiply_gpu";
static constexpr char func_add_cpu[] = "add_cpu";
static constexpr char func_add_gpu[] = "add_gpu";
static constexpr char func_backward_add_cpu[] = "backward_add_cpu";
static constexpr char func_backward_add_gpu[] = "backward_add_gpu";
static constexpr char func_logaddexp_cpu[] = "logaddexp_cpu";

template<const char* func>
void TVMBinaryBroadcastCompute(const nnvm::NodeAttrs& attrs,
                               const mxnet::OpContext& ctx,
                               const std::vector<TBlob>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  if (outputs[0].shape_.Size() == 0U) return;  // skip zero-size tensor
  // prepare tblobs and TVMArgs
  std::vector<TBlob> tblobs = {inputs[0], inputs[1], outputs[0], outputs[0]};
  std::vector<int> type_codes;
  std::vector<TVMValue> values;

  const size_t num_args = 4;
  type_codes.resize(num_args);
  values.resize(num_args);
  for (size_t i = 0; i < num_args; ++i) {
    tblobs[i] = PrependAxes(tblobs[i], maxdim);
    type_codes[i] = kArrayHandle;
    values[i].v_handle = const_cast<DLTensor*>(&(tblobs[i].dltensor()));
  }

  std::string funcname = std::string(func);
  MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
    funcname += set_req(req_type);
  });

  tvm::runtime::TVMArgs tvm_args(&values[0], &type_codes[0], num_args);
  tvm::runtime::TVMOpModule::Get()->CallEx(funcname, ctx, tblobs, tvm_args);
}

enum AxisType
{
  XReduce,     // axis k's broadcast axis
  YReduce,     // axis 1 - k's broadcast axis
  XYIter       // other axis
};

enum ReductionType
{
  Reduce,      // axis k's broadcast axis
  Iter         // axis k's iter axis
};

template<const char* func>
void TVMBinaryBroadcastBackwardUseIn(const nnvm::NodeAttrs& attrs,
                                     const mxnet::OpContext& ctx,
                                     const std::vector<TBlob>& inputs,
                                     const std::vector<OpReqType>& req,
                                     const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 2U);
  const int ndim = inputs[0].shape_.ndim();
  const int nin = outputs.size();
  const TShape& oshape = inputs[0].shape_;
  TShape ishape[nin];
  for (int k = 0; k < nin; ++k) {
    ishape[k] = PrependAxes(inputs[1 + k], ndim).shape_;
  }
  for (int k = 0; k < 2; ++k) {
    // dispatch by broadcast dims
    // seperate outputs[k] iter dim from outputs[1 - k] reduce dim
    const TShape& xs = ishape[k], ys = ishape[1 - k];
    // get axis type
    std::vector<AxisType> axis_type(ndim);
    for (int i = 0; i < ndim; ++i) {
      if (oshape[i] != xs[i]) {
        axis_type[i] = XReduce;
      } else if (oshape[i] != ys[i]) {
        axis_type[i] = YReduce;
      } else {
        axis_type[i] = XYIter;
      }
    }
    // get reduction type of x with seperation dims inserted
    std::vector<ReductionType> seperated_type;
    std::vector<AxisType> seperated_axis_type;
    std::vector<int> seperated_shape;
    for (int i = 0; i < ndim; ++i) {
      ReductionType val;
      if (i > 0 && axis_type[i - 1] != XReduce && axis_type[i] != XReduce
          && axis_type[i - 1] != axis_type[i]) {
        seperated_type.push_back(Reduce);
        seperated_axis_type.push_back(XReduce);
        seperated_shape.push_back(1);
      }
      if (axis_type[i] == XReduce) {
        val = Reduce;
      } else {
        val = Iter;
      }
      seperated_type.push_back(val);
      seperated_shape.push_back(oshape[i]);
      seperated_axis_type.push_back(axis_type[i]);
    }
    // Sequeeze continuous dims of the same type
    std::vector<AxisType> otype;
    std::vector<int> ov;
    int size = seperated_type.size();
    for (int i = 0; i < size; ++i) {
      if (i > 0 && seperated_type[i] == seperated_type[i - 1]) {
        ov.back() *= seperated_shape[i];
        CHECK_EQ(otype.back(), seperated_axis_type[i]);
      } else {
        ov.push_back(seperated_shape[i]);
        otype.push_back(seperated_axis_type[i]);
      }
    }
    // Padding to maxdim
    for (int i = ov.size(); i < 2 * maxdim - 1; ++i) {
      ov.push_back(1);
      otype.push_back(XReduce);
    }
    // Calculate reduce1st
    int reduce1st = otype[0] == XReduce;
    // Calculate iv, xy, and yv
    std::vector<int> iv, xv, yv;
    for (int i = reduce1st; i < 2 * maxdim - 1; i += 2) {
      iv.push_back(ov[i]);
    }
    for (int i = 0; i < 2 * maxdim - 1; ++i) {
      if (otype[i] == XReduce) {
        xv.push_back(1);
      } else {
        xv.push_back(ov[i]);
      }
    }
    for (int i = 0; i < 2 * maxdim - 1; ++i) {
      if (otype[i] == YReduce) {
        yv.push_back(1);
      } else {
        yv.push_back(ov[i]);
      }
    }

    // Prepare tblobs and TVMArgs
    TShape oshape_tvm(ov.begin(), ov.end());
    TShape xshape_tvm(xv.begin(), xv.end());
    TShape yshape_tvm(yv.begin(), yv.end());
    TShape ishape_tvm(iv.begin(), iv.end());
    std::vector<TBlob> tblobs = {inputs[0].reshape(oshape_tvm),
                                 inputs[1 + k].reshape(xshape_tvm),
                                 inputs[2 - k].reshape(yshape_tvm),
                                 outputs[k].reshape(ishape_tvm),
                                 outputs[k].reshape(ishape_tvm)};
    std::vector<int> type_codes;
    std::vector<TVMValue> values;
    const int num_args = 5;
    type_codes.resize(num_args);
    values.resize(num_args);
    for (size_t i = 0; i < num_args; ++i) {
      type_codes[i] = kArrayHandle;
      values[i].v_handle = const_cast<DLTensor*>(&(tblobs[i].dltensor()));
    }

    // Set attrs
    std::string funcname = std::string(func);
    funcname += set_attr("output", std::to_string(k));
    funcname += set_attr("reduce1st", std::to_string(reduce1st));
    MXNET_ASSIGN_REQ_SWITCH(req[k], req_type, {
      funcname += set_req(req_type);
    });
    tvm::runtime::TVMArgs tvm_args(&values[0], &type_codes[0], num_args);
    tvm::runtime::TVMOpModule::Get()->CallEx(funcname, ctx, tblobs, tvm_args);
  }
}

template<const char* func>
void TVMBinaryBroadcastBackwardUseNone(const nnvm::NodeAttrs& attrs,
                                       const mxnet::OpContext& ctx,
                                       const std::vector<TBlob>& inputs,
                                       const std::vector<OpReqType>& req,
                                       const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 2U);
  const TShape& oshape = PrependAxes(inputs[0], maxdim).shape_;
  for (int k = 0; k < 2; ++k) {
    // dispatch by backward
    TShape ishape = PrependAxes(outputs[k], maxdim).shape_;
    std::vector<ReductionType> reduction_type;
    for (int i = 0; i < maxdim; ++i) {
      if (oshape[i] != ishape[i]) {
        reduction_type.push_back(Reduce);
      } else {
        reduction_type.push_back(Iter);
      }
    }
    // Calculate ov
    std::vector<int> tv;
    for (int i = 0; i < maxdim; ++i) {
      if (i > 0 && reduction_type[i] == reduction_type[i - 1]) {
        tv.back() *= oshape[i];
      } else {
        tv.push_back(oshape[i]);
      }
    }
    // Prepend to maxdim
    std::vector<int> ov(maxdim - tv.size(), 1), iv;
    for (auto const& i: tv) {
      ov.push_back(i);
    }
    // Calculate reduce1st
    int reduce1st = reduction_type[0] == Reduce;
    reduce1st = (reduce1st + maxdim - tv.size()) % 2;

    // Calculate iv
    for (uint32_t i = reduce1st; i < ov.size(); i += 2) {
      iv.push_back(ov[i]);
    }

    // Prepare tblobs and TVMArgs
    TShape oshape_tvm(ov.begin(), ov.end());
    TShape ishape_tvm(iv.begin(), iv.end());
    std::vector<TBlob> tblobs = {inputs[0].reshape(oshape_tvm),
                                 outputs[k].reshape(ishape_tvm),
                                 outputs[k].reshape(ishape_tvm)};
    std::vector<int> type_codes;
    std::vector<TVMValue> values;
    const size_t num_args = 3;
    type_codes.resize(num_args);
    values.resize(num_args);
    for (size_t i = 0; i < num_args; ++i) {
      type_codes[i] = kArrayHandle;
      values[i].v_handle = const_cast<DLTensor*>(&(tblobs[i].dltensor()));
    }

    std::string funcname = std::string(func);
    funcname += set_attr("output", std::to_string(k));
    funcname += set_attr("reduce1st", std::to_string(reduce1st));
    MXNET_ASSIGN_REQ_SWITCH(req[k], req_type, {
      funcname += set_req(req_type);
    });

    tvm::runtime::TVMArgs tvm_args(&values[0], &type_codes[0], num_args);
    tvm::runtime::TVMOpModule::Get()->CallEx(funcname, ctx, tblobs, tvm_args);
  }
}

#define MXNET_OPERAOTR_REGISTER_BACKWARD_NP_BINARY_BROADCAST_USE_NONE(name)  \
  NNVM_REGISTER_OP(_backward_npi_##name)                                     \
  .set_num_inputs(1)                                                         \
  .set_num_outputs(2)                                                        \
  .set_attr<nnvm::TIsBackward>("TIsBackward", true)                          \
  .set_attr<FCompute>("FCompute<cpu>",                                       \
    TVMBinaryBroadcastBackwardUseNone<func_backward_##name##_cpu>);

#define MXNET_OPERAOTR_REGISTER_BACKWARD_NP_BINARY_BROADCAST_USE_IN(name)  \
  NNVM_REGISTER_OP(_backward_npi_##name)                                   \
  .set_num_inputs(3)                                                       \
  .set_num_outputs(2)                                                      \
  .set_attr<nnvm::TIsBackward>("TIsBackward", true)                        \
  .set_attr<nnvm::FInplaceOption>("FInplaceOption",                        \
    [](const NodeAttrs& attrs){                                            \
      return std::vector<std::pair<int, int> >{{0, 0}};                    \
    })                                                                     \
  .set_attr<FCompute>("FCompute<cpu>",                                     \
    TVMBinaryBroadcastBackwardUseIn<func_backward_##name##_cpu>);

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(_npi_multiply)
.set_attr<FCompute>("FCompute<cpu>", TVMBinaryBroadcastCompute<func_multiply_cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_npi_multiply"});

MXNET_OPERAOTR_REGISTER_BACKWARD_NP_BINARY_BROADCAST_USE_IN(multiply);

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(_npi_add)
.set_attr<FCompute>("FCompute<cpu>", TVMBinaryBroadcastCompute<func_add_cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_npi_add"});

MXNET_OPERAOTR_REGISTER_BACKWARD_NP_BINARY_BROADCAST_USE_NONE(add);

#if MXNET_USE_CUDA
NNVM_REGISTER_OP(_npi_multiply)
.set_attr<FCompute>("FCompute<gpu>", TVMBinaryBroadcastCompute<func_multiply_gpu>);

NNVM_REGISTER_OP(_backward_npi_multiply)
.set_attr<FCompute>("FCompute<gpu>", TVMBinaryBroadcastBackwardUseIn<func_backward_multiply_gpu>);

NNVM_REGISTER_OP(_npi_add)
.set_attr<FCompute>("FCompute<gpu>", TVMBinaryBroadcastCompute<func_multiply_add>);

NNVM_REGISTER_OP(_backward_npi_add)
.set_attr<FCompute>("FCompute<gpu>", TVMBinaryBroadcastBackwardUseNone<func_backward_add_gpu>);

#endif  // MXNET_USE_CUDA

static constexpr char func_multiply_scalar_cpu[] = "multiply_scalar_cpu";
static constexpr char func_multiply_scalar_gpu[] = "multiply_scalar_gpu";
static constexpr char func_backward_multiply_scalar_cpu[] = "backward_multiply_scalar_cpu";
static constexpr char func_backward_multiply_scalar_gpu[] = "backward_multiply_scalar_gpu";
static constexpr char func_add_scalar_cpu[] = "add_scalar_cpu";
static constexpr char func_add_scalar_gpu[] = "add_scalar_gpu";
static constexpr char func_backward_add_scalar_cpu[] = "backward_add_scalar_cpu";
static constexpr char func_backward_add_scalar_gpu[] = "backward_add_scalar_gpu";

template<const char* func>
void TVMBinaryBroadcastScalarCompute(const nnvm::NodeAttrs& attrs,
                                     const mxnet::OpContext& ctx,
                                     const std::vector<TBlob>& inputs,
                                     const std::vector<OpReqType>& req,
                                     const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  if (outputs[0].shape_.Size() == 0U) return;  // skip zero-size tensor

  // prepare tblobs and TVMArgs
  std::vector<TBlob> tblobs = {inputs[0], outputs[0], outputs[0]};
  std::vector<int> type_codes;
  std::vector<TVMValue> values;

  // prepend axes
  for (uint32_t i = 0; i < tblobs.size(); ++i)
    tblobs[i] = PrependAxes(tblobs[i], maxdim);

  const size_t num_args = 4;  // one input tensor, one scalar param, and one output
  type_codes.resize(num_args);
  values.resize(num_args);


  // input tensor setup
  type_codes[0] = kArrayHandle;
  values[0].v_handle = const_cast<DLTensor*>(&(tblobs[0].dltensor()));

  // scalar param
  type_codes[1] = kDLFloat;
  values[1].v_float64 = nnvm::get<double>(attrs.parsed);

  // output tensor
  type_codes[2] = kArrayHandle;
  values[2].v_handle = const_cast<DLTensor*>(&(tblobs[1].dltensor()));

  // output tensor
  type_codes[3] = kArrayHandle;
  values[3].v_handle = const_cast<DLTensor*>(&(tblobs[1].dltensor()));

  std::string funcname = std::string(func);
  MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
    funcname += set_req(req_type);
  });

  tvm::runtime::TVMArgs tvm_args(&values[0], &type_codes[0], num_args);
  tvm::runtime::TVMOpModule::Get()->CallEx(funcname, ctx, tblobs, tvm_args);
}

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(_npi_multiply_scalar)
.set_attr<FCompute>("FCompute<cpu>", TVMBinaryBroadcastScalarCompute<func_multiply_scalar_cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_npi_multiply_scalar"});

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(_backward_npi_multiply_scalar)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>",
  TVMBinaryBroadcastScalarCompute<func_backward_multiply_scalar_cpu>);

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(_npi_add_scalar)
.set_attr<FCompute>("FCompute<cpu>", TVMBinaryBroadcastScalarCompute<func_add_scalar_cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_npi_add_scalar"});

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(_backward_npi_add_scalar)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>",
  TVMBinaryBroadcastScalarCompute<func_backward_add_scalar_cpu>);

#if MXNET_USE_CUDA
NNVM_REGISTER_OP(_npi_multiply_scalar)
.set_attr<FCompute>("FCompute<gpu>", TVMBinaryBroadcastScalarCompute<func_multiply_scalar_gpu>);

NNVM_REGISTER_OP(_backward_npi_multiply_scalar)
.set_attr<FCompute>("FCompute<gpu>", TVMBinaryBroadcastScalarCompute<func_backward_multiply_gpu>);

NNVM_REGISTER_OP(_npi_add_scalar)
.set_attr<FCompute>("FCompute<gpu>", TVMBinaryBroadcastScalarCompute<func_add_scalar_gpu>);

NNVM_REGISTER_OP(_backward_npi_add_scalar)
.set_attr<FCompute>("FCompute<gpu>", TVMBinaryBroadcastScalarCompute<func_backward_add_gpu>);
#endif  // MXNET_USE_CUDA

#endif  // MXNET_USE_TVM_OP

}  // namespace op
}  // namespace mxnet
