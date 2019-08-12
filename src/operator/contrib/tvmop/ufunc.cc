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
 * Copyright (c) 2019 by Contributors
 * \file ufunc.cc
 * \brief
 * \author Yizhi Liu
 */
#ifdef MXNET_USE_TVM_OP
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/c_runtime_api.h>
#include <mxnet/base.h>
#include "../../tensor/elemwise_binary_broadcast_op.h"
#include "../../tvmop/op_module.h"
#include "../../tensor/elemwise_binary_op.h"
#include "../../tensor/broadcast_reduce_op.h"

namespace mxnet {
namespace op {

static constexpr char func_vadd_cpu[] = "vadd";
static constexpr char func_vadd_gpu[] = "cuda_vadd";

template<const char* func>
void TVMBroadcastCompute(const nnvm::NodeAttrs& attrs,
                         const mxnet::OpContext& ctx,
                         const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  tvm::runtime::TVMOpModule::Get()->Call(func, ctx, {inputs[0], inputs[1], outputs[0]});
}

NNVM_REGISTER_OP(_contrib_tvm_vadd)
    .set_num_inputs(2)
    .set_num_outputs(1)
    .add_argument("a", "NDArray-or-Symbol", "first input")
    .add_argument("b", "NDArray-or-Symbol", "second input")
    .set_attr<mxnet::FInferShape>("FInferShape", BinaryBroadcastShape)
    .set_attr<nnvm::FInferType>("FInferType", mxnet::op::ElemwiseType<2, 1>)
#if MXNET_USE_CUDA
    .set_attr<mxnet::FCompute>("FCompute<gpu>", mxnet::op::TVMBroadcastCompute<func_vadd_gpu>)
#endif  // MXNET_USE_CUDA
    .set_attr<mxnet::FCompute>("FCompute<cpu>", mxnet::op::TVMBroadcastCompute<func_vadd_cpu>);

struct CumprodParam : public dmlc::Parameter<CumprodParam> {
  dmlc::optional<int> axis;
  DMLC_DECLARE_PARAMETER(CumprodParam) {
    DMLC_DECLARE_FIELD(axis)
      .set_default(dmlc::optional<int>())
      .describe("Axis along which the cumulative product is computed."
        " The default (None) is to compute the cumprod over the flattened array.");
  }
};

DMLC_REGISTER_PARAMETER(CumprodParam);

inline bool CumprodShape(const nnvm::NodeAttrs& attrs,
                         mxnet::ShapeVector *in_attrs,
                         mxnet::ShapeVector *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  const CumprodParam &param = nnvm::get<CumprodParam>(attrs.parsed);

  if (param.axis.has_value()) {
    return ElemwiseShape<1, 1>(attrs, in_attrs, out_attrs);
  } else {
    TShape out_shape(1, in_attrs->at(0).Size());
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, out_shape);
    return shape_is_known(out_attrs->at(0));
  }
}

template<const char* func>
void TVMCumprodCompute(const nnvm::NodeAttrs& attrs,
                       const mxnet::OpContext& ctx,
                       const std::vector<TBlob>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  if (inputs[0].shape_.Size() == 0U) {
    return;
  }
  const CumprodParam &param = nnvm::get<CumprodParam>(attrs.parsed);
  std::string funcname = func;
  if (!param.axis.has_value()) {
    funcname += "axis_None";
  } else {
    int axis = CheckAxis(param.axis.value(), inputs[0].shape_.ndim());
    if (inputs[0].shape_.ndim() > 0) {
      funcname += "axis_" + std::to_string(axis);
    } else {
      funcname += "axis_None";
    }
  }
  tvm::runtime::TVMOpModule::Get()->Call(funcname, ctx, {inputs[0], outputs[0]});
}

template<const char* func>
void TVMBackwardCumprodCompute(const nnvm::NodeAttrs& attrs,
                               const mxnet::OpContext& ctx,
                               const std::vector<TBlob>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  if (outputs[0].shape_.Size() == 0U) {
    return;
  }
  const CumprodParam &param = nnvm::get<CumprodParam>(attrs.parsed);
  std::string funcname = func;
  if (!param.axis.has_value()) {
    funcname += "axis_None";
  } else {
    int axis = CheckAxis(param.axis.value(), inputs[0].shape_.ndim());
    if (inputs[1].shape_.ndim() > 0) {
      funcname += "axis_" + std::to_string(axis);
    } else {
      funcname += "axis_None";
    }
  }
  funcname += "req_";
  MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
    if (req_type == kWriteTo) {
      funcname += "kWriteTo";
    } else {
      funcname += "kAddTo";
    }
    tvm::runtime::TVMOpModule::Get()->Call(funcname, ctx, {inputs[0], inputs[1], outputs[0], outputs[0]});
  })
}

static constexpr char func_vcumprod_cpu[] = "vcumprod";
static constexpr char func_vcumprod_gpu[] = "cuda_vcumprod";
static constexpr char func_backward_vcumprod_cpu[] = "backward_vcumprod";
static constexpr char func_backward_vcumprod_gpu[] = "cuda_backward_vcumprod";

NNVM_REGISTER_OP(_np_tvm_vcumprod)
    .set_attr_parser(ParamParser<CumprodParam>)
    .set_num_inputs(1)
    .set_num_outputs(1)
    .set_attr<nnvm::FListInputNames>("FListInputNames",
    [](const NodeAttrs& attrs) {
      return std::vector<std::string>{"data"};
    })
    .set_attr<mxnet::FInferShape>("FInferShape", CumprodShape)
    .set_attr<nnvm::FInferType>("FInferType", mxnet::op::ElemwiseType<1, 1>)
    .set_attr<mxnet::FCompute>("FCompute<cpu>", mxnet::op::TVMCumprodCompute<func_vcumprod_cpu>)
#if MXNET_USE_CUDA
    .set_attr<mxnet::FCompute>("FCompute<gpu>", mxnet::op::TVMCumprodCompute<func_vcumprod_gpu>)
#endif  // MXNET_USE_CUDA
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_np_tvm_vcumprod"})
    .add_argument("data", "NDArray-or-Symbol", "first input")
    .add_arguments(CumprodParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_np_tvm_vcumprod)
    .set_attr_parser(ParamParser<CumprodParam>)
    .set_num_inputs(2)
    .set_num_outputs(1)
    .set_attr<nnvm::TIsBackward>("TIsBackward", true)
#if MXNET_USE_CUDA
    .set_attr<FCompute>("FCompute<gpu>", mxnet::op::TVMBackwardCumprodCompute<func_backward_vcumprod_gpu>)
#endif  // MXNET_USE_CUDA
    .set_attr<FCompute>("FCompute<cpu>", mxnet::op::TVMBackwardCumprodCompute<func_backward_vcumprod_cpu>);

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_TVM_OP
