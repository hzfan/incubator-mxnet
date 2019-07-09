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

#include "./np_cb-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(NumpyCbParam);

NNVM_REGISTER_OP(_npi_cb)
.describe(R"code()code" ADD_FILELINE)
.set_attr_parser(ParamParser<NumpyCbParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCompute>("FCompute<cpu>", NumpyCbForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_npi_cb"})
.add_argument("data", "NDArray-or-Symbol", "Input ndarray")
.add_arguments(NumpyCbParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_npi_cb)
.set_attr_parser(ParamParser<NumpyCbParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", NumpyCbBackward<cpu>);

}  // namespace op
}  // namespace mxnet
