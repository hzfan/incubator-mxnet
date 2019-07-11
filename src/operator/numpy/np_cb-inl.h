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

#ifndef MXNET_OPERATOR_NUMPY_NP_CB_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_CB_INL_H_

#include <dmlc/parameter.h>
#include <mxnet/operator_util.h>
#include <vector>
#include <utility>
#include <algorithm>
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"
#include "../tensor/broadcast_reduce_op.h"

namespace mxnet {
namespace op {

typedef int (PYFUNC)();

struct NumpyCbParam: public dmlc::Parameter<NumpyCbParam> {
  
  std::string pyfunc;
  DMLC_DECLARE_PARAMETER(NumpyCbParam) {
    DMLC_DECLARE_FIELD(pyfunc)
    .describe("pyfunc");
  }
  /*
  template<typename DType>
  inline parameter::FieldEntry<DType>& DECLARE(
      parameter::ParamManagerSingleton<PType> *manager,
      const std::string &key, DType &ref) { // NOLINT(*)
    parameter::FieldEntry<DType> *e =
        new parameter::FieldEntry<DType>();
    e->Init(key, this->head(), ref);
    manager->manager.AddEntry(key, e);
    return *e;
  } */
};

PYFUNC* GetAddr(std::string str) {
  std::size_t start = str.find("0x");
  std::size_t end = str.find(">");
  std::string sub = str.substr(start, end - start);
  std::stringstream ss;
  ss << std::hex << sub;
  unsigned long long addr;
  ss >> addr;
  PYFUNC* ret = (PYFUNC*) addr;
  return ret;
}

template<typename xpu>
void NumpyCbForward(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob& out_data = outputs[0];
  const NumpyCbParam& param = nnvm::get<NumpyCbParam>(attrs.parsed);
  std::cout << "pyfunc = " << param.pyfunc;
  // PYFUNC* f = GetAddr(param.pyfunc);
  // std::cout << "f = " << f();

  // MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
  //     out_data.FlatTo1D<xpu, DType>(s) = (*f)();
  // });
    
}

template<typename xpu>
void NumpyCbBackward(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const std::vector<TBlob>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  using namespace mshadow;
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_CB_INL_H_
