#ifndef MXNET_OPERATOR_CONTRIB_UTILS_INL_H_
#define MXNET_OPERATOR_CONTRIB_UTILS_INL_H_

#ifdef MXNET_USE_TVM_OP
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/c_runtime_api.h>
#include <mxnet/base.h>
#include <string>
#include <vector>
#include "../../tvmop/op_module.h"

namespace mxnet {
namespace op {

inline int SplitSch(const ::tvm::runtime::TVMOpConfig& config,
                    const ::std::string& name,
                    const std::vector<int>& size) {
  const ::tvm::runtime::OtherOptionSpace& space = config.get_space(name);
  int weight = config.get_weight(name);
  int num_space = space.size(), num_size = size.size();
  for (int i = 0; i < num_space; ++i) {
    bool flag = true;
    for (int j = 0; j < num_size; ++j) {
      if (size[j] % space[i].get_val() != 0) {
        flag = false;
        break;
      }
    }
    if (flag) {
      return i * weight;
    }
  }
  return -1;
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_TVM_OP

#endif  // MXNET_OPERATOR_CONTRIB_UTILS_INL_H_