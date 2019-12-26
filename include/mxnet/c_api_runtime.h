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
 * \file c_api_runtime.h
 * \brief TVM-FFI-like APIs
 */
#ifndef MXNET_C_API_RUNTIME_H_
#define MXNET_C_API_RUNTIME_H_

#include <memory>
#include <mxnet/c_api.h>

/*!
 * \brief Union type of values
 *  being passed through API and function calls.
 */
// typedef union {
//   int64_t v_int64;
//   double v_float64;
//   void* v_handle;
//   const char* v_str;
// } Value;

// typedef struct {
  
// } IntTuple;

// MXNET_DLL size_t _npi_zeros(size_t op_handle, size_t shape);

// MXNET_DLL size_t _npi_zeros(Value* arg_values, int* type_codes, int num_args)

// MXNET_DLL size_t _npi_zeros_dummy(size_t op_handle, size_t shape);


template<typename T>
class ArrayBuilder {
public:
  // ArrayBuilder() = default;
  inline ArrayBuilder() = default;
  inline ArrayBuilder(size_t size): data_(reinterpret_cast<T*>(new T[size])),
                             size_(size) {}
  inline void resize(size_t size) {
    data_.reset(reinterpret_cast<T*>(new T[size]));
    size_ = size;
  }
  inline T& operator[](int i) {
    return data_.get()[i];
  }
  // ~ArrayBuilder() = default;
  // ArrayBuilder(ArrayBuilder<T>&&) = default;
  // ArrayBuilder& operator =(ArrayBuilder&&) = default;
protected:
  friend class Int64ArrayPtr;
  std::unique_ptr<T[]> data_;
  size_t size_;
};

class Int64ArrayPtr {
public:
  inline Int64ArrayPtr() = default;
  inline Int64ArrayPtr(const ArrayBuilder<int64_t>& arr): data_(reinterpret_cast<Int64Array*>(new Int64Array)) {
    data_->data = arr.data_.get();
    data_->size = arr.size_;
  }
  inline void reset(const ArrayBuilder<int64_t>& arr) {
    data_.reset(reinterpret_cast<Int64Array*>(new Int64Array));
    data_->data = arr.data_.get();
    data_->size = arr.size_;
  }
  inline Int64Array* get() {
    return data_.get();
  }
private:
  std::unique_ptr<Int64Array> data_;
};

#endif // MXNET_C_API_RUNTIME_H_
