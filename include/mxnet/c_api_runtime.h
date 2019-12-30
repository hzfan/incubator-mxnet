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

#include <mxnet/c_api.h>
#include <dmlc/logging.h>
#include <memory>

/*! \brief Inhibit C++ name-mangling for MXNet functions. */
#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/*!
 * \brief Union type of values
 *  being passed through API and function calls.
 */
typedef union {
  int64_t v_int64;
  double v_float64;
  size_t v_handle;
  const char* v_str;
} Value;

typedef enum {
  kInt = 0U,
  kUInt = 1U,
  kFloat = 2U,
  kHandle = 3U,
  kNull = 4U,
  kArrayHandle = 7U
} TypeCode;

MXNET_DLL void _npi_zeros(Value* arg_values, TypeCode* type_codes, int num_args, Value* ret_val, TypeCode* ret_type_code);

MXNET_DLL void _npi_zeros_dummy(Value* arg_values, TypeCode* type_codes, int num_args, Value* ret_val, TypeCode* ret_type_code);

typedef struct {
  int64_t* data;
  size_t size;
} Int64Array;


#ifdef __cplusplus
}
#endif  // __cplusplus

// template<typename T>
// class ArrayBuilder {
// public:
//   // ArrayBuilder() = default;
//   inline ArrayBuilder() = default;
//   inline ArrayBuilder(size_t size): data_(reinterpret_cast<T*>(new T[size])),
//                              size_(size) {}
//   inline void resize(size_t size) {
//     data_.reset(reinterpret_cast<T*>(new T[size]));
//     size_ = size;
//   }
//   inline T& operator[](int i) {
//     return data_.get()[i];
//   }
//   // ~ArrayBuilder() = default;
//   // ArrayBuilder(ArrayBuilder<T>&&) = default;
//   // ArrayBuilder& operator =(ArrayBuilder&&) = default;
// protected:
//   friend class Int64ArrayPtr;
//   std::unique_ptr<T[]> data_;
//   size_t size_;
// };

// class Int64ArrayPtr {
// public:
//   inline Int64ArrayPtr() = default;
//   inline Int64ArrayPtr(const ArrayBuilder<int64_t>& arr): data_(reinterpret_cast<Int64Array*>(new Int64Array)) {
//     data_->data = arr.data_.get();
//     data_->size = arr.size_;
//   }
//   inline void reset(const ArrayBuilder<int64_t>& arr) {
//     data_.reset(reinterpret_cast<Int64Array*>(new Int64Array));
//     data_->data = arr.data_.get();
//     data_->size = arr.size_;
//   }
//   inline Int64Array* get() {
//     return data_.get();
//   }
// private:
//   std::unique_ptr<Int64Array> data_;
// };

struct Int64ArrayCtor {
  inline void operator() (Int64Array& obj, size_t size) {
    obj.data = new int64_t[size];
    obj.size = size;
  }
};

struct Int64ArrayDeleter {
  inline void operator() (const Int64Array& obj) {
    delete[] obj.data;
  }
};

struct Int64ArrayMoveCtor {
  inline void operator() (Int64Array& obj, Int64Array&& other) {
    obj.data = other.data;
    obj.size = other.size;
    other.data = nullptr;
    other.size = 0;
  }
};

struct Int64ArrayMoveAssign {
  inline void operator() (Int64Array& obj, Int64Array&& other) {
    std::swap(obj.data, other.data);
    std::swap(obj.size, other.size);
  }
};

struct Int64ArrayWrapper {
  inline Int64ArrayWrapper() = default;
  // inline Int64ArrayWrapper(size_t size) {
  //   Int64ArrayCtor func;
  //   func.operator()(this->obj, size);
  // }
  inline ~Int64ArrayWrapper() {
    Int64ArrayDeleter func;
    func.operator()(this->obj);
  }
  inline Int64ArrayWrapper(const Int64ArrayWrapper& other) {
    CHECK(false) << "Int64ArrayWrapper should not be copy-constructed";
  }
  inline Int64ArrayWrapper& operator= (const Int64ArrayWrapper& other) {
    CHECK(false) << "Int64ArrayWrapper should not be copy-assigned";
    return *this;
  }
  inline Int64ArrayWrapper(Int64ArrayWrapper&& other) {
    Int64ArrayMoveCtor func;
    func.operator()(this->obj, std::move(other.obj));
  }
  inline Int64ArrayWrapper& operator=(Int64ArrayWrapper&& other) {
    Int64ArrayMoveAssign func;
    func.operator()(this->obj, std::move(other.obj));
    return *this;
  }
  inline Int64ArrayWrapper&& move() {
    return std::move(*this);
  }
  Int64Array obj;
};

#endif // MXNET_C_API_RUNTIME_H_
