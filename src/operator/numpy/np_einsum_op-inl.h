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
 * \file np_einsum_op-inl.h
 * \brief Function definition of numpy-compatible einsum operator
 */

#ifndef MXNET_OPERATOR_NUMPY_NP_EINSUM_OP_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_EINSUM_OP_INL_H_

#include <mxnet/operator_util.h>
#include <string>
#include <vector>
#include "../../common/static_array.h"
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../mshadow_op.h"
#include "../elemwise_op_common.h"


namespace mxnet {
namespace op {

#define NPY_MAXDIMS 32
#define NPY_MAXARGS 32


inline TShape get_stride(const TShape& shape) {
  int ndim = shape.ndim(), prod = 1;
  TShape stride = TShape(ndim, -1);
  for (int i = ndim - 1; i >= 0; i--) {
    stride[i] = shape[i] > 1 ? prod : 0;
    prod = prod * shape[i];
  }
  return stride;
}


inline TShape pad(const TShape& shape, int odim) {
  int ndim = shape.ndim();
  CHECK_GE(odim, ndim);
  TShape ret(odim, 1);
  for (int idim = 0; idim < ndim; ++idim) {
    ret[idim] = shape[idim];
  }
  return ret;
}


/*
 * Parses the subscripts for one operand into an output of 'ndim'
 * labels. The resulting 'op_labels' array will have:
 *  - the ASCII code of the label for the first occurrence of a label;
 *  - the (negative) offset to the first occurrence of the label for
 *    repeated labels;
 *  - zero for broadcast dimensions, if subscripts has an ellipsis.
 * For example:
 *  - subscripts="abbcbc",  ndim=6 -> op_labels=[97, 98, -1, 99, -3, -2]
 *  - subscripts="ab...bc", ndim=6 -> op_labels=[97, 98, 0, 0, -3, 99]
 */
inline int parse_operand_subscripts(const char *subscripts, int length,
                                    int ndim, int iop, char *op_labels,
                                    char *label_counts, int *min_label, int *max_label) {
  using namespace mxnet_op;
  int i;
  int idim = 0;
  int ellipsis = -1;

  /* Process all labels for this operand */
  for (i = 0; i < length; ++i) {
    int label = subscripts[i];

    /* A proper label for an axis. */
    if (label > 0 && isalpha(label)) {
      /* Check we don't exceed the operator dimensions. */
      CHECK(idim < ndim)
        << "einstein sum subscripts string contains "
        << "too many subscripts for operand "
        << iop;

      op_labels[idim++] = label;
      if (label < *min_label) {
        *min_label = label;
      }
      if (label > *max_label) {
        *max_label = label;
      }
      label_counts[label]++;
    } else if (label == '.') {
      /* The beginning of the ellipsis. */
      /* Check it's a proper ellipsis. */
      CHECK(!(ellipsis != -1 || i + 2 >= length
              || subscripts[++i] != '.' || subscripts[++i] != '.'))
        << "einstein sum subscripts string contains a "
        << "'.' that is not part of an ellipsis ('...') "
        << "in operand "
        << iop;

      ellipsis = idim;
    } else {
        CHECK(label == ' ')
          << "invalid subscript '" << static_cast<char>(label)
          << "' in einstein sum "
          << "subscripts string, subscripts must "
          << "be letters";
      }
  }

  /* No ellipsis found, labels must match dimensions exactly. */
  if (ellipsis == -1) {
    CHECK(idim == ndim)
      << "operand has more dimensions than subscripts "
      << "given in einstein sum, but no '...' ellipsis "
      << "provided to broadcast the extra dimensions.";
  } else if (idim < ndim) {
    /* Ellipsis found, may have to add broadcast dimensions. */
    /* Move labels after ellipsis to the end. */
    for (i = 0; i < idim - ellipsis; ++i) {
      op_labels[ndim - i - 1] = op_labels[idim - i - 1];
    }
    /* Set all broadcast dimensions to zero. */
    for (i = 0; i < ndim - idim; ++i) {
      op_labels[ellipsis + i] = 0;
    }
  }

  /*
    * Find any labels duplicated for this operand, and turn them
    * into negative offsets to the axis to merge with.
    *
    * In C, the char type may be signed or unsigned, but with
    * twos complement arithmetic the char is ok either way here, and
    * later where it matters the char is cast to a signed char.
    */
  for (idim = 0; idim < ndim - 1; ++idim) {
    int label = op_labels[idim];
    /* If it is a proper label, find any duplicates of it. */
    if (label > 0) {
      /* Search for the next matching label. */
      char *next = reinterpret_cast<char*>(memchr(op_labels + idim + 1, label, ndim - idim - 1));

      while (next != NULL) {
        /* The offset from next to op_labels[idim] (negative). */
        *next = static_cast<char>((op_labels + idim) - next);
        /* Search for the next matching label. */
        next = reinterpret_cast<char*>(memchr(next + 1, label, op_labels + ndim - 1 - next));
      }
    }
  }
  return 0;
}


/*
 * Parses the subscripts for the output operand into an output that
 * includes 'ndim_broadcast' unlabeled dimensions, and returns the total
 * number of output dimensions, or -1 if there is an error. Similarly
 * to parse_operand_subscripts, the 'out_labels' array will have, for
 * each dimension:
 *  - the ASCII code of the corresponding label;
 *  - zero for broadcast dimensions, if subscripts has an ellipsis.
 */
inline int parse_output_subscripts(const char *subscripts, int length,
                                   int ndim_broadcast,
                                   const char *label_counts, char *out_labels) {
  using namespace mxnet_op;
  int i, bdim;
  int ndim = 0;
  int ellipsis = 0;

  /* Process all the output labels. */
  for (i = 0; i < length; ++i) {
    int label = subscripts[i];

    /* A proper label for an axis. */
    if (label > 0 && isalpha(label)) {
      /* Check that it doesn't occur again. */
      CHECK(memchr(subscripts + i + 1, label, length - i - 1) == NULL)
        << "einstein sum subscripts string includes "
        << "output subscript '" << static_cast<char>(label)
        << "' multiple times";

      /* Check that it was used in the inputs. */
      CHECK(label_counts[label] != 0)
        << "einstein sum subscripts string included "
        << "output subscript '" << static_cast<char>(label)
        << "' which never appeared "
        << "in an input";

      /* Check that there is room in out_labels for this label. */
      CHECK(ndim < NPY_MAXDIMS)
        << "einstein sum subscripts string contains "
        << "too many subscripts in the output";

      out_labels[ndim++] = label;
    } else if (label == '.') {
      /* The beginning of the ellipsis. */
      /* Check it is a proper ellipsis. */
      CHECK(!(ellipsis || i + 2 >= length
              || subscripts[++i] != '.' || subscripts[++i] != '.'))
        << "einstein sum subscripts string "
        << "contains a '.' that is not part of "
        << "an ellipsis ('...') in the output";

      /* Check there is room in out_labels for broadcast dims. */
      CHECK(ndim + ndim_broadcast <= NPY_MAXDIMS)
        << "einstein sum subscripts string contains "
        << "too many subscripts in the output";

      ellipsis = 1;
      for (bdim = 0; bdim < ndim_broadcast; ++bdim) {
        out_labels[ndim++] = 0;
      }
    } else {
      CHECK(label == ' ')
        << "invalid subscript '" << static_cast<char>(label)
        << "' in einstein sum "
        << "subscripts string, subscripts must "
        << "be letters";
    }
  }

  /* If no ellipsis was found there should be no broadcast dimensions. */
  CHECK(!(!ellipsis && ndim_broadcast > 0))
    << "output has more dimensions than subscripts "
    << "given in einstein sum, but no '...' ellipsis "
    << "provided to broadcast the extra dimensions.";

  return ndim;
}


inline void get_combined_dims_view(const TBlob& op, int iop,
                                   char *labels,
                                   TShape* newshape,
                                   TShape* newstride) {
  using namespace mxnet_op;
  int idim, ndim, icombine, combineoffset;
  int icombinemap[NPY_MAXDIMS];
  int newdim;

  const TShape& shape = op.shape_;
  TShape stride;
  stride = get_stride(shape);
  ndim = op.shape_.ndim();
  newdim = newshape->ndim();

  /* Initialize the dimensions and strides to zero */
  for (idim = 0; idim < newdim; ++idim) {
    (*newshape)[idim] = 0;
    (*newstride)[idim] = 0;
  }

  /* Copy the dimensions and strides, except when collapsing */
  icombine = 0;
  for (idim = 0; idim < ndim; ++idim) {
    /*
     * The char type may be either signed or unsigned, we
     * need it to be signed here.
     */
    int label = (signed char)labels[idim];
    /* If this label says to merge axes, get the actual label */
    if (label < 0) {
      combineoffset = label;
      label = labels[idim+label];
    } else {
      combineoffset = 0;
      if (icombine != idim) {
        labels[icombine] = labels[idim];
      }
      icombinemap[idim] = icombine;
    }
    /* If the label is 0, it's an unlabeled broadcast dimension */
    if (label == 0) {
      (*newshape)[icombine] = shape[idim];
      (*newstride)[icombine] = stride[idim];
    } else {
      /* Update the combined axis dimensions and strides */
      int i = icombinemap[idim + combineoffset];
      CHECK(!(combineoffset < 0 && (*newshape)[i] != 0 &&
              (*newshape)[i] != shape[idim]))
        << "dimensions in operand " << iop
        << " for collapsing index '" << label
        << "' don't match (" << static_cast<int>((*newshape)[i])
        << " != " << shape[idim] << ")";
      (*newshape)[i] = shape[idim];
      (*newstride)[i] += stride[idim];
    }

    /* If the label didn't say to combine axes, increment dest i */
    if (combineoffset == 0) {
      icombine++;
    }
  }
}


inline static int prepare_op_axes(int ndim, int iop, char *labels,
                                  int *axes, int ndim_iter, char *iter_labels) {
  using namespace mxnet_op;
  int i, label, ibroadcast;

  ibroadcast = ndim-1;
  for (i = ndim_iter-1; i >= 0; --i) {
    label = iter_labels[i];
    /*
     * If it's an unlabeled broadcast dimension, choose
     * the next broadcast dimension from the operand.
     */
    if (label == 0) {
      while (ibroadcast >= 0 && labels[ibroadcast] != 0) {
        --ibroadcast;
      }
      /*
       * If we used up all the operand broadcast dimensions,
       * extend it with a "newaxis"
       */
      if (ibroadcast < 0) {
        axes[i] = -1;
      } else {
        /* Otherwise map to the broadcast axis */
        axes[i] = ibroadcast;
        --ibroadcast;
      }
    } else {
      /* It's a labeled dimension, find the matching one */
      char *match = reinterpret_cast<char*>(memchr(labels, label, ndim));
      /* If the op doesn't have the label, broadcast it */
      if (match == NULL) {
        axes[i] = -1;
      } else {
        /* Otherwise use it */
        axes[i] = match - labels;
      }
    }
  }
  return 0;
}


template<int dimension, int req, bool back>
struct numpy_einsum {
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType* out,
                                  mxnet::common::StaticArray<DType*, NPY_MAXARGS> op,
                                  mshadow::Shape<dimension> oshape,
                                  mshadow::Shape<dimension> ostride,
                                  mshadow::Shape<dimension> reduceshape,
                                  mshadow::Shape<dimension> reducestride,
                                  mshadow::Shape<dimension> itershape,
                                  mxnet::common::StaticArray<mshadow::Shape<dimension>,
                                                             NPY_MAXARGS> iterstride,
                                  int nop,
                                  int iop0,
                                  const DType* out_grad) {
    using namespace mxnet_op;
    index_t oidx = back ? dot(unravel(dot(unravel(i, oshape), ostride), itershape),
                              iterstride[iop0]) : i;
    if (req == kWriteTo) {
      out[oidx] = (DType)0;
    }
    for (int j = 0; j < reduceshape.Size(); j++) {
      mshadow::Shape<dimension> idx = unravel(dot(unravel(j, reduceshape), reducestride) +
                                              dot(unravel(i, oshape), ostride),
                                              itershape);
      DType tmp = back ? out_grad[dot(idx, iterstride[nop])] :  (DType)1;
      for (int iop = 0; iop < nop; ++iop) {
        if (iop != iop0) {
          index_t k = dot(idx, iterstride[iop]);
          tmp = tmp * op[iop][k];
        }
      }
      out[oidx] = out[oidx] + tmp;
    }
  }
};


template<typename xpu, bool back>
inline void NumpyEinsumProcess(const std::vector<TBlob>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<TBlob>& outputs,
                               const char *subscripts, int nop,
                               const OpContext& ctx) {
  using namespace mxnet_op;
  int iop, label, min_label = 127, max_label = 0;
  char label_counts[128];
  char op_labels[NPY_MAXARGS][NPY_MAXDIMS];
  char output_labels[NPY_MAXDIMS], *iter_labels;
  int idim, ndim_output, ndim_broadcast, ndim_iter;

  int op_axes_arrays[NPY_MAXARGS][NPY_MAXDIMS];
  int *op_axes[NPY_MAXARGS];

  std::vector<TShape> opshape(nop), opstride_true(nop);
  TShape oshape, ostride_true;
  TShape reduceshape;
  std::vector<TShape> remainshape(nop);

  TShape ostride, reducestride;
  std::vector<TShape> opstride(nop), remainstride(nop);

  oshape = back ? inputs[0].shape_ : outputs[0].shape_;
  ostride_true = get_stride(oshape);

  /* nop+1 (+1 is for the output) must fit in NPY_MAXARGS */
  CHECK(nop < NPY_MAXARGS)
    << "too many operands provided to einstein sum function";
  CHECK(nop >= 1)
    << "not enough operands provided to einstein sum function";

  /* Step 1: Parse the subscripts string into label_counts and op_labels */
  memset(label_counts, 0, sizeof(label_counts));
  for (iop = 0; iop < nop; ++iop) {
    int length = static_cast<int>(strcspn(subscripts, ",-"));

    CHECK(!(iop == nop - 1 && subscripts[length] == ','))
      << "more operands provided to einstein sum function "
      << "than specified in the subscripts string";
    CHECK(!(iop < nop-1 && subscripts[length] != ','))
      << "fewer operands provided to einstein sum function "
      << "than specified in the subscripts string";
    CHECK_GE(parse_operand_subscripts(subscripts, length,
                                      inputs[iop + back].shape_.ndim(),
                                      iop, op_labels[iop], label_counts,
                                      &min_label, &max_label), 0);

    /* Move subscripts to the start of the labels for the next op */
    subscripts += length;
    if (iop < nop - 1) {
      subscripts++;
    }
  }

  /*
   * Find the number of broadcast dimensions, which is the maximum
   * number of labels == 0 in an op_labels array.
   */
  ndim_broadcast = 0;
  for (iop = 0; iop < nop; ++iop) {
    int count_zeros = 0;
    int ndim;
    char *labels = op_labels[iop];

    ndim = inputs[iop + back].shape_.ndim();
    for (idim = 0; idim < ndim; ++idim) {
      if (labels[idim] == 0) {
        ++count_zeros;
      }
    }

    if (count_zeros > ndim_broadcast) {
      ndim_broadcast = count_zeros;
    }
  }

  /*
   * If there is no output signature, fill output_labels and ndim_output
   * using each label that appeared once, in alphabetical order.
   */
  if (subscripts[0] == '\0') {
    /* If no output was specified, always broadcast left, as usual. */
    for (ndim_output = 0; ndim_output < ndim_broadcast; ++ndim_output) {
      output_labels[ndim_output] = 0;
    }
    for (label = min_label; label <= max_label; ++label) {
      if (label_counts[label] == 1) {
        CHECK(ndim_output < NPY_MAXDIMS)
          << "einstein sum subscript string has too many "
          << "distinct labels";
        output_labels[ndim_output++] = label;
      }
    }
  } else {
    CHECK(subscripts[0] == '-' && subscripts[1] == '>')
      << "einstein sum subscript string does not "
      << "contain proper '->' output specified";
    subscripts += 2;

    /* Parse the output subscript string. */
    ndim_output = parse_output_subscripts(subscripts, strlen(subscripts),
                                          ndim_broadcast, label_counts,
                                          output_labels);
    CHECK_GE(ndim_output, 0);
  }

  /*
   * Step 2:
   * Process all the input ops, combining dimensions into their
   * diagonal where specified.
   */
  for (iop = 0; iop < nop; ++iop) {
    char *labels = op_labels[iop];
    int combine, ndim;

    ndim = inputs[iop + back].shape_.ndim();

    /*
     * Check whether any dimensions need to be combined
     *
     * The char type may be either signed or unsigned, we
     * need it to be signed here.
     */
    combine = 0;
    for (idim = 0; idim < ndim; ++idim) {
      if ((signed char)labels[idim] < 0) {
        combine++;
      }
    }

    /* If any dimensions are combined, create a view which combines them */
    if (combine) {
      TShape tshape(ndim - combine, -1);
      TShape tstride(ndim - combine, -1);
      get_combined_dims_view(inputs[iop + back], iop, labels,
                             &tshape, &tstride);
      opshape[iop] = tshape;
      opstride_true[iop] = tstride;
    } else {
      /* No combining needed */
      opshape[iop] = inputs[iop + back].shape_;
      opstride_true[iop] = get_stride(opshape[iop]);
    }
  }

  /*
   * Step 3:
   * Set up the labels for the iterator (output + combined labels).
   * Can just share the output_labels memory, because iter_labels
   * is output_labels with some more labels appended.
   */
  iter_labels = output_labels;
  ndim_iter = ndim_output;
  for (label = min_label; label <= max_label; ++label) {
    if (label_counts[label] > 0 &&
        memchr(output_labels, label, ndim_output) == NULL) {
      CHECK(ndim_iter < NPY_MAXDIMS)
        << "too many subscripts in einsum";
      iter_labels[ndim_iter++] = label;
    }
  }
  TShape itershape(ndim_iter, -1), iterstride_true(ndim_iter, -1);
  std::vector<TShape> iterstride(nop + 1, TShape(ndim_iter, 0));

  /* Step 4: Set up the op_axes for the iterator */
  itershape = TShape(ndim_iter, -1);
  iterstride_true = TShape(ndim_iter, -1);
  for (iop = 0; iop < nop; ++iop) {
    op_axes[iop] = op_axes_arrays[iop];
    CHECK_GE(prepare_op_axes(opshape[iop].ndim(), iop, op_labels[iop],
             op_axes[iop], ndim_iter, iter_labels), 0);
    for (idim = 0; idim < ndim_iter; idim++) {
      if (op_axes[iop][idim] != -1) {
        iterstride[iop][idim] = opstride_true[iop][op_axes[iop][idim]];
        itershape[idim] = opshape[iop][op_axes[iop][idim]];
      }
    }
  }
  for (idim = 0; idim < ndim_output; ++idim) {
    iterstride[nop][idim] = ostride_true[idim];
  }
  iterstride_true = get_stride(itershape);

  reduceshape = TShape(ndim_iter - ndim_output, 0);
  for (idim = ndim_output; idim < ndim_iter; ++idim) {
    reduceshape[idim - ndim_output] = itershape[idim];
  }
  for (iop = 0; iop < nop; iop++) {
    remainshape[iop] = TShape(ndim_iter - opshape[iop].ndim(), 0);
    int j = 0;
    for (idim = 0; idim < ndim_iter; idim++) {
      if (op_axes_arrays[iop][idim] == -1) {
        remainshape[iop][j++] = itershape[idim];
      }
    }
  }

  // calculate stride
  ostride = TShape(ndim_output, 0);
  for (idim = 0; idim < ndim_output; ++idim) {
    ostride[idim] = iterstride_true[idim];
  }
  reducestride = TShape(ndim_iter - ndim_output, 0);
  for (idim = ndim_output; idim < ndim_iter; ++idim) {
    reducestride[idim - ndim_output] = iterstride_true[idim];
  }
  for (iop = 0; iop < nop; ++iop) {
    opstride[iop] = TShape(opshape[iop].ndim(), 0);
    remainstride[iop] = TShape(ndim_iter - opshape[iop].ndim(), 0);
    int j = 0;
    for (idim = 0; idim < ndim_iter; ++idim) {
      if (op_axes_arrays[iop][idim] != -1) {
        opstride[iop][op_axes_arrays[iop][idim]] = iterstride_true[idim];
      } else {
        remainstride[iop][j++] = iterstride_true[idim];
      }
    }
  }

  // exclude the 0-dim case
  if (ndim_iter == 0) {
    ndim_iter = 1;
  }
  itershape = pad(itershape, ndim_iter);
  for (iop = 0; iop <= nop; ++iop) {
    iterstride[iop] = pad(iterstride[iop], ndim_iter);
  }
  oshape = pad(oshape, ndim_iter);
  ostride = pad(ostride, ndim_iter);
  reduceshape = pad(reduceshape, ndim_iter);
  reducestride = pad(reducestride, ndim_iter);
  for (iop = 0; iop < nop; ++iop) {
    opshape[iop] = pad(opshape[iop], ndim_iter);
    opstride[iop] = pad(opstride[iop], ndim_iter);
    remainshape[iop] = pad(remainshape[iop], ndim_iter);
    remainstride[iop] = pad(remainstride[iop], ndim_iter);
  }

  if (!back) {
    if (oshape.Size() == 0) {
      return;
    }
    const TBlob &out_data = outputs[0];
    MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
      mxnet::common::StaticArray<DType*, NPY_MAXARGS> op;
      for (iop = 0; iop < nop; ++iop) {
        op[iop] = inputs[iop].dptr<DType>();
      }
      MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
        MXNET_NDIM_SWITCH(ndim_iter, dimension, {
          mxnet::common::StaticArray<mshadow::Shape<dimension>, NPY_MAXARGS> iterstride_arr;
          for (iop = 0; iop <= nop; ++iop) {
            iterstride_arr[iop] = iterstride[iop].get<dimension>();
          }
          Kernel<numpy_einsum<dimension, req_type, 0>,
                 xpu>::Launch(ctx.get_stream<xpu>(),
                              oshape.Size(),
                              out_data.dptr<DType>(),
                              op,
                              oshape.get<dimension>(),
                              ostride.get<dimension>(),
                              reduceshape.get<dimension>(),
                              reducestride.get<dimension>(),
                              itershape.get<dimension>(),
                              iterstride_arr,
                              nop,
                              -1,
                              reinterpret_cast<DType*>(NULL));
        })
      })
    })
  } else {
    if (oshape.Size() == 0) {
      for (iop = 0; iop < nop; ++iop) {
        const TBlob& out_data = outputs[iop];
        if (opshape[iop].Size() > 0) {
          MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
            MXNET_ASSIGN_REQ_SWITCH(req[iop], req_type, {
              if (req_type == kWriteTo) {
                out_data.FlatTo1D<xpu, DType>(ctx.get_stream<xpu>()) = 0;
              }
            })
          })
        }
      }
      return;
    }
    for (int i = 0; i < nop; ++i) {
      const TBlob &out_data = outputs[i];
      const TBlob &out_grad = inputs[0];
      MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
        mxnet::common::StaticArray<DType*, NPY_MAXARGS> op;
        for (iop = 0; iop < nop; ++iop) {
          op[iop] = inputs[iop + back].dptr<DType>();
        }
        MXNET_ASSIGN_REQ_SWITCH(req[i], req_type, {
          MXNET_NDIM_SWITCH(ndim_iter, dimension, {
            mxnet::common::StaticArray<mshadow::Shape<dimension>, NPY_MAXARGS> iterstride_arr;
            for (iop = 0; iop <= nop; ++iop) {
              iterstride_arr[iop] = iterstride[iop].get<dimension>();
            }
            Kernel<numpy_einsum<dimension, req_type, 1>,
                  xpu>::Launch(ctx.get_stream<xpu>(),
                              opshape[i].Size(),
                              out_data.dptr<DType>(),
                              op,
                              opshape[i].get<dimension>(),
                              opstride[i].get<dimension>(),
                              remainshape[i].get<dimension>(),
                              remainstride[i].get<dimension>(),
                              itershape.get<dimension>(),
                              iterstride_arr,
                              nop,
                              i,
                              out_grad.dptr<DType>());
          })
        })
      })
    }
  }
}

struct NumpyEinsumParam: public dmlc::Parameter<NumpyEinsumParam> {
  int num_args;
  std::string  subscripts;
  DMLC_DECLARE_PARAMETER(NumpyEinsumParam) {
    DMLC_DECLARE_FIELD(num_args).set_lower_bound(1)
      .describe("Number of input arrays.");
    DMLC_DECLARE_FIELD(subscripts)
      .set_default("")
      .describe("Specifies the subscripts for summation as comma separated list"
      " of subscript labels. An implicit (classical Einstein summation) calculation"
      " is performed unless the explicit indicator ‘->’ is included as well as"
      " subscript labels of the precise output form.");
  }
};

template<typename xpu>
inline void NumpyEinsumForward(const nnvm::NodeAttrs& attrs,
                               const OpContext& ctx,
                               const std::vector<TBlob>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  const NumpyEinsumParam &param = nnvm::get<NumpyEinsumParam>(attrs.parsed);
  int num_args = param.num_args;
  const char* subscripts = param.subscripts.c_str();
  CHECK_EQ(inputs.size(), num_args);
  CHECK_EQ(outputs.size(), 1U);
  NumpyEinsumProcess<xpu, 0>(inputs, req, outputs, subscripts, num_args, ctx);
}


template<typename xpu>
inline void NumpyEinsumBackward(const nnvm::NodeAttrs& attrs,
                                const OpContext& ctx,
                                const std::vector<TBlob>& inputs,
                                const std::vector<OpReqType>& req,
                                const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow_op;
  const NumpyEinsumParam &param = nnvm::get<NumpyEinsumParam>(attrs.parsed);
  int num_args = param.num_args;
  const char* subscripts = param.subscripts.c_str();
  CHECK_EQ(inputs.size(), 1 + num_args);
  CHECK_EQ(outputs.size(), num_args);
  NumpyEinsumProcess<xpu, 1>(inputs, req, outputs, subscripts, num_args, ctx);
}


}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_EINSUM_OP_INL_H_
