# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: skip-file

"""Doc placeholder for numpy ops with prefix _np."""


def _np_ones_like(a):
    """Return an array of ones with the same shape and type as a given array.

    Parameters
    ----------
    a : ndarray
        The shape and data-type of `a` define these same attributes of
        the returned array.

    Returns
    -------
    out : ndarray
        Array of ones with the same shape and type as `a`.
    """
    pass


def _np_zeros_like(a):
    """Return an array of zeros with the same shape and type as a given array.

    Parameters
    ----------
    a : ndarray
        The shape and data-type of `a` define these same attributes of
        the returned array.

    Returns
    -------
    out : ndarray
        Array of zeros with the same shape and type as `a`.
    """
    pass


def _np_cumsum(a, axis=None, dtype=None, out=None):
    """Return the cumulative sum of the elements along a given axis.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int, optional
        Axis along which the cumulative sum is computed. The default
        (None) is to compute the cumsum over the flattened array.
    dtype : dtype, optional
        Type of the returned array and of the accumulator in which the
        elements are summed.  If `dtype` is not specified, it defaults
        to the dtype of `a`, unless `a` has an integer dtype with a
        precision less than that of the default platform integer.  In
        that case, the default platform integer is used.
    out : ndarray, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output
        but the type will be cast if necessary. See `doc.ufuncs`
        (Section "Output arguments") for more details.

    Returns
    -------
    cumsum_along_axis : ndarray.
        A new array holding the result is returned unless `out` is
        specified, in which case a reference to `out` is returned. The
        result has the same size as `a`, and the same shape as `a` if
        `axis` is not None or `a` is a 1-d array.

    Examples
    --------
    >>> a = np.array([[1,2,3], [4,5,6]])
    >>> a
    array([[1, 2, 3],
           [4, 5, 6]])
    >>> np.cumsum(a)
    array([ 1,  3,  6, 10, 15, 21])
    >>> np.cumsum(a, dtype=float)     # specifies type of output value(s)
    array([  1.,   3.,   6.,  10.,  15.,  21.])
    >>> np.cumsum(a,axis=0)      # sum over rows for each of the 3 columns
    array([[1, 2, 3],
           [5, 7, 9]])
    >>> np.cumsum(a,axis=1)      # sum over columns for each of the 2 rows
    array([[ 1,  3,  6],
           [ 4,  9, 15]])

    """
    pass


def _np__linalg_svd(a):
    r"""
    svd(a)

    Singular Value Decomposition.

    When `a` is a 2D array, it is factorized as ``ut @ np.diag(s) @ v``,
    where `ut` and `v` are 2D orthonormal arrays and `s` is a 1D
    array of `a`'s singular values. When `a` is higher-dimensional, SVD is
    applied in stacked mode as explained below.

    Parameters
    ----------
    a : (..., M, N) ndarray 
        A real or complex array with ``a.ndim >= 2`` and ``M <= N``.

    Returns
    -------
    ut: (..., M, M) ndarray
        Orthonormal array(s). The first ``a.ndim - 2`` dimensions have the same
        size as those of the input `a`.
    s : (..., M) ndarray
        Vector(s) with the singular values, within each vector sorted in
        descending order. The first ``a.ndim - 2`` dimensions have the same
        size as those of the input `a`.
    v : (..., M, N) ndarray
        Orthonormal array(s). The first ``a.ndim - 2`` dimensions have the same
        size as those of the input `a`.

    Notes
    -----

    The decomposition is performed using LAPACK routine ``_gesvd``.

    SVD is usually described for the factorization of a 2D matrix :math:`A`.
    The higher-dimensional case will be discussed below. In the 2D case, SVD is
    written as :math:`A = U^T S V`, where :math:`A = a`, :math:`U^T = ut`,
    :math:`S= \mathtt{np.diag}(s)` and :math:`V = v`. The 1D array `s`
    contains the singular values of `a` and `ut` and `v` are orthonormal. The rows
    of `v` are the eigenvectors of :math:`A^T A` and the columns of `ut` are
    the eigenvectors of :math:`A A^T`. In both cases the corresponding
    (possibly non-zero) eigenvalues are given by ``s**2``.

    If `a` has more than two dimensions, then broadcasting rules apply.
    This means that SVD is working in "stacked" mode: it iterates over 
    all indices of the first ``a.ndim - 2`` dimensions and for each
    combination SVD is applied to the last two indices. The matrix `a` 
    can be reconstructed from the decomposition with either 
    ``(ut * s[..., None, :]) @ v`` or
    ``ut @ (s[..., None] * v)``. (The ``@`` operator denotes batch matrix multiplication)

    Examples
    --------
    >>> a = np.arange(54).reshape(6, 9)
    >>> ut, s, v = np.linalg.svd(a)
    >>> ut.shape, s.shape, v.shape
    ((6, 6), (6,), (6, 9))
    >>> s = s.reshape(6, 1)
    >>> ret = np.dot(ut, s * v)
    >>> (ret - a > 1e-3).sum()
    array(0.)
    >>> (ret - a < -1e-3).sum()
    array(0.)
    """
    pass
