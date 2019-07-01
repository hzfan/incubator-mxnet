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


def _np_trace(a, offset=0, axis1=0, axis2=1, out=None):
	    """trace(a, offset=0, axis1=0, axis2=1, out=None)

	    Return the sum along diagonals of the array.

	    If `a` is 2-D, the sum along its diagonal with the given offset
	    is returned, i.e., the sum of elements ``a[i,i+offset]`` for all i.

	    If `a` has more than two dimensions, then the axes specified by axis1 and
	    axis2 are used to determine the 2-D sub-arrays whose traces are returned.
	    The shape of the resulting array is the same as that of `a` with `axis1`
	    and `axis2` removed.

	    Parameters
	    ----------
	    a : ndarray
	        Input array, from which the diagonals are taken.
	    offset : int, optional
	        Offset of the diagonal from the main diagonal. Can be both positive
	        and negative. Defaults to 0.
	    axis1, axis2 : int, optional
	        Axes to be used as the first and second axis of the 2-D sub-arrays
	        from which the diagonals should be taken. Defaults are the first two
	        axes of `a`.
	    out : ndarray, optional
	        Array into which the output is placed. It must be of the right shape
	        and right type to hold the output.

	    Returns
	    -------
	    sum_along_diagonals : ndarray
	        If `a` is 2-D, the sum along the diagonal is returned.  If `a` has
	        larger dimensions, then an array of sums along diagonals is returned.

	    Examples
	    --------
	    >>> a = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
	    >>> np.trace(a)
	    array(3.)
	    >>> a = np.arange(8).reshape((2, 2, 2))
	    >>> np.trace(a)
	    array([6., 8.])
	    >>> a = np.arange(24).reshape((2, 2, 2, 3))
	    >>> np.trace(a).shape
	    (2, 3)
	    """
	    pass
