#include <pybind11/pybind11.h>
#include <mxnet/tuple.h>

namespace py = pybind11;

namespace pybind11 { 
namespace detail {
    template <> struct type_caster<TShape> {
    public:
      PYBIND11_TYPE_CASTER(Tuple, _("TShape"));
      bool load(handle src, bool) {
        PyObject *source = src.ptr();
        Py_ssize_t size = PySequence_Length(source);
        value = TShape(size, 0);
        for (Py_ssize_t i = 0; i < size; ++i) {
          PyObject *item = PySequence_GetItem(source, i);
          value[i] = handle(item).cast<dim_t>();
          Py_DECREF(item);
        }
      }
    };
    // }
    // template <> struct type_caster<inty> {
    // public:
    //     /**
    //      * This macro establishes the name 'inty' in
    //      * function signatures and declares a local variable
    //      * 'value' of type inty
    //      */
    //     PYBIND11_TYPE_CASTER(inty, _("inty"));

    //     /**
    //      * Conversion part 1 (Python->C++): convert a PyObject into a inty
    //      * instance or return false upon failure. The second argument
    //      * indicates whether implicit conversions should be applied.
    //      */
    //     bool load(handle src, bool) {
    //         /* Extract PyObject from handle */
    //         PyObject *source = src.ptr();
    //         /* Try converting into a Python integer value */
    //         PyObject *tmp = PyNumber_Long(source);
    //         if (!tmp)
    //             return false;
    //         /* Now try to convert into a C++ int */
    //         value.long_value = PyLong_AsLong(tmp);
    //         Py_DECREF(tmp);
    //         /* Ensure return code was OK (to avoid out-of-range errors etc) */
    //         return !(value.long_value == -1 && !PyErr_Occurred());
    //     }

    //     /**
    //      * Conversion part 2 (C++ -> Python): convert an inty instance into
    //      * a Python object. The second and third arguments are used to
    //      * indicate the return value policy and parent object (for
    //      * ``return_value_policy::reference_internal``) and are generally
    //      * ignored by implicit casters.
    //      */
    //     static handle cast(inty src, return_value_policy /* policy */, handle /* parent */) {
    //         return PyLong_FromLong(src.long_value);
    //     }
    // };

    // template<>
    // template<typename ValueType>
    // struct type_caster<Tuple<ValueType> > {
    //   public:
    //   PYBIND11_TYPE_CASTER(Tuple<ValueType>, _("Tuple<ValueType>"));
    //   bool load(handle src, bool) {
    //     PyObject *source = src.ptr();
    //     Py_ssize_t size = PySequence_Length(source);
    //     value.vec.reserve(size);
    //     for (Py_ssize_t i = 0; i < size; ++i) {
    //       PyObject *item = PySequence_GetItem(source, i);
    //       value.vec.emplace_back(handle(item).cast<ValueType>());
    //       Py_DECREF(item);
    //     }
    //     return true;
    //   }
    // };
} // namespace pybind11
} // namespace detail