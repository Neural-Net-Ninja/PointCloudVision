import numpy as np
from typing import Any


def get_max_dtype(dtype1: np.dtype, dtype2: Any) -> np.dtype:
    """
    Returns the dtype with the highest priority between two dtypes.

    :param dtype1: First data type object
    :type dtype1: numpy.dtype
    :param dtype2: Second data type object
    :type dtype2: Any
    :return: Data type object with the highest priority
    :rtype: numpy.dtype
    """
    dtype_hierarchy = {
        np.dtype('int32'): 1,
        np.dtype('int64'): 2,
        np.dtype('float32'): 3,
        np.dtype('float64'): 4,
        np.dtype('object'): 5
    }

    priority1 = dtype_hierarchy.get(dtype1, 0)
    priority2 = dtype_hierarchy.get(dtype2, 0)

    return dtype1 if priority1 >= priority2 else dtype2


def test_get_max_dtype(self):
    """Tests `get_max_dtype` method. Returns the dtype with the highest priority."""
    self.assertEqual(get_max_dtype(np.dtype('int32'), np.dtype('int64')), np.dtype('int64'))
    self.assertEqual(get_max_dtype(np.dtype('float32'), np.dtype('int32')), np.dtype('float32'))
    self.assertEqual(get_max_dtype(np.dtype('float64'), np.dtype('float32')), np.dtype('float64'))
    self.assertEqual(get_max_dtype(np.dtype('object'), np.dtype('float64')), np.dtype('object'))
    self.assertEqual(get_max_dtype(np.dtype('int32'), np.dtype('int32')), np.dtype('int32'))