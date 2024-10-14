# metrics_library/tests/test_type_check.py

import pytest
import numpy as np
from metrics_library.type_check import type_check, TypeCheckError
from typing import Any, Callable

# Sample functions to test the decorator
@type_check(enabled=True, a=int, b=str)
def concatenate(a: int, b: str) -> str:
    return f"{a}{b}"

@type_check(enabled=True, dtypes={'array': np.float64}, array=np.ndarray)
def process_array(array: np.ndarray) -> float:
    return np.mean(array)

@type_check(enabled=True, flag=bool)
def toggle(flag: bool) -> bool:
    return not flag

@type_check(enabled=True, custom=Callable)
def execute(custom: Callable) -> Any:
    return custom()

@type_check(enabled=True, a=np.ndarray, b=np.ndarray, dtypes={"a": np.int64, "b": np.float64})
def combine_arrays(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a + b

# Tests for correct type usage
def test_concatenate_success():
    result = concatenate(5, "test")
    assert result == "5test"

def test_process_array_success():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    result = process_array(arr)
    assert result == 2.0

def test_toggle_success():
    assert toggle(True) is False
    assert toggle(False) is True

def test_execute_success():
    def sample_func():
        return "Executed"
    assert execute(sample_func) == "Executed"

def test_combine_arrays_success():
    a = np.array([1, 2, 3], dtype=np.int64)
    b = np.array([0.5, 1.5, 2.5], dtype=np.float64)
    result = combine_arrays(a, b)
    expected = np.array([1.5, 3.5, 5.5])
    np.testing.assert_array_almost_equal(result, expected)

# Tests for incorrect type usage
def test_concatenate_failure():
    with pytest.raises(TypeCheckError) as exc_info:
        concatenate("5", "test")  # a should be int
    assert "Argument 'a' must be of type int, not str" in str(exc_info.value)

    with pytest.raises(TypeCheckError) as exc_info:
        concatenate(5, 123)  # b should be str
    assert "Argument 'b' must be of type str" in str(exc_info.value)

def test_process_array_failure():
    with pytest.raises(TypeCheckError) as exc_info:
        process_array("not an array")  # array should be NDArray[np.float64]
    assert "Argument 'array' must be of type numpy.ndarray" in str(exc_info.value)

    arr_wrong_dtype = np.array([1, 2, 3], dtype=np.int32)
    with pytest.raises(TypeCheckError) as exc_info:
        process_array(arr_wrong_dtype)
    assert "Argument 'array' must be of type numpy.ndarray with dtype float64, not ndarray" in str(exc_info.value)

def test_toggle_failure():
    with pytest.raises(TypeCheckError) as exc_info:
        toggle("True")  # flag should be bool
    assert "Argument 'flag' must be of type bool" in str(exc_info.value)

def test_execute_failure():
    with pytest.raises(TypeCheckError) as exc_info:
        execute("not a Callable")  # custom should be callable
    assert "Argument 'custom' must be of type Callable, not str" in str(exc_info.value)

def test_combine_arrays_failure():
    # Test for incorrect type (not an array)
    with pytest.raises(TypeCheckError) as exc_info:
        combine_arrays("not an array", np.array([0.5, 1.5, 2.5], dtype=np.float64))
    assert "Argument 'a' must be of type numpy.ndarray, not str" in str(exc_info.value)

    # Test for incorrect dtype
    a_wrong_dtype = np.array([1, 2, 3], dtype=np.int32)  # Using int32 instead of int64
    b = np.array([0.5, 1.5, 2.5], dtype=np.float64)
    with pytest.raises(TypeCheckError) as exc_info:
        combine_arrays(a_wrong_dtype, b)
    assert "Argument 'a' must be of type numpy.ndarray with dtype int64, not ndarray" in str(exc_info.value)

    # Test for incorrect dtype of second array
    a_correct = np.array([1, 2, 3], dtype=np.int64)
    b_wrong_dtype = np.array([1, 2, 3], dtype=np.int64)  # Should be float64
    with pytest.raises(TypeCheckError) as exc_info:
        combine_arrays(a_correct, b_wrong_dtype)
    assert "Argument 'b' must be of type numpy.ndarray with dtype float64, not ndarray" in str(exc_info.value)

# Tests for decorator disabled
@type_check(enabled=False, a=int, b=str)
def no_check(a: int, b: str) -> str:
    return f"{a}{b}"

def test_no_check():
    # Even with incorrect types, no TypeCheckError should be raised
    assert no_check("5", 123) == "5123"