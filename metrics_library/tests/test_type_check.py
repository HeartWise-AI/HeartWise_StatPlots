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

# Additional Sample Functions for Testing Multiple dtypes
@type_check(
    enabled=True,
    y_true=np.ndarray,
    y_pred=np.ndarray,
    dtypes={
        "y_true": (np.int64, np.float64),  # Accept both int64 and float64
        "y_pred": np.float64
    },
    metric_func=Callable,
    n_iterations=int,
)
def sample_function(y_true: np.ndarray, y_pred: np.ndarray, metric_func: Callable, n_iterations: int):
    pass 

@type_check(
    enabled=True,
    data=np.ndarray,
    dtypes={
        "data": (np.int32, np.int64, np.float32, np.float64),
    },
)
def another_sample_function(data: np.ndarray):
    pass

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

# Tests for another_sample_function accepting multiple dtypes for data
def test_another_sample_function_valid_dtypes():
    # data as int32
    data_int32 = np.array([1, 2, 3], dtype=np.int32)
    try:
        another_sample_function(data_int32)
    except TypeCheckError:
        pytest.fail("TypeCheckError was raised unexpectedly with valid dtype (int32).")

    # data as int64
    data_int64 = np.array([1, 2, 3], dtype=np.int64)
    try:
        another_sample_function(data_int64)
    except TypeCheckError:
        pytest.fail("TypeCheckError was raised unexpectedly with valid dtype (int64).")

    # data as float32
    data_float32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    try:
        another_sample_function(data_float32)
    except TypeCheckError:
        pytest.fail("TypeCheckError was raised unexpectedly with valid dtype (float32).")

    # data as float64
    data_float64 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    try:
        another_sample_function(data_float64)
    except TypeCheckError:
        pytest.fail("TypeCheckError was raised unexpectedly with valid dtype (float64).")

def test_another_sample_function_invalid_dtypes():
    # data with unsupported dtype (object)
    data_object = np.array([1, '2', 3], dtype=object)
    with pytest.raises(TypeCheckError) as exc_info:
        another_sample_function(data_object)
    assert "data" in str(exc_info.value)
    assert "numpy.ndarray with dtype in (int32, int64, float32, float64)" in str(exc_info.value)

    # data with unsupported dtype (bool)
    data_bool = np.array([True, False, True], dtype=bool)
    with pytest.raises(TypeCheckError) as exc_info:
        another_sample_function(data_bool)
    assert "data" in str(exc_info.value)
    assert "numpy.ndarray with dtype in (int32, int64, float32, float64)" in str(exc_info.value)

# Tests for decorator disabled
@type_check(enabled=False, a=int, b=str)
def no_check(a: int, b: str) -> str:
    return f"{a}{b}"

def test_no_check():
    # Even with incorrect types, no TypeCheckError should be raised
    assert no_check("5", 123) == "5123"