# heartwise_statplots/utils/type_check.py

import inspect
from typing import Union, Tuple, Type, Callable, Any, Optional, Dict, get_origin, get_args
import numpy as np
import functools
from collections.abc import Callable as ABC_Callable  # Import Callable from collections.abc


class TypeCheckError(TypeError):
    def __init__(
        self,
        arg_name: str,
        expected_type: Union[type, tuple, str],
        actual_type: type,
        additional_info: str = "",
    ):
        expected_type_str = self._get_type_name(expected_type)
        message = f"Argument '{arg_name}' must be of type {expected_type_str}, not {actual_type.__name__}"
        if additional_info:
            message += f". {additional_info}"
        super().__init__(message)

    def _get_type_name(self, expected_type: Union[type, tuple, str]) -> str:
        if isinstance(expected_type, str):
            return expected_type
        elif isinstance(expected_type, tuple):
            return " or ".join([self._get_type_name(t) for t in expected_type])
        else:
            origin = get_origin(expected_type)
            if origin is Union:
                args = get_args(expected_type)
                return " or ".join([self._get_type_name(arg) for arg in args])
            elif origin is ABC_Callable:  # Updated check
                return "Callable"
            elif hasattr(expected_type, "__name__"):
                return expected_type.__name__
            else:
                return str(expected_type)

    @staticmethod
    def _get_type_name_static(expected_type: Union[type, tuple, str]) -> str:
        if isinstance(expected_type, str):
            return expected_type
        elif isinstance(expected_type, tuple):
            return " or ".join([TypeCheckError._get_type_name_static(t) for t in expected_type])
        else:
            origin = get_origin(expected_type)
            if origin is Union:
                args = get_args(expected_type)
                return " or ".join([TypeCheckError._get_type_name_static(arg) for arg in args])
            elif origin is ABC_Callable:  # Updated check
                return "Callable"
            elif hasattr(expected_type, "__name__"):
                return expected_type.__name__
            else:
                return str(expected_type)


def type_check(
    enabled: bool = True,
    dtypes: Optional[Dict[str, Union[np.dtype, type]]] = None,
    **type_hints,
) -> Callable:
    """
    Decorator to enforce type checks on function arguments.

    Args:
        enabled (bool): Flag to enable or disable type checking.
        dtypes (dict, optional): A dictionary specifying expected dtypes
                                 for numpy.ndarray arguments. Example:
                                 {'y_true': np.float64, 'y_pred': np.float64}
        **kwargs: Mapping of argument names to expected types.

    Raises:
        TypeCheckError: If any type check fails.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not enabled:
                return func(*args, **kwargs)

            # Bind the arguments to the function's signature
            bound_args = inspect.signature(func).bind(*args, **kwargs)
            bound_args.apply_defaults()

            for arg_name, expected_type in type_hints.items():
                if arg_name in bound_args.arguments:
                    value = bound_args.arguments[arg_name]

                    # Handle Union types and other generics
                    origin = get_origin(expected_type)
                    if origin is Union:
                        expected_types = get_args(expected_type)
                        if not isinstance(value, expected_types):
                            expected_types_str = " or ".join(
                                [TypeCheckError._get_type_name_static(t) for t in expected_types]
                            )
                            raise TypeCheckError(
                                arg_name,
                                expected_types_str,
                                type(value),
                            )
                    elif origin is ABC_Callable:  # Updated check
                        if not callable(value):
                            raise TypeCheckError(
                                arg_name,
                                "Callable",
                                type(value),
                            )
                    elif isinstance(expected_type, tuple):
                        if not isinstance(value, expected_type):
                            expected_types_str = " or ".join(
                                [TypeCheckError._get_type_name_static(t) for t in expected_type]
                            )
                            raise TypeCheckError(
                                arg_name,
                                expected_types_str,
                                type(value),
                            )
                    elif expected_type == np.ndarray:
                        if not isinstance(value, np.ndarray):
                            raise TypeCheckError(arg_name, "numpy.ndarray", type(value))

                        if dtypes and arg_name in dtypes:
                            expected_dtype = dtypes[arg_name]

                            # Handle tuple of dtypes
                            if isinstance(expected_dtype, tuple):
                                expected_dtypes = []
                                for dtype in expected_dtype:
                                    if isinstance(dtype, type) and issubclass(dtype, np.generic):
                                        expected_dtypes.append(np.dtype(dtype))
                                    elif isinstance(dtype, np.dtype):
                                        expected_dtypes.append(dtype)
                                    else:
                                        raise ValueError(
                                            f"Unsupported dtype specification: {dtype}"
                                        )

                                # Check if value.dtype matches any of the expected dtypes
                                if not any(
                                    np.issubdtype(value.dtype, dt) for dt in expected_dtypes
                                ):
                                    dtype_names = ", ".join(str(dt) for dt in expected_dtypes)
                                    raise TypeCheckError(
                                        arg_name,
                                        f"numpy.ndarray with dtype in ({dtype_names})",
                                        type(value),
                                        f"Array has dtype {value.dtype}",
                                    )
                            else:
                                # Single dtype specification
                                if isinstance(expected_dtype, type) and issubclass(
                                    expected_dtype, np.generic
                                ):
                                    expected_dtype = np.dtype(expected_dtype)
                                if not np.issubdtype(value.dtype, expected_dtype):
                                    raise TypeCheckError(
                                        arg_name,
                                        f"numpy.ndarray with dtype {expected_dtype}",
                                        type(value),
                                        f"Array has dtype {value.dtype}",
                                    )
                    else:
                        if not isinstance(value, expected_type):
                            expected_type_str = TypeCheckError._get_type_name_static(
                                expected_type
                            )
                            raise TypeCheckError(
                                arg_name,
                                expected_type_str,
                                type(value),
                            )
            return func(*args, **kwargs)

        # Set the signature of the original function to preserve introspection
        wrapper.__signature__ = inspect.signature(func)
        return wrapper

    return decorator
