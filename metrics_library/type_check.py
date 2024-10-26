import functools
import inspect
from typing import Callable, Any, Optional, Dict, Union
import numpy as np
from typing import get_origin, get_args, Union as TypingUnion

class TypeCheckError(Exception):
    def __init__(self, func_name: str, arg_name: str, expected_type: Any, actual_type: type, additional_info: Optional[str] = None):
        self.func_name = func_name
        self.arg_name = arg_name
        self.expected_type = self._get_friendly_type_name(expected_type)
        self.actual_type = self._get_friendly_type_name(actual_type)
        self.additional_info = additional_info
        message = f"Argument '{arg_name}' must be of type {expected_type}, not {actual_type.__name__}"
        if additional_info:
            message += f" {additional_info}"
        super().__init__(message)
        
    def _get_friendly_type_name(self, t: Any) -> str:
        """
        Generates a user-friendly type name.

        Args:
            t (Any): The type to be converted to a friendly name.

        Returns:
            str: A user-friendly name of the type.
        """
        # Handle types from the typing module
        if hasattr(t, '__origin__'):
            origin = t.__origin__
            if origin is Callable:
                return 'Callable'
            elif origin is Union:
                return 'Union'
            # Add more conditions here if needed
            else:
                return origin.__name__
        elif isinstance(t, type):
            return t.__name__
        else:
            return str(t)        

def isinstance_of_type(value: Any, expected_type: Any) -> bool:
    origin = get_origin(expected_type)
    args = get_args(expected_type)

    if origin is TypingUnion:
        return any(isinstance_of_type(value, arg) for arg in args)

    if origin is Callable:
        return callable(value)

    if origin is list:
        if not isinstance(value, list):
            return False
        if not args:
            return True  # No subtype specified
        return all(isinstance_of_type(item, args[0]) for item in value)

    if origin is dict:
        if not isinstance(value, dict):
            return False
        key_type, val_type = args
        return all(isinstance_of_type(k, key_type) and isinstance_of_type(v, val_type) for k, v in value.items())

    # Add more type checks as needed

    return isinstance(value, expected_type)

def type_check(enabled: bool = True, dtypes: Optional[Dict[str, Union[np.dtype, type]]] = None, **type_hints) -> Callable:
    """
    Decorator to enforce type checks on function arguments.

    Args:
        enabled (bool): Flag to enable or disable type checking.
        dtypes (dict, optional): A dictionary specifying expected dtypes
                                  for numpy.ndarray arguments. Example:
                                  {'y_true': np.float64, 'y_pred': np.float64}
        **type_hints: Mapping of argument names to expected types.

    Raises:
        TypeCheckError: If any type check fails.
    """    
    def decorator(func: Callable) -> Callable:
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not enabled:
                return func(*args, **kwargs)

            try:
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
            except TypeError as e:
                raise TypeError(f"In function '{func.__name__}': {e}")

            for arg_name, expected_type in type_hints.items():
                if arg_name in bound_args.arguments:
                    value = bound_args.arguments[arg_name]

                    if expected_type in (np.ndarray,):
                        if not isinstance(value, np.ndarray):
                            raise TypeCheckError(
                                func.__name__,
                                arg_name,
                                "numpy.ndarray",
                                type(value)
                            )

                        if dtypes and arg_name in dtypes:
                            expected_dtype = dtypes[arg_name]

                            # Handle tuple of dtypes
                            if isinstance(expected_dtype, tuple):
                                # Convert all dtype specifications to np.dtype
                                expected_dtypes = []
                                for dtype in expected_dtype:
                                    if isinstance(dtype, type) and issubclass(dtype, np.generic):
                                        expected_dtypes.append(np.dtype(dtype))
                                    elif isinstance(dtype, np.dtype):
                                        expected_dtypes.append(dtype)
                                    else:
                                        raise ValueError(f"Unsupported dtype specification: {dtype}")

                                # Check if value.dtype matches any of the expected dtypes
                                if not any(np.issubdtype(value.dtype, dt) for dt in expected_dtypes):
                                    dtype_names = ", ".join(str(dt) for dt in expected_dtypes)
                                    raise TypeCheckError(
                                        func.__name__,
                                        arg_name,
                                        f"numpy.ndarray with dtype in ({dtype_names})",
                                        type(value),
                                        f"Array has dtype {value.dtype}"
                                    )
                            else:
                                # Single dtype specification
                                if isinstance(expected_dtype, type) and issubclass(expected_dtype, np.generic):
                                    expected_dtype = np.dtype(expected_dtype)
                                elif isinstance(expected_dtype, np.dtype):
                                    pass
                                else:
                                    raise ValueError(f"Unsupported dtype specification: {expected_dtype}")

                                if not np.issubdtype(value.dtype, expected_dtype):
                                    raise TypeCheckError(
                                        func.__name__,
                                        arg_name,
                                        f"numpy.ndarray with dtype {expected_dtype}",
                                        type(value),
                                        f"Array has dtype {value.dtype}"
                                    )
                    else:
                        # Handle expected_type being a tuple of types or single type
                        if isinstance(expected_type, tuple) or get_origin(expected_type) is TypingUnion:
                            if not isinstance_of_type(value, expected_type):
                                if get_origin(expected_type) is TypingUnion:
                                    expected_type_str = " or ".join(getattr(arg, '__name__', str(arg)) for arg in get_args(expected_type))
                                else:
                                    expected_type_str = " or ".join(t.__name__ if hasattr(t, '__name__') else str(t) for t in expected_type)
                                raise TypeCheckError(
                                    func.__name__,
                                    arg_name,
                                    expected_type_str,
                                    type(value)
                                )
                        else:
                            if not isinstance_of_type(value, expected_type):
                                expected_type_str = getattr(expected_type, '__name__', str(expected_type))
                                actual_type_str = getattr(type(value), '__name__', str(type(value)))
                                raise TypeCheckError(
                                    func.__name__,
                                    arg_name,
                                    expected_type_str,
                                    type(value)
                                )

            return func(*args, **kwargs)

        return wrapper
    return decorator