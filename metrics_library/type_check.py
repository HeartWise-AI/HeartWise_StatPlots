# metrics_library/type_check.py
import inspect
from typing import Union, Tuple, Type, Callable, Any, Optional, Dict
import numpy as np
import functools

class TypeCheckError(TypeError):
    def __init__(self, arg_name: str, expected_type: Union[type, tuple, str], actual_type: type, additional_info: str = ""):
        message = f"Argument '{arg_name}' must be of type {expected_type}, not {actual_type.__name__}"
        if additional_info:
            message += f". {additional_info}"
        super().__init__(message)

def type_check(enabled: bool = True, dtypes: Optional[Dict[str, Union[np.dtype, type]]] = None, **type_hints) -> Callable:
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not enabled:
                return func(*args, **kwargs)

            bound_args = func.__signature__.bind(*args, **kwargs)
            bound_args.apply_defaults()

            for arg_name, expected_type in type_hints.items():
                if arg_name in bound_args.arguments:
                    value = bound_args.arguments[arg_name]
                    
                    if isinstance(expected_type, tuple):
                        if not isinstance(value, expected_type):
                            raise TypeCheckError(arg_name, " or ".join(t.__name__ for t in expected_type), type(value))
                    elif expected_type == np.ndarray:
                        if not isinstance(value, np.ndarray):
                            raise TypeCheckError(arg_name, "numpy.ndarray", type(value))
                        
                        if dtypes and arg_name in dtypes:
                            expected_dtype = dtypes[arg_name]
                            if isinstance(expected_dtype, type) and issubclass(expected_dtype, np.generic):
                                expected_dtype = np.dtype(expected_dtype)
                            if not np.issubdtype(value.dtype, expected_dtype):
                                raise TypeCheckError(
                                    arg_name,
                                    f"numpy.ndarray with dtype {expected_dtype}",
                                    type(value),
                                    f"Array has dtype {value.dtype}"
                                )
                    elif not isinstance(value, expected_type):
                        raise TypeCheckError(arg_name, expected_type.__name__, type(value))

            return func(*args, **kwargs)
        
        func.__signature__ = inspect.signature(func)
        return wrapper
    return decorator