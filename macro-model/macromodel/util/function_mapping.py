from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel


def get_functions(
    functions_desc: Any,
    loc: str,
    func_dir: Path,
) -> dict[str, Any]:
    functions = {}
    func_dirs = list(func_dir.glob("*"))
    func_names = [fd.stem for fd in func_dirs if fd.name not in ["__init__.py", "__pycache__"]]
    for function_desc in func_names:
        if "lib" in function_desc:
            continue

        # Check if the function exists in the config file
        if function_desc not in functions_desc:
            raise ValueError(function_desc + " not defined in the config file for " + loc)

        cls_name = functions_desc[function_desc]["name"]["value"]
        module = __import__(loc + ".func." + function_desc, fromlist=[cls_name])
        cls = getattr(module, cls_name)

        # Create an object
        if "parameters" not in functions_desc[function_desc] or functions_desc[function_desc]["parameters"] is None:
            functions[function_desc] = cls()
        else:
            func_parameters = {
                k: functions_desc[function_desc]["parameters"][k]["value"]
                for k in functions_desc[function_desc]["parameters"].keys()
            }
            functions[function_desc] = cls(**func_parameters)

    return functions


def functions_from_model(model: BaseModel, loc: str) -> dict[str, Any]:
    loaded_classes = {}
    for field_name, field_value in model:
        path_name = field_value.path_name
        name = field_value.name
        parameters = field_value.parameters

        module = __import__(f"{loc}.func.{path_name}", fromlist=[name])
        cls = getattr(module, name)

        loaded_classes[path_name] = cls(**parameters)

    return loaded_classes


def update_functions(
    model: BaseModel, loc: str, functions: dict[str, Any], force_reset: Optional[list[str]] = None
) -> None:
    if force_reset is None:
        force_reset = []
    for func_name, new_func_config in model.__dict__.items():
        existing_func = functions.get(func_name, None)
        if existing_func is None:
            raise ValueError(f"Function {func_name} not found in functions dictionary")

        # Check if function needs to be reinstantiated
        if (
            existing_func is None
            or func_name in force_reset
            or existing_func.__class__.__name__ != new_func_config.name  # noqa
        ):
            # Different class or no existing function, or we want to force rest, so we
            # need to reinstantiate
            module = __import__(f"{loc}.func.{new_func_config.path_name}", fromlist=[new_func_config.name])
            cls = getattr(module, new_func_config.name)
            # setattr(functions, func_name, cls(**new_func_config.parameters))
            functions[func_name] = cls(**new_func_config.parameters)
        else:
            # Same class, just update parameters
            for param, value in new_func_config.parameters.items():
                setattr(existing_func, param, value)
