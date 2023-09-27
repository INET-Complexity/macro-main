from pathlib import Path

from typing import Any


def get_functions(
    functions_desc: dict[str, Any],
    loc: str,
    func_dir: Path,
) -> dict[str, Any]:
    functions = {}
    func_dirs = list(func_dir.glob("*"))
    func_names = [fd.name[:-3] for fd in func_dirs if fd.name not in ["__init__.py", "__pycache__"]]
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
