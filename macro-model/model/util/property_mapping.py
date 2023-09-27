import numpy as np

from enum import Enum, EnumMeta


def map_to_enum(values: np.ndarray, property_enum: EnumMeta) -> np.ndarray[Enum]:
    enum_map = {p.value: p for p in property_enum}
    return list(map(enum_map.get, values))
