from typing import Any

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

from macro_data.configuration.countries import Country


class Region(str):
    """A string subclass that represents a region code with additional metadata."""

    def __new__(cls, code: str, parent_country: Country, name: str, va_ratio: float = 1.0):
        instance = super().__new__(cls, code)
        instance._parent_country = parent_country
        instance._name = name
        instance._va_ratio = va_ratio
        return instance

    @classmethod
    def from_code(cls, extended_code: str, name: str | None = None):
        """Create a region from a code."""
        parent_country = extended_code.split("_")[0]
        name = name or extended_code
        return cls(extended_code, Country(parent_country), name)

    @property
    def parent_country(self) -> Country:
        return self._parent_country

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_eu_country(self) -> bool:
        return self._parent_country.is_eu_country

    @property
    def va_ratio(self) -> float:
        return self._va_ratio

    @va_ratio.setter
    def va_ratio(self, value: float):
        self._va_ratio = value

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        """Provide Pydantic schema for the Region class."""
        return core_schema.json_schema(
            core_schema.str_schema(),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda x: str(x),
                return_schema=core_schema.str_schema(),
            ),
        )

    def __getnewargs__(self):
        return str(self), self._parent_country, self._name, self._va_ratio
