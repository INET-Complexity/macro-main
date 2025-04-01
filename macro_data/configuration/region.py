from macro_data.configuration.countries import Country


class Region(str):
    """A string subclass that represents a region code with additional metadata."""

    def __new__(cls, code: str, parent_country: Country, name: str):
        instance = super().__new__(cls, code)
        instance._parent_country = parent_country
        instance._name = name
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
