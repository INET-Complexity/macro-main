"""
This module provides country-related configuration and utilities for the macroeconomic model.
It defines country codes, names, and EU membership status, along with conversion utilities
between different country code formats.

The module uses ISO 3166-1 alpha-3 codes for country identification and provides:
- A comprehensive Country enum for all supported countries
- EU country membership information
- Conversion between 2-letter and 3-letter country codes
- Country name mappings

Example:
    ```python
    from macro_data.configuration.countries import Country

    # Create a country instance
    france = Country.FRANCE

    # Check if it's an EU country
    is_eu = france.is_eu_country  # True

    # Convert codes
    two_letter = france.to_two_letter_code()  # 'FR'
    three_letter = Country.convert_two_letter_to_three('FR')  # 'FRA'
    ```
"""

from pathlib import Path

import yaml
from strenum import StrEnum

# get this file's directory path
THIS_FILE_PATH = Path(__file__).parent.resolve()

with open(THIS_FILE_PATH / "3_codes.yaml", "r") as f:
    country_codes = yaml.safe_load(f)

with open(THIS_FILE_PATH / "country_names.yaml", "r") as f:
    country_names = yaml.safe_load(f)

inverse_country_codes = {v: k for k, v in country_codes.items()}


EU_COUNTRIES = [
    "AUT",  # Austria
    "BEL",  # Belgium
    "CZE",  # Czech Republic
    "DNK",  # Denmark
    "FIN",  # Finland
    "FRA",  # France
    "DEU",  # Germany
    "GRC",  # Greece
    "HUN",  # Hungary
    "IRL",  # Ireland
    "ITA",  # Italy
    "LUX",  # Luxembourg
    "NLD",  # Netherlands
    "POL",  # Poland
    "PRT",  # Portugal
    "SVK",  # Slovakia
    "ESP",  # Spain
    "SWE",  # Sweden
    "EST",  # Estonia
    "LVA",  # Latvia
    "SVN",  # Slovenia
    "LTU",  # Lithuania
    "HRV",  # Croatia
    "CYP",  # Cyprus
    "MLT",  # Malta
    "ROU",  # Romania
    "BGR",  # Bulgaria
]


class Country(StrEnum):
    """
    Enumeration of countries using ISO 3166-1 alpha-3 codes.

    This class provides a type-safe way to work with country codes in the model.
    It includes all recognized countries and a special ROW (Rest of World) designation.
    The enum values are the standard 3-letter ISO country codes.

    Special features:
    - Includes EU membership status checking via is_eu_country property
    - Provides conversion between 2-letter and 3-letter codes
    - String representation matches the country code

    Example:
        ```python
        country = Country.FRANCE
        print(country)  # 'FRA'
        print(country.is_eu_country)  # True
        print(country.to_two_letter_code())  # 'FR'
        ```
    """

    FRANCE = "FRA"
    GERMANY = "DEU"
    ITALY = "ITA"
    UNITED_KINGDOM = "GBR"
    AUSTRIA = "AUT"

    UNITED_STATES = "USA"
    CANADA = "CAN"
    JAPAN = "JPN"
    MEXICO = "MEX"

    AFGHANISTAN = "AFG"
    ALBANIA = "ALB"
    ALGERIA = "DZA"
    ANDORRA = "AND"
    ANGOLA = "AGO"
    ANTIGUA_AND_BARBUDA = "ATG"
    ARGENTINA = "ARG"
    ARMENIA = "ARM"
    AUSTRALIA = "AUS"
    AZERBAIJAN = "AZE"
    BAHAMAS = "BHS"
    BAHRAIN = "BHR"
    BANGLADESH = "BGD"
    BARBADOS = "BRB"
    BELARUS = "BLR"
    BELGIUM = "BEL"
    BELIZE = "BLZ"
    BENIN = "BEN"
    BHUTAN = "BTN"
    BOLIVIA = "BOL"
    BOSNIA_AND_HERZEGOVINA = "BIH"
    BOTSWANA = "BWA"
    BRAZIL = "BRA"
    BRUNEI = "BRN"
    BULGARIA = "BGR"
    BURKINA_FASO = "BFA"
    BURUNDI = "BDI"
    CABO_VERDE = "CPV"
    CAMBODIA = "KHM"
    CAMEROON = "CMR"
    CENTRAL_AFRICAN_REPUBLIC = "CAF"
    CHAD = "TCD"
    CHILE = "CHL"
    CHINA = "CHN"
    COLOMBIA = "COL"
    COMOROS = "COM"
    COSTA_RICA = "CRI"
    COTE_DIVOIRE = "CIV"
    CROATIA = "HRV"
    CUBA = "CUB"
    CYPRUS = "CYP"
    CZECH_REPUBLIC = "CZE"
    DEMOCRATIC_REPUBLIC_OF_THE_CONGO = "COD"
    DENMARK = "DNK"
    DJIBOUTI = "DJI"
    DOMINICA = "DMA"
    DOMINICAN_REPUBLIC = "DOM"
    EAST_TIMOR = "TLS"
    ECUADOR = "ECU"
    EGYPT = "EGY"
    EL_SALVADOR = "SLV"
    EQUATORIAL_GUINEA = "GNQ"
    ERITREA = "ERI"
    ESTONIA = "EST"
    ESWATINI = "SWZ"
    ETHIOPIA = "ETH"
    FIJI = "FJI"
    FINLAND = "FIN"
    GABON = "GAB"
    GAMBIA = "GMB"
    GEORGIA = "GEO"
    GHANA = "GHA"
    GREECE = "GRC"
    GRENADA = "GRD"
    GUATEMALA = "GTM"
    GUINEA = "GIN"
    GUINEA_BISSAU = "GNB"
    GUYANA = "GUY"
    HAITI = "HTI"
    HONDURAS = "HND"
    HUNGARY = "HUN"
    ICELAND = "ISL"
    INDIA = "IND"
    INDONESIA = "IDN"
    IRAN = "IRN"
    IRAQ = "IRQ"
    IRELAND = "IRL"
    ISRAEL = "ISR"
    JAMAICA = "JAM"
    JORDAN = "JOR"
    KAZAKHSTAN = "KAZ"
    KENYA = "KEN"
    KIRIBATI = "KIR"
    KOSOVO = "XKX"
    KOREA_NORTH = "PRK"
    KOREA_SOUTH = "KOR"
    KUWAIT = "KWT"
    KYRGYZSTAN = "KGZ"
    LAOS = "LAO"
    LATVIA = "LVA"
    LEBANON = "LBN"
    LESOTHO = "LSO"
    LIBERIA = "LBR"
    LIBYA = "LBY"
    LIECHTENSTEIN = "LIE"
    LITHUANIA = "LTU"
    LUXEMBOURG = "LUX"
    MADAGASCAR = "MDG"
    MALAWI = "MWI"
    MALAYSIA = "MYS"
    MALDIVES = "MDV"
    MALI = "MLI"
    MALTA = "MLT"
    MARSHALL_ISLANDS = "MHL"
    MAURITANIA = "MRT"
    MAURITIUS = "MUS"
    MOLDOVA = "MDA"
    MONACO = "MCO"
    MONGOLIA = "MNG"
    MONTENEGRO = "MNE"
    MOROCCO = "MAR"
    MOZAMBIQUE = "MOZ"
    MYANMAR = "MMR"
    NAMIBIA = "NAM"
    NAURU = "NRU"
    NEPAL = "NPL"
    NETHERLANDS = "NLD"
    NEW_ZEALAND = "NZL"
    NICARAGUA = "NIC"
    NIGER = "NER"
    NIGERIA = "NGA"
    NORTH_MACEDONIA = "MKD"
    NORWAY = "NOR"
    OMAN = "OMN"
    PAKISTAN = "PAK"
    PALAU = "PLW"
    PALESTINE = "PSE"
    PANAMA = "PAN"
    PAPUA_NEW_GUINEA = "PNG"
    PARAGUAY = "PRY"
    PERU = "PER"
    PHILIPPINES = "PHL"
    POLAND = "POL"
    PORTUGAL = "PRT"
    QATAR = "QAT"
    REPUBLIC_OF_THE_CONGO = "COG"
    ROMANIA = "ROU"
    RUSSIA = "RUS"
    RWANDA = "RWA"
    SAINT_KITTS_AND_NEVIS = "KNA"
    SAINT_LUCIA = "LCA"
    SAINT_VINCENT_AND_THE_GRENADINES = "VCT"
    SAMOA = "WSM"
    SAN_MARINO = "SMR"
    SAO_TOME_AND_PRINCIPE = "STP"
    SAUDI_ARABIA = "SAU"
    SENEGAL = "SEN"
    SERBIA = "SRB"
    SEYCHELLES = "SYC"
    SIERRA_LEONE = "SLE"
    SINGAPORE = "SGP"
    SLOVAKIA = "SVK"
    SLOVENIA = "SVN"
    SOLOMON_ISLANDS = "SLB"
    SOMALIA = "SOM"
    SOUTH_AFRICA = "ZAF"
    SOUTH_SUDAN = "SSD"
    SPAIN = "ESP"
    SRI_LANKA = "LKA"
    SUDAN = "SDN"
    SURINAME = "SUR"
    SWEDEN = "SWE"
    SWITZERLAND = "CHE"
    SYRIA = "SYR"
    TAJIKISTAN = "TJK"
    TANZANIA = "TZA"
    THAILAND = "THA"
    TOGO = "TGO"
    TONGA = "TON"
    TRINIDAD_AND_TOBAGO = "TTO"
    TUNISIA = "TUN"
    TURKEY = "TUR"
    TURKMENISTAN = "TKM"
    TUVALU = "TUV"
    UGANDA = "UGA"
    UKRAINE = "UKR"
    UNITED_ARAB_EMIRATES = "ARE"
    URUGUAY = "URY"
    UZBEKISTAN = "UZB"
    VANUATU = "VUT"
    VATICAN_CITY = "VAT"
    VENEZUELA = "VEN"
    VIETNAM = "VNM"
    YEMEN = "YEM"
    ZAMBIA = "ZMB"
    ZIMBABWE = "ZWE"

    REST_OF_WORLD = "ROW"

    def __str__(self):
        """
        Get the string representation of the country code.

        Returns:
            str: The 3-letter country code
        """
        return country_names.get(self.value, self.value)

    def to_two_letter_code(self):
        """
        Convert the country's 3-letter code to its corresponding 2-letter code.

        Returns:
            str: The ISO 3166-1 alpha-2 country code

        Raises:
            KeyError: If the country code mapping is not found
        """
        return country_codes[self.value]

    @staticmethod
    def convert_two_letter_to_three(two_letter_code: str):
        """
        Convert a 2-letter country code to its corresponding 3-letter code.

        Args:
            two_letter_code (str): The ISO 3166-1 alpha-2 country code

        Returns:
            str: The corresponding ISO 3166-1 alpha-3 country code

        Raises:
            KeyError: If the country code mapping is not found
        """
        return inverse_country_codes[two_letter_code]

    @property
    def is_eu_country(self):
        """
        Check if the country is a member of the European Union.

        Returns:
            bool: True if the country is an EU member, False otherwise
        """
        return self.value in EU_COUNTRIES
