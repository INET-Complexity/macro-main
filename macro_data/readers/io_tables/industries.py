"""
This module defines the industry classifications used in the OECD Inter-Country Input Output (ICIO) tables.
It provides two main industry groupings that enable analysis at different levels of granularity.

Key Features:
1. AGGREGATED_INDUSTRIES: High-level sector groupings (e.g., 'C' for all manufacturing)
2. ALL_INDUSTRIES: Detailed industry codes maintaining full granularity

Industry Classification System:
The industry codes follow the ISIC Rev. 4 classification system:

Primary Sector:
- A: Agriculture, forestry and fishing
  * A01: Crop and animal production
  * A03: Fishing and aquaculture
- B: Mining and quarrying
  * B05: Coal mining
  * B07: Metal ores mining
  * B09: Mining support activities

Secondary Sector:
- C: Manufacturing
  * C10T12: Food, beverages and tobacco
  * C13T15: Textiles, wearing apparel, leather
  * C16: Wood products
  * C17: Paper products
  * C19: Petroleum products
  * C20: Chemical products
  * C21: Pharmaceutical products
  * C22: Rubber and plastic products
  * C23: Non-metallic mineral products
  * C24: Basic metals
  * C25: Fabricated metal products
  * C26: Computer, electronic and optical
  * C27: Electrical equipment
  * C28: Machinery and equipment n.e.c.
  * C29: Motor vehicles
  * C30: Other transport equipment
  * C31T33: Furniture, other manufacturing, repair
- D: Electricity, gas, steam and air conditioning
- E: Water supply, sewerage, waste management
- F: Construction

Tertiary Sector:
- G: Wholesale and retail trade
- H: Transportation and storage
  * H49: Land transport
  * H50: Water transport
  * H51: Air transport
  * H52: Warehousing
  * H53: Postal and courier activities
- I: Accommodation and food services
- J: Information and communication
  * J58T60: Publishing and broadcasting
  * J61: Telecommunications
  * J62: IT and information services
- K: Financial and insurance activities
- L: Real estate activities
- M: Professional, scientific and technical
- N: Administrative and support services
- O: Public administration and defence
- P: Education
- Q: Human health and social work
- R_S: Arts, entertainment, recreation, other services

Usage Examples:
    ```python
    from macro_data.readers.io_tables.industries import AGGREGATED_INDUSTRIES, ALL_INDUSTRIES

    # High-level sector analysis
    manufacturing = [ind for ind in AGGREGATED_INDUSTRIES if ind == 'C']
    services = [ind for ind in AGGREGATED_INDUSTRIES if ind in ['G', 'H', 'I', 'J', 'K']]

    # Detailed industry analysis
    chemical_sector = [ind for ind in ALL_INDUSTRIES if ind in ['C20', 'C21']]
    transport = [ind for ind in ALL_INDUSTRIES if ind.startswith('H')]
    ```

Notes:
    - Industry codes use 'T' to indicate ranges (e.g., C10T12 = C10-C12)
    - Some sectors (D, E, F, G, etc.) don't have detailed subsectors
    - R_S combines arts, entertainment, and other services
"""

# High-level industry codes grouped by primary sector
AGGREGATED_INDUSTRIES = [
    "A",  # Agriculture, forestry and fishing
    "B",  # Mining and quarrying
    "C",  # Manufacturing (includes C10T12 through C31T33)
    "D",  # Electricity, gas, steam and air conditioning supply
    "E",  # Water supply; sewerage, waste management
    "F",  # Construction
    "G",  # Wholesale and retail trade
    "H",  # Transportation and storage (includes H49-H53)
    "I",  # Accommodation and food service activities
    "J",  # Information and communication (includes J58T60-J62)
    "K",  # Financial and insurance activities
    "L",  # Real estate activities
    "M",  # Professional, scientific and technical activities
    "N",  # Administrative and support service activities
    "O",  # Public administration and defence
    "P",  # Education
    "Q",  # Human health and social work activities
    "R_S",  # Arts, entertainment, recreation and other services
]

# Detailed industry codes with full ISIC Rev. 4 granularity
ALL_INDUSTRIES = [
    "A01",
    "A03",
    "B05",
    "B07",
    "B09",
    "C10T12",
    "C13T15",
    "C16",
    "C17",
    "C19",
    "C20",
    "C21",
    "C22",
    "C23",
    "C24",
    "C25",
    "C26",
    "C27",
    "C28",
    "C29",
    "C30",
    "C31T33",
    "D",
    "E",
    "F",
    "G",
    "H49",
    "H50",
    "H51",
    "H52",
    "H53",
    "I",
    "J58T60",
    "J61",
    "J62",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R_S",
]
