# %%
import os

import pandas as pd

# %%

files = sorted(
    os.listdir(
        "/Users/jmoran/Projects/macrocosm/macromodel/macro-main/tests/test_macro_data/unit/sample_raw_data/hfcs/2014"
    )
)

hfcs_data = [
    pd.read_csv(
        f"/Users/jmoran/Projects/macrocosm/macromodel/macro-main/tests/test_macro_data/unit/sample_raw_data/hfcs/2014/{file}"
    )
    for file in files
]

# %%
df = hfcs_data[0]
df.head(5)
# %%
