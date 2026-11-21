from dataclasses import dataclass

import numpy as np

from macro_data.configuration.countries import Country as CountryName
from macro_data.configuration.region import Region as RegionName
from macromodel.country import Country


@dataclass
class RegionalAggregator:
    aggregation_structure: dict[CountryName, list[CountryName | RegionName]]

    def sync_central_banks(self, countries: dict[str, Country]):
        """Synchronize the central banks of the countries in the aggregation structure."""
        for country, regions in self.aggregation_structure.items():
            outputs = np.array([countries[region].economy.ts.current("total_output") for region in regions])
            weights = outputs / np.sum(outputs)

            inflation_rates = np.array([countries[region].economy.ts.current("ppi_inflation")[0] for region in regions])
            total_growth_rates = np.array(
                [countries[region].economy.ts.current("total_growth")[0] for region in regions]
            )

            avg_inflation = np.sum(weights * inflation_rates)
            avg_growth = np.sum(weights * total_growth_rates)

            policy_rate = countries[regions[0]].central_bank.compute_rate(inflation=avg_inflation, growth=avg_growth)

            for region in regions:
                countries[region].central_bank.ts.override_current("policy_rate", [policy_rate])
