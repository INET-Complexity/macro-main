# Contributing to macromodel Package

This guide explains how to contribute to the `macromodel` package, which contains the core simulation engine and economic agent behaviors for the macroeconomic simulation framework.

## Package Overview

The `macromodel` package implements an **agent-based macroeconomic simulation** where economic actors (firms, households, banks, governments) interact through markets over time. It takes processed data from the `macro_data` package and runs dynamic simulations of economic interactions.

## Architecture Overview

The macromodel package follows an agent-based architecture with these key components:

### Core Components

- **Agents**: Economic actors with specific behaviors and data
- **TimeSeries**: High-level wrappers for temporal data management
- **Functions**: Pluggable behavioral implementations
- **Markets**: Mechanisms for clearing transactions between agents
- **Configuration**: Pydantic-based system for behavioral parameters

### Directory Structure

```
macromodel/
├── agents/                    # Economic agent types
│   ├── agent/                # Base agent class
│   ├── firms/                # Firm agents
│   │   ├── func/            # Behavioral functions
│   │   └── firm_ts.py       # Firm timeseries
│   ├── households/           # Household agents
│   ├── banks/               # Bank agents
│   └── governments/         # Government agents
├── markets/                  # Market clearing mechanisms
├── countries/               # Country-level modeling
├── configurations/          # Configuration classes
├── timeseries.py           # Base TimeSeries class
└── simulation.py           # Main simulation orchestration
```

## Agent Architecture

### Agent Initialization Pattern

All agents are initialized using the `from_pickled_agent` class method that takes synthetic data from the `macro_data` package:

```python
@classmethod
def from_pickled_agent(
    cls,
    synthetic_firms: SyntheticFirms,  # From macro_data package
    configuration: FirmsConfiguration,
    country_name: str,
    all_country_names: list[str],
    goods_criticality_matrix: pd.DataFrame | np.ndarray,
    average_initial_price: np.ndarray,
    industries: list[str],
    add_emissions: bool = False,
) -> "Firms":
    """Create firms from synthetic data and configuration."""
    
    # Initialize TimeSeries with synthetic data
    ts = FirmTimeSeries.from_data(synthetic_firms.firm_data, ...)
    
    # Load behavioral functions from configuration
    functions = functions_from_model(configuration.functions, "macromodel.agents.firms")
    
    # Create agent instance
    return cls(
        ts=ts,
        functions=functions,
        configuration=configuration,
        # ... other parameters
    )
```

### Agent Types and Their Roles

- **Firms**: Production, employment, investment, pricing decisions
- **Households**: Consumption, saving, housing, financial decisions
- **Individuals**: Labor supply, wage negotiation, mobility
- **Banks**: Credit supply, interest rates, risk management
- **Central Bank**: Monetary policy, policy rates
- **Central Government**: Fiscal policy, taxation, transfers
- **Government Entities**: Public consumption, regulation
- **Rest of World**: International trade, capital flows

## TimeSeries Objects

### TimeSeries Architecture

The `TimeSeries` class serves as a **high-level wrapper around arrays** that provides temporal data management for agents:

```python
class TimeSeries:
    """High-level wrapper for temporal data management."""
    
    def __init__(self, **kwargs):
        """Initialize with variable names and initial values."""
        self.variables = {}
        for name, initial_value in kwargs.items():
            self.variables[name] = [initial_value]
    
    def current(self, item: str):
        """Get the most recent value."""
        return self.variables[item][-1]
    
    def prev(self, item: str, n: int = 1):
        """Get the value n periods ago."""
        return self.variables[item][-n-1]
    
    def initial(self, item: str):
        """Get the initial value."""
        return self.variables[item][0]
    
    def historic(self, item: str):
        """Get the complete history."""
        return self.variables[item]
```

### Agent-Specific TimeSeries Classes

Each agent type has its own specialized TimeSeries class:

```python
# Example: FirmTimeSeries
class FirmTimeSeries(TimeSeries):
    """Time series data container for tracking firm-level economic variables."""
    
    @classmethod
    def from_data(cls, data: pd.DataFrame, industries: list[str], ...):
        """Create a FirmTimeSeries instance from initial data."""
        return cls(
            production=data["Production"].values,
            price=data["Price"].values,
            inventory=data["Inventory"].values,
            employment=data["Employment"].values,
            debt=data["Debt"].values,
            deposits=data["Deposits"].values,
            # ... dozens of other economic variables
        )
```

### Usage Patterns

Agents use TimeSeries objects to:

```python
# Access current state
current_production = self.ts.current("production")
current_price = self.ts.current("price")

# Access historical data for decision-making
previous_sales = self.ts.prev("sales")
initial_inventory = self.ts.initial("inventory")

# Update state variables
new_production = self.functions["production"].compute_production(...)
self.ts.production.append(new_production)
```

## Function System

### Abstract Function Architecture

The function system implements a **strategy pattern** where different behavioral implementations can be plugged in via configuration:

#### Abstract Base Classes

Each behavioral domain has an abstract base class defining the interface:

```python
# Example: ProductionSetter
class ProductionSetter(ABC):
    """Abstract base class for determining firms' production processes."""
    
    @abstractmethod
    def compute_limiting_intermediate_inputs_stock(
        self, 
        current_intermediate_inputs_stock: np.ndarray,
        intermediate_inputs_use: np.ndarray,
    ) -> np.ndarray:
        """Calculate production possible with available intermediate inputs."""
        pass
    
    @abstractmethod
    def compute_production(
        self,
        desired_production: np.ndarray,
        current_labour_inputs: np.ndarray,
        current_limiting_intermediate_inputs: np.ndarray,
        current_limiting_capital_inputs: np.ndarray,
    ) -> np.ndarray:
        """Calculate actual production levels based on constraints."""
        pass
```

#### Concrete Implementations

Multiple implementations provide different behavioral models:

```python
class PureLeontief(ProductionSetter):
    """Fixed-proportions production function."""
    
    def compute_production(self, ...):
        # Implementation for Leontief production
        return np.minimum.reduce([
            desired_production,
            current_labour_inputs,
            current_limiting_intermediate_inputs,
            current_limiting_capital_inputs
        ])

class CriticalAndImportantLeontief(ProductionSetter):
    """Leontief with critical/non-critical input distinctions."""
    
    def compute_production(self, ...):
        # Implementation with input criticality
        pass

class Linear(ProductionSetter):
    """Linear production with input substitutability."""
    
    def compute_production(self, ...):
        # Implementation with input substitution
        pass
```

#### Function Categories

Each agent type has multiple function categories in its `func/` directory:

**For Firms:**
- `production.py`: How goods are produced
- `prices.py`: How prices are set
- `wage_setter.py`: How wages are determined
- `target_production.py`: How production targets are calculated
- `credit_demand.py`: How borrowing decisions are made
- `demography.py`: How entry/exit occurs

**For Households:**
- `consumption.py`: How consumption decisions are made
- `savings.py`: How savings are allocated
- `housing.py`: How housing decisions are made

### Adding New Functions

To add a new behavioral function:

1. **Create the abstract interface** (if it doesn't exist):

```python
# File: macromodel/agents/firms/func/my_new_function.py
from abc import ABC, abstractmethod

class MyNewFunction(ABC):
    """Abstract base class for new behavioral function."""
    
    @abstractmethod
    def compute_something(self, input_data: np.ndarray) -> np.ndarray:
        """Compute something based on input data."""
        pass
```

2. **Implement concrete versions**:

```python
class SimpleImplementation(MyNewFunction):
    """Simple implementation of the new function."""
    
    def __init__(self, parameter1: float = 1.0, parameter2: float = 0.5):
        self.parameter1 = parameter1
        self.parameter2 = parameter2
    
    def compute_something(self, input_data: np.ndarray) -> np.ndarray:
        return input_data * self.parameter1 + self.parameter2

class ComplexImplementation(MyNewFunction):
    """Complex implementation with different logic."""
    
    def compute_something(self, input_data: np.ndarray) -> np.ndarray:
        # More complex implementation
        pass
```

3. **Update the configuration** (see Configuration System section)

## Configuration System

### Pydantic-Based Configuration

The configuration system uses **Pydantic models** to define and validate configurations:

```python
# Example: FirmsConfiguration
class Production(BaseModel):
    """Configuration for production function."""
    name: Literal["PureLeontief", "CriticalAndImportantLeontief", "Linear"] = "PureLeontief"
    path_name: str = "production"
    parameters: dict[str, Any] = {}

class Prices(BaseModel):
    """Configuration for pricing function."""
    name: Literal["MarkUp", "Marginal", "Adaptive"] = "MarkUp"
    path_name: str = "prices"
    parameters: dict[str, Any] = {}

class FirmsFunctions(BaseModel):
    """Container for all firm function configurations."""
    production: Production = Production()
    prices: Prices = Prices()
    wage_setter: WageSetter = WageSetter()
    target_production: TargetProduction = TargetProduction()
    credit_demand: CreditDemand = CreditDemand()
    demography: Demography = Demography()

class FirmsConfiguration(BaseModel):
    """Complete configuration for firm agents."""
    parameters: FirmsParameters = FirmsParameters()
    functions: FirmsFunctions = FirmsFunctions()
    calculate_hill_exponent: bool = True
```

### Dynamic Function Loading

The `function_mapping.py` utility dynamically loads and instantiates function classes:

```python
def functions_from_model(model: BaseModel, loc: str) -> dict[str, Any]:
    """Create function instances from a Pydantic model specification."""
    loaded_classes = {}
    for field_name, field_value in model:
        path_name = field_value.path_name
        name = field_value.name
        parameters = field_value.parameters
        
        # Dynamic import based on configuration
        module = __import__(f"{loc}.func.{path_name}", fromlist=[name])
        cls = getattr(module, name)
        loaded_classes[path_name] = cls(**parameters)
    return loaded_classes
```

### Configuration Usage

```python
# Create configuration
config = FirmsConfiguration(
    functions=FirmsFunctions(
        production=Production(name="PureLeontief"),
        prices=Prices(name="MarkUp", parameters={"markup_rate": 0.15}),
        wage_setter=WageSetter(name="Adaptive")
    )
)

# Configuration automatically loads appropriate functions
functions = functions_from_model(config.functions, "macromodel.agents.firms")

# Agent uses configured functions
production_level = functions["production"].compute_production(...)
price_level = functions["prices"].compute_prices(...)
```

### Adding New Configuration Options

To add a new function to the configuration:

1. **Update the function configuration class**:

```python
class MyNewFunction(BaseModel):
    """Configuration for new function."""
    name: Literal["SimpleImplementation", "ComplexImplementation"] = "SimpleImplementation"
    path_name: str = "my_new_function"
    parameters: dict[str, Any] = {}
```

2. **Add to the agent's function container**:

```python
class FirmsFunctions(BaseModel):
    # ... existing functions
    my_new_function: MyNewFunction = MyNewFunction()
```

3. **Update the agent to use the new function**:

```python
# In agent's behavioral method
result = self.functions["my_new_function"].compute_something(input_data)
```

## Adding New Agent Types

To add a new agent type:

### 1. Create Agent Class

```python
# File: macromodel/agents/my_new_agent/my_new_agent.py
from macromodel.agents.agent.agent import Agent
from macromodel.agents.my_new_agent.my_new_agent_ts import MyNewAgentTimeSeries

class MyNewAgent(Agent):
    """New agent type for specific economic behavior."""
    
    def __init__(self, ts: MyNewAgentTimeSeries, functions: dict, configuration, ...):
        super().__init__(ts, functions, configuration, ...)
        # Agent-specific initialization
    
    @classmethod
    def from_pickled_agent(cls, synthetic_data, configuration, ...):
        """Initialize from synthetic data."""
        ts = MyNewAgentTimeSeries.from_data(synthetic_data, ...)
        functions = functions_from_model(configuration.functions, "macromodel.agents.my_new_agent")
        return cls(ts=ts, functions=functions, configuration=configuration, ...)
    
    def agent_step(self):
        """Main behavioral step in each time period."""
        # Use functions to make decisions
        decision = self.functions["my_function"].compute_decision(...)
        
        # Update timeseries
        self.ts.decision.append(decision)
        
        # Interact with markets
        self.participate_in_market(decision)
```

### 2. Create TimeSeries Class

```python
# File: macromodel/agents/my_new_agent/my_new_agent_ts.py
from macromodel.timeseries import TimeSeries

class MyNewAgentTimeSeries(TimeSeries):
    """Time series for new agent type."""
    
    @classmethod
    def from_data(cls, data: pd.DataFrame, ...):
        """Create TimeSeries from initial data."""
        return cls(
            decision=data["Decision"].values,
            state_variable=data["StateVariable"].values,
            # ... other variables
        )
```

### 3. Create Function Directory

```
macromodel/agents/my_new_agent/func/
├── my_function.py          # Abstract base class and implementations
├── another_function.py     # Another behavioral function
└── __init__.py
```

### 4. Create Configuration

```python
# File: macromodel/configurations/my_new_agent_configuration.py
class MyNewAgentConfiguration(BaseModel):
    """Configuration for new agent type."""
    functions: MyNewAgentFunctions = MyNewAgentFunctions()
    parameters: MyNewAgentParameters = MyNewAgentParameters()
```

## Testing Your Implementation

### Agent Behavior Testing

```python
# Test individual agent behaviors
def test_agent_production():
    """Test agent production function."""
    agent = create_test_agent()
    
    # Test function directly
    result = agent.functions["production"].compute_production(...)
    assert result > 0
    
    # Test agent step
    initial_production = agent.ts.current("production")
    agent.agent_step()
    new_production = agent.ts.current("production")
    assert new_production != initial_production
```

### Function Testing

```python
# Test function implementations
def test_production_function():
    """Test production function implementation."""
    func = PureLeontief()
    
    inputs = np.array([100, 50, 75])
    result = func.compute_production(inputs, ...)
    
    assert result.shape == inputs.shape
    assert np.all(result >= 0)
```

### Configuration Testing

```python
# Test configuration system
def test_configuration_loading():
    """Test that configurations load correct functions."""
    config = FirmsConfiguration(
        functions=FirmsFunctions(
            production=Production(name="PureLeontief")
        )
    )
    
    functions = functions_from_model(config.functions, "macromodel.agents.firms")
    
    assert "production" in functions
    assert isinstance(functions["production"], PureLeontief)
```

## Common Patterns

### Agent Decision-Making Pattern

```python
def agent_step(self):
    """Standard agent step pattern."""
    # 1. Gather information
    current_state = self.ts.current("state_variable")
    market_info = self.get_market_information()
    
    # 2. Make decisions using functions
    decision = self.functions["decision_function"].compute_decision(
        current_state=current_state,
        market_info=market_info,
        parameters=self.configuration.parameters
    )
    
    # 3. Update internal state
    self.ts.decision.append(decision)
    
    # 4. Interact with markets
    self.participate_in_market(decision)
    
    # 5. Record outcomes
    outcome = self.get_market_outcome()
    self.ts.outcome.append(outcome)
```

### Market Interaction Pattern

```python
def participate_in_market(self, decision):
    """Standard market interaction pattern."""
    # Create market orders
    orders = self.create_market_orders(decision)
    
    # Submit to appropriate markets
    for market_name, order in orders.items():
        market = self.get_market(market_name)
        market.add_order(order)
```

### Function Implementation Pattern

```python
class MyFunction(MyFunctionBase):
    """Standard function implementation pattern."""
    
    def __init__(self, param1: float = 1.0, param2: float = 0.5):
        """Initialize with parameters from configuration."""
        self.param1 = param1
        self.param2 = param2
    
    def compute_something(self, input_data: np.ndarray) -> np.ndarray:
        """Implement the abstract method."""
        # Validate inputs
        assert isinstance(input_data, np.ndarray)
        assert input_data.ndim == 1
        
        # Apply function logic
        result = input_data * self.param1 + self.param2
        
        # Validate outputs
        assert result.shape == input_data.shape
        
        return result
```

## Simulation Loop Architecture

### Main Simulation Class

The `Simulation` class orchestrates the entire macroeconomic simulation with these key components:

```python
class Simulation:
    """Main simulation orchestrator."""
    
    def __init__(self, ...):
        self.countries = {}                    # National economies
        self.rest_of_the_world = RestOfTheWorld()  # External sector
        self.goods_market = GoodsMarket()      # Global trade
        self.exchange_rates = ExchangeRates()  # Currency dynamics
        self.timestep = Timestep()             # Time management
        self.regional_aggregator = None        # Currency unions
```

### Simulation Loop Structure

Each timestep follows a **five-phase architecture** with precise ordering:

#### Phase 1: Country-Level Processing (Sequential)

```python
for country in self.countries.values():
    # Update exchange rates
    exchange_rate = self.exchange_rates.get_current_exchange_rates_from_usd_to_lcu(...)
    
    # Country processing phases
    country.initialisation_phase(exchange_rate)  # Demographics, exchange rates
    country.estimation_phase()                   # Expectations, forecasting
    country.target_setting_phase()               # Production targets, wages
    country.clear_labour_market()                # Employment matching
    country.update_planning_metrics()            # Forward-looking indicators
```

#### Phase 2: Regional Coordination (If Applicable)

```python
if self.regional_aggregator:
    # Synchronize central banks across currency unions
    self.regional_aggregator.sync_central_banks(self.countries)
```

#### Phase 3: Local Market Clearing (Sequential)

```python
for country in self.countries.values():
    # Domestic markets (housing, credit)
    country.prepare_housing_market_clearing()
    country.clear_housing_market()
    country.prepare_credit_market_clearing()
    country.clear_credit_market()
    country.process_housing_market_clearing()
    country.process_credit_market_clearing()
    
    # Prepare for global goods market
    country.prepare_goods_market_clearing()
```

#### Phase 4: Global Market Clearing (Simultaneous)

```python
# Global goods market - all countries participate simultaneously
self.goods_market.prepare()    # Collect supply/demand from all countries
self.goods_market.clear()      # Execute clearing mechanism
self.goods_market.record()     # Update agent states with results
```

#### Phase 5: Post-Market Updates

```python
# Record outcomes and update metrics
self.rest_of_the_world.record_bought_goods()
for country in self.countries.values():
    country.update_realised_metrics()      # Comprehensive updates
    country.update_population_structure()  # Demographics

# Advance time
self.timestep.step()
```

### Market Clearing Mechanisms

The simulation supports **multiple clearing algorithms** that can be configured:

#### Available Clearing Algorithms

```python
# Different clearing strategies
lib_default.py      # Random matching with priorities
lib_pro_rata.py     # Pro-rata allocation mechanism  
lib_water_bucket.py # Water bucket allocation strategy
```

#### Market Types and Timing

- **Labor Market**: Cleared within each country during country processing
- **Housing Market**: Cleared within each country (domestic transactions)
- **Credit Market**: Cleared within each country (domestic lending)
- **Goods Market**: **Global clearing** - all countries participate simultaneously

#### Market Configuration

```python
# Example: Goods Market Configuration
class GoodsMarketConfiguration(BaseModel):
    clearing_mechanism: Literal["lib_default", "lib_pro_rata", "lib_water_bucket"] = "lib_default"
    use_trade_proportions: bool = True
    enable_supply_chain_persistence: bool = True
    clearing_parameters: dict[str, Any] = {}
```

### Multi-Country Coordination

#### Regional Aggregator

For currency unions (e.g., Eurozone), the `RegionalAggregator` provides:

```python
class RegionalAggregator:
    """Coordinates monetary policy across currency unions."""
    
    def sync_central_banks(self, countries: dict[str, Country]):
        """Synchronize central bank policies across member countries."""
        # Compute output-weighted average inflation and growth
        # Set unified monetary policy rates
        # Update all member central banks
```

#### Exchange Rate Dynamics

```python
class ExchangeRates:
    """Manages currency exchange rates over time."""
    
    def get_current_exchange_rates_from_usd_to_lcu(self, country: Country, ...):
        """Get current exchange rate for a country."""
        # Based on previous inflation and growth rates
        # Affects all international transactions
```

### Time Management

#### Timestep Class

```python
class Timestep:
    """Manages simulation time progression."""
    
    def __init__(self, year: int = 2020, month: int = 1, increment: int = 1):
        self.year = year
        self.month = month
        self.increment = increment  # Usually 1 month
    
    def step(self):
        """Advance time by one increment."""
        self.month += self.increment
        if self.month > 12:
            self.year += 1
            self.month = 1
```

#### Time Synchronization

- All agents and markets operate on the same timestep
- Sequential processing within timesteps
- Time advances only after all processing is complete

### Simulation Configuration

#### Hierarchical Configuration

```python
class SimulationConfiguration(BaseModel):
    """Main simulation configuration."""
    
    t_max: int = 120                                    # Simulation length (months)
    seed: Optional[int] = None                          # Random seed
    country_configurations: dict[str, CountryConfiguration] = {}
    row_configuration: ROWConfiguration = ROWConfiguration()
    goods_market_configuration: GoodsMarketConfiguration = GoodsMarketConfiguration()
    exchange_rates_configuration: ExchangeRatesConfiguration = ExchangeRatesConfiguration()
```

#### Country-Level Configuration

Each country has comprehensive configuration covering:
- All agent types (firms, households, banks, government)
- Market configurations (labor, credit, housing)
- Behavioral parameters and policy settings
- Time series tracking preferences

### Adding New Market Types

To add a new market type:

#### 1. Create Market Class

```python
# File: macromodel/markets/my_new_market/my_new_market.py
class MyNewMarket:
    """New market type for specific transactions."""
    
    def __init__(self, configuration: MyNewMarketConfiguration):
        self.configuration = configuration
        self.buyers = []
        self.sellers = []
        self.clearing_mechanism = None
    
    def prepare(self):
        """Collect supply and demand from participants."""
        pass
    
    def clear(self):
        """Execute clearing mechanism."""
        pass
    
    def record(self):
        """Update agent states with transaction results."""
        pass
```

#### 2. Integrate with Country Class

```python
# In country.py
def clear_my_new_market(self):
    """Clear the new market type."""
    self.my_new_market.prepare()
    self.my_new_market.clear()
    self.my_new_market.record()
```

#### 3. Add to Simulation Loop

```python
# In simulation.py iterate() method
for country in self.countries.values():
    # Add to appropriate phase
    country.clear_my_new_market()
```

### Adding New Clearing Mechanisms

To add a new clearing mechanism:

#### 1. Create Clearing Algorithm

```python
# File: macromodel/markets/clearing/lib_my_algorithm.py
def clear_market(buyers, sellers, configuration):
    """Custom clearing algorithm."""
    # Implement matching logic
    # Return transactions
    pass
```

#### 2. Register in Configuration

```python
class MarketConfiguration(BaseModel):
    clearing_mechanism: Literal["lib_default", "lib_pro_rata", "lib_water_bucket", "lib_my_algorithm"] = "lib_default"
```

#### 3. Integrate in Market Class

```python
def clear(self):
    """Execute configured clearing mechanism."""
    if self.configuration.clearing_mechanism == "lib_my_algorithm":
        from macromodel.markets.clearing.lib_my_algorithm import clear_market
        transactions = clear_market(self.buyers, self.sellers, self.configuration)
```

### Critical Architecture Patterns

#### Sequential vs. Simultaneous Processing

- **Country-level markets** clear sequentially (each country processes independently)
- **Global goods market** clears simultaneously (all countries participate together)
- This distinction is crucial for maintaining economic consistency

#### Phase Dependencies

- Exchange rates must be updated before goods market clearing
- Labor markets must clear before production decisions
- Local markets must clear before global market participation

#### Multi-Level Configuration

- **Simulation-level**: Overall parameters, time limits, global settings
- **Country-level**: National policies, demographics, institutional settings
- **Agent-level**: Individual behavioral parameters and functions

## Configuration System Architecture

### Pydantic-Based Configuration

The macromodel package uses a comprehensive configuration system built on **Pydantic BaseModel** classes. This provides type safety, validation, and flexible parameter management for complex macroeconomic simulations.

#### Benefits of Pydantic BaseModel

1. **Type Safety**: Runtime validation and type checking
2. **IDE Support**: Auto-completion and type hints
3. **Validation**: Automatic parameter validation with constraints
4. **Documentation**: Self-documenting through type annotations
5. **Serialization**: Built-in JSON/YAML support
6. **Default Values**: Sensible defaults for all parameters

#### Basic Configuration Pattern

```python
from pydantic import BaseModel, Field
from typing import Literal, Any

class MyAgentConfiguration(BaseModel):
    """Configuration for agent behavior and parameters."""
    
    # Parameters with validation
    adjustment_speed: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Speed of adjustment (0-1)"
    )
    
    # Function selection with restricted options
    pricing_function: Literal["MarkUp", "Marginal", "Adaptive"] = "MarkUp"
    
    # Complex nested parameters
    policy_parameters: dict[str, Any] = {
        "intervention_threshold": 0.05,
        "policy_strength": 1.0
    }
```

### Configuration Hierarchy

The configuration system follows a hierarchical structure:

#### Top-Level: SimulationConfiguration

```python
class SimulationConfiguration(BaseModel):
    """Main configuration for the entire simulation."""
    
    # Simulation parameters
    t_max: int = 120                          # Simulation length (months)
    seed: Optional[int] = None                # Random seed
    
    # Hierarchical configurations
    country_configurations: dict[str, CountryConfiguration] = {}
    row_configuration: RestOfTheWorldConfiguration = RestOfTheWorldConfiguration()
    goods_market_configuration: GoodsMarketConfiguration = GoodsMarketConfiguration()
    exchange_rates_configuration: ExchangeRatesConfiguration = ExchangeRatesConfiguration()
```

#### Country-Level: CountryConfiguration

```python
class CountryConfiguration(BaseModel):
    """Configuration for a single country's economy."""
    
    # Economic system
    economy: EconomyConfiguration = EconomyConfiguration()
    
    # Agent configurations
    individuals: IndividualsConfiguration = IndividualsConfiguration()
    households: HouseholdsConfiguration = HouseholdsConfiguration()
    firms: FirmsConfiguration = FirmsConfiguration()
    banks: BanksConfiguration = BanksConfiguration()
    central_bank: CentralBankConfiguration = CentralBankConfiguration()
    central_government: CentralGovernmentConfiguration = CentralGovernmentConfiguration()
    government_entities: GovernmentEntitiesConfiguration = GovernmentEntitiesConfiguration()
    
    # Market configurations
    labour_market: LabourMarketConfiguration = LabourMarketConfiguration()
    housing_market: HousingMarketConfiguration = HousingMarketConfiguration()
    credit_market: CreditMarketConfiguration = CreditMarketConfiguration()
    
    # Exchange rates
    exchange_rates: ExchangeRatesConfiguration = ExchangeRatesConfiguration()
```

#### Agent-Level: Detailed Configuration

Each agent type has comprehensive configuration covering parameters and functions:

```python
class FirmsConfiguration(BaseModel):
    """Configuration for firm agents."""
    
    # Behavioral parameters
    parameters: FirmsParameters = FirmsParameters()
    
    # Behavioral functions
    functions: FirmsFunctions = FirmsFunctions()
    
    # Special settings
    calculate_hill_exponent: bool = True
    add_emissions: bool = False
```

### Function Configuration Pattern

The system uses a standardized pattern for configuring behavioral functions:

#### Function Configuration Structure

```python
class FunctionName(BaseModel):
    """Configuration for a specific behavioral function."""
    
    name: Literal["Option1", "Option2", "Option3"] = "DefaultOption"
    path_name: str = "module_path"
    parameters: dict[str, Any] = {}
```

#### Example: Firm Price Setting

```python
class Prices(BaseModel):
    """Configuration for firm price setting behavior."""
    
    name: Literal["DefaultPriceSetter", "ExogenousPriceSetter"] = "DefaultPriceSetter"
    path_name: str = "prices"
    parameters: dict[str, Any] = {
        "price_setting_noise_std": 0.05,
        "price_setting_speed_gf": 1.0,
        "price_setting_speed_dp": 0.0,
        "price_setting_speed_cp": 0.0,
    }
```

#### Agent Function Collections

Each agent type has a collection of configurable functions:

```python
class FirmsFunctions(BaseModel):
    """Collection of all firm behavioral functions."""
    
    production: Production = Production()
    prices: Prices = Prices()
    wage_setter: WageSetter = WageSetter()
    target_production: TargetProduction = TargetProduction()
    credit_demand: CreditDemand = CreditDemand()
    demography: Demography = Demography()
```

### Parameter Configuration

#### Parameter Types and Validation

The system supports various parameter types with comprehensive validation:

```python
class FirmsParameters(BaseModel):
    """Economic and behavioral parameters for firms."""
    
    # Utilization rates (0-1)
    capital_inputs_utilisation_rate: float = Field(1.0, ge=0.0, le=1.0)
    intermediate_inputs_utilisation_rate: float = Field(1.0, ge=0.0, le=1.0)
    
    # Industry-specific arrays
    capital_inputs_delay: list[int] = [0 for _ in range(18)]
    depreciation_rates: list[float] = [0.0 for _ in range(18)]
    
    # Behavioral parameters
    expected_capacity_utilisation: float = Field(0.8, ge=0.0, le=1.0)
    investment_price_elasticity: float = Field(-0.5, le=0.0)
    
    # Policy parameters
    carbon_tax_rate: float = Field(0.0, ge=0.0)
    regulatory_compliance_cost: float = Field(0.0, ge=0.0)
```

#### Parameter Categories

- **Economic Parameters**: Financial ratios, depreciation rates, utilization rates
- **Behavioral Parameters**: Adjustment speeds, elasticities, noise levels
- **Policy Parameters**: Tax rates, regulatory costs, intervention thresholds
- **Technical Parameters**: Delays, time constants, numerical tolerances

### Dynamic Function Loading

The configuration system enables dynamic function loading:

#### Function Mapping Utility

```python
def functions_from_model(model: BaseModel, loc: str) -> dict[str, Any]:
    """Create function instances from configuration."""
    loaded_functions = {}
    
    for field_name, field_value in model:
        # Extract configuration
        function_name = field_value.name
        path_name = field_value.path_name
        parameters = field_value.parameters
        
        # Dynamic import
        module_path = f"{loc}.func.{path_name}"
        module = __import__(module_path, fromlist=[function_name])
        function_class = getattr(module, function_name)
        
        # Instantiate with parameters
        loaded_functions[path_name] = function_class(**parameters)
    
    return loaded_functions
```

#### Usage in Agent Initialization

```python
# In agent initialization
functions = functions_from_model(
    configuration.functions,
    "macromodel.agents.firms"
)

# Agent uses configured functions
production_level = functions["production"].compute_production(...)
price_level = functions["prices"].compute_prices(...)
```

### Configuration Usage Patterns

#### Creating Configurations

```python
# Using defaults
config = FirmsConfiguration()

# Customizing parameters
config = FirmsConfiguration(
    parameters=FirmsParameters(
        capital_inputs_utilisation_rate=0.9,
        expected_capacity_utilisation=0.85
    ),
    functions=FirmsFunctions(
        production=Production(name="PureLeontief"),
        prices=Prices(
            name="DefaultPriceSetter",
            parameters={"price_setting_noise_std": 0.02}
        )
    )
)
```

#### Simulation-Level Configuration

```python
# Complete simulation configuration
simulation_config = SimulationConfiguration(
    t_max=240,  # 20 years
    seed=42,
    country_configurations={
        "FRA": CountryConfiguration(
            firms=FirmsConfiguration(
                parameters=FirmsParameters(capital_inputs_utilisation_rate=0.95),
                functions=FirmsFunctions(
                    production=Production(name="CriticalAndImportantLeontief")
                )
            )
        ),
        "DEU": CountryConfiguration(
            firms=FirmsConfiguration(
                parameters=FirmsParameters(capital_inputs_utilisation_rate=0.92)
            )
        )
    }
)
```

### YAML Integration

#### Configuration Serialization

```python
# Save configuration to YAML
config_dict = simulation_config.model_dump()
with open("simulation_config.yaml", "w") as f:
    yaml.dump(config_dict, f)

# Load configuration from YAML
with open("simulation_config.yaml", "r") as f:
    config_dict = yaml.safe_load(f)
    
simulation_config = SimulationConfiguration(**config_dict)
```

#### Example YAML Structure

```yaml
t_max: 240
seed: 42
country_configurations:
  FRA:
    firms:
      parameters:
        capital_inputs_utilisation_rate: 0.95
        expected_capacity_utilisation: 0.85
      functions:
        production:
          name: CriticalAndImportantLeontief
          path_name: production
          parameters: {}
        prices:
          name: DefaultPriceSetter
          path_name: prices
          parameters:
            price_setting_noise_std: 0.02
            price_setting_speed_gf: 1.0
    banks:
      parameters:
        capital_adequacy_ratio: 0.08
        firm_loans_debt_to_equity_ratio: 0.03
```

### Adding New Configuration Parameters

#### 1. Add to Parameter Class

```python
class FirmsParameters(BaseModel):
    # ... existing parameters
    
    # New parameter with validation
    new_behavioral_parameter: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="New behavioral parameter description"
    )
```

#### 2. Add to Function Configuration

```python
class MyNewFunction(BaseModel):
    """Configuration for new behavioral function."""
    
    name: Literal["SimpleImpl", "ComplexImpl", "AdaptiveImpl"] = "SimpleImpl"
    path_name: str = "my_new_function"
    parameters: dict[str, Any] = {
        "sensitivity": 0.5,
        "threshold": 0.1
    }
```

#### 3. Add to Agent Function Collection

```python
class FirmsFunctions(BaseModel):
    # ... existing functions
    my_new_function: MyNewFunction = MyNewFunction()
```

### Configuration Best Practices

#### Parameter Validation

```python
class MyParameters(BaseModel):
    """Well-validated parameter configuration."""
    
    # Use Field for validation
    ratio_parameter: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Must be between 0 and 1"
    )
    
    # Use Literal for restricted choices
    strategy: Literal["conservative", "aggressive", "adaptive"] = "conservative"
    
    # Use positive constraints
    positive_value: float = Field(gt=0.0, description="Must be positive")
```

#### Documentation

```python
class WellDocumentedConfig(BaseModel):
    """
    Configuration for economic agent behavior.
    
    This configuration controls how agents make decisions and interact
    with markets. All parameters have economic interpretations.
    """
    
    elasticity: float = Field(
        default=-0.5,
        le=0.0,
        description="Price elasticity of demand (negative value expected)"
    )
    
    adjustment_speed: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Speed of adjustment to equilibrium (0=no adjustment, 1=instant)"
    )
```

#### Configuration Testing

```python
def test_configuration_validation():
    """Test configuration validation."""
    
    # Valid configuration
    config = FirmsConfiguration(
        parameters=FirmsParameters(capital_inputs_utilisation_rate=0.9)
    )
    assert config.parameters.capital_inputs_utilisation_rate == 0.9
    
    # Invalid configuration should raise error
    with pytest.raises(ValidationError):
        FirmsConfiguration(
            parameters=FirmsParameters(capital_inputs_utilisation_rate=1.5)  # > 1.0
        )
```

### Configuration Architecture Benefits

1. **Type Safety**: Runtime validation prevents configuration errors
2. **Flexibility**: Easy to swap implementations and adjust parameters
3. **Reproducibility**: Configurations can be saved and reloaded exactly
4. **Documentation**: Self-documenting through type annotations
5. **Extensibility**: New parameters and functions can be added seamlessly
6. **External Management**: YAML support enables configuration without code changes
7. **Validation**: Automatic parameter validation with meaningful error messages
8. **IDE Support**: Auto-completion and type checking during development

This configuration system enables researchers to easily customize complex macroeconomic models, conduct sensitivity analysis, and ensure reproducible research while maintaining code quality and type safety.

## Architecture Summary

The macromodel architecture provides:

1. **Flexible Agent System**: Agents can be easily extended with new behaviors
2. **Configurable Functions**: Behavioral functions can be swapped via configuration
3. **Temporal Data Management**: TimeSeries objects provide efficient time-based data access
4. **Type-Safe Configuration**: Pydantic models ensure configuration validity
5. **Dynamic Function Loading**: Functions are loaded at runtime based on configuration
6. **Extensible Design**: New agents, functions, and behaviors can be added without modifying existing code
7. **Multi-Phase Simulation Loop**: Precise ordering of operations across countries and markets
8. **Multiple Market Clearing**: Different algorithms for different market types
9. **Multi-Country Coordination**: Support for currency unions and international trade

This architecture enables researchers to implement diverse economic theories and behavioral models within a consistent, maintainable framework while supporting complex multi-country macroeconomic simulations.