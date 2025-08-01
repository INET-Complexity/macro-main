# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup
This project uses the `macromodel` conda environment.

## Development Commands

### Code Formatting and Linting
```bash
# Format code with black and sort imports
./run_style.sh

# Or run individually:
black --config="pyproject.toml" .
isort . --settings-path pyproject.toml
```

### Testing
```bash
# Run all tests
pytest

# Run specific test modules
pytest tests/test_macro_data/
pytest tests/test_macromodel/

# Run with coverage
pytest --cov=macro_data --cov=macromodel
```

### Installation
```bash
# Development installation with all tools
pip install -e ".[dev]" --config-settings editable_mode=strict

# Basic installation
pip install .

# Documentation tools
pip install -e ".[docs]"
```

## Architecture Overview

This is a comprehensive macroeconomic simulation framework with two main packages:

### macro_data Package
- **Purpose**: Data preprocessing, harmonization, and initialization
- **Key Component**: `DataWrapper` class - primary interface for data preprocessing
- **Functionality**: 
  - Country-level economic data processing
  - Rest of World aggregation
  - Exchange rates and trade relationships
  - Market structure initialization
  - Input-Output table processing via `icio_reader` and `wiod_reader`

### macromodel Package
- **Purpose**: Core simulation engine and economic behaviors
- **Key Component**: `Simulation` class - orchestrates multi-country economic interactions
- **Agent-Based Structure**:
  - `agents/`: Economic actors (banks, firms, households, individuals, central_bank, central_government, government_entities)
  - `markets/`: Market clearing mechanisms (goods_market, credit_market, housing_market, labour_market)
  - `country/`: Country-level aggregation and modeling
  - `rest_of_the_world/`: External economy interactions

### Key Architectural Patterns

**Agent-Based Modeling**: Each economic actor type has its own module with behavioral functions in `func/` subdirectories.

**Market Clearing**: Multiple clearing mechanisms available:
- `lib_default.py`, `lib_pro_rata.py`, `lib_water_bucket.py` for different allocation strategies

**Configuration System**: Extensive configuration classes in `configurations/` for customizing:
- Country-specific parameters (`CountryConfiguration`)
- Market behaviors (`CreditMarketConfiguration`, `GoodsMarketConfiguration`, etc.)
- Simulation settings (`SimulationConfiguration`)

**Time Series Tracking**: Dedicated `*_ts.py` files alongside main classes for temporal data collection.

**Data Flow**:
1. Raw data → `macro_data` preprocessing → `DataWrapper.save()` → `.pkl` files
2. Load data via `DataWrapper.init_from_pickle()`
3. Initialize `Simulation.from_datawrapper()` with configurations
4. Run simulation with `model.run()`
5. Save results to HDF5 format with `model.save()`

## Key Integration Points

- **IO Tables**: ICIO and WIOD readers in `macro_data/readers/io_tables/`
- **Exchange Rates**: Handled by `ExchangeRates` class with country-specific dynamics
- **Regional Aggregation**: `RegionalAggregator` for multi-region modeling
- **Policy Implementation**: Carbon pricing and regulations in `macromodel/policy/`

## Testing Structure

Tests mirror the package structure with comprehensive unit tests for:
- Data processing components
- Individual agent behaviors
- Market clearing mechanisms
- Configuration validation
- End-to-end simulation functionality

Use `conftest.py` files for test fixtures and shared test data setup.

## Development Workflow

### Git Commit Guidelines
**IMPORTANT**: Create commits for every modification to maintain detailed tracking on GitHub. Each logical change should be committed separately with descriptive messages explaining:
- What was changed
- Why the change was made
- Any relevant context or implications

Examples:
```bash
git commit -m "Add SubstitutionBundlesConfiguration to households config

- Enable CES utility function with configurable elasticity
- Support bundle definitions for substitutable goods
- Maintain backward compatibility with existing consumption"

git commit -m "Implement CESHouseholdConsumption class

- Calculate dynamic consumption shares based on prices
- Handle tax rate changes in substitution logic
- Store initial conditions for bundle normalization"
```

This granular commit approach helps with:
- Code review and debugging
- Feature development tracking
- Rollback capabilities
- Collaboration visibility