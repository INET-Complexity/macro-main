# Project Structure and Contribution Guide

This guide helps you understand where to contribute to the macroeconomic simulation framework based on what you want to achieve.

## Architecture Overview

The project consists of two main packages with distinct responsibilities:

### macro_data Package

- **Purpose**: Data preprocessing, harmonization, and initialization
- **Key responsibility**: Transform raw data sources into standardized formats for simulation
- **When to contribute here**:
  - Adding new data sources (e.g., emissions data, employment data)
  - Improving data processing algorithms
  - Adding new reader classes for external data
  - Modifying data harmonization logic

### macromodel Package  

- **Purpose**: Core simulation engine and economic agent behaviors
- **Key responsibility**: Run the actual macroeconomic simulations
- **When to contribute here**:
  - Adding new economic agents or agent behaviors
  - Implementing new market clearing mechanisms
  - Modifying simulation logic
  - Adding new economic models or theories

## Decision Tree: Where Should I Contribute?

### 🗂️ **Contributing to Data** → [macro_data Guide](macro_data.md)

Choose `macro_data` if you want to:

- ✅ Add a new data source (World Bank, IMF, Eurostat, etc.)
- ✅ Improve how existing data is processed or cleaned
- ✅ Add new reader classes for external datasets
- ✅ Modify data harmonization between sources
- ✅ Change how synthetic agents are initialized from data

**Examples**: Adding methane emissions data, improving GDP data processing, creating a new population reader

### 🏛️ **Contributing to Simulation** → [macromodel Guide](macromodel.md)

Choose `macromodel` if you want to:

- ✅ Change how agents behave (firm decisions, household consumption)
- ✅ Add new types of economic agents
- ✅ Implement new market clearing algorithms
- ✅ Modify simulation dynamics or time progression
- ✅ Add new economic theories or models

**Examples**: Implementing new firm pricing strategies, adding central bank policy rules, creating new market mechanisms

## Package Structure

```
project/
├── macro_data/                    # Data preprocessing and initialization
│   ├── readers/                   # Interface with external data sources
│   ├── processing/                # Transform data into synthetic agents
│   ├── configuration/             # Data configuration classes
│   └── data_wrapper.py           # Main data orchestration
│
├── macromodel/                    # Core simulation engine
│   ├── agents/                    # Economic agent behaviors
│   ├── markets/                   # Market clearing mechanisms
│   ├── countries/                 # Country-level coordination
│   ├── configurations/            # Simulation configuration classes
│   └── simulation.py             # Main simulation orchestration
│
└── docs/                         # Documentation
    └── contributing/             # Contribution guides
        ├── macro_data.md         # Detailed guide for data contributions
        ├── macromodel.md         # Detailed guide for model contributions
        ├── development.md        # Development workflow
        ├── style_guide.md        # Code style guidelines
        └── testing.md            # Testing guidelines
```

## Getting Started

1. **Read the appropriate detailed guide**:
   - [macro_data Guide](macro_data.md) for data-related contributions
   - [macromodel Guide](macromodel.md) for simulation-related contributions

2. **Set up your development environment**: See [Development Guide](development.md)

3. **Follow code style guidelines**: See [Style Guide](style_guide.md)

4. **Write tests**: See [Testing Guidelines](testing.md)

## Quick Reference

| What do you want to do? | Package | Detailed Guide |
|-------------------------|---------|----------------|
| Add new data source | `macro_data` | [macro_data](macro_data.md) |
| Improve data processing | `macro_data` | [macro_data](macro_data.md) |
| Change agent behavior | `macromodel` | [macromodel](macromodel.md) |
| Add new agent type | `macromodel` | [macromodel](macromodel.md) |
| Implement new market | `macromodel` | [macromodel](macromodel.md) |
| Modify simulation logic | `macromodel` | [macromodel](macromodel.md) |

---

This structure ensures clear separation of concerns: **data preparation** vs. **simulation execution**. Understanding this distinction will help you contribute to the right part of the codebase.
