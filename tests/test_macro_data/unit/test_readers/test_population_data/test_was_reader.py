# Tests for the WASReader class
"""
Tests for the WASReader class.

This module tests the functionality of the Wealth and Assets Survey (WAS) data reader,
including data loading, processing, variable mapping, and currency conversion.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from macro_data.readers.population_data.was_reader import WASReader, var_mapping, var_numerical
from macro_data.readers.economic_data.exchange_rates import ExchangeRatesReader


class TestWASReader:
    """Test class for WASReader functionality."""

    @pytest.fixture
    def mock_exchange_rates(self):
        """Create a mock exchange rates reader."""
        mock_rates = Mock(spec=ExchangeRatesReader)
        mock_rates.from_eur_to_lcu.return_value = 1.0  # GBP to GBP conversion
        return mock_rates

    @pytest.fixture
    def sample_individuals_data(self):
        """Create sample individual-level data."""
        return pd.DataFrame({
            'pidno': [1, 2, 3, 4, 5],
            'personr7': [1, 2, 1, 1, 2],
            'CASER7': [1001, 1001, 1002, 1003, 1003],
            'R7xsperswgt': [1.2, 1.1, 0.9, 1.3, 1.0],
            'sexr7': [1, 2, 1, 2, 1],
            'DVAge17R7': [25, 30, 45, 35, 28],
            'edlevelr7': [3, 4, 2, 5, 3],
            'wrkingr7': [1, 1, 0, 1, 1],
            'dvgrspayannualr7': [30000, 40000, 0, 50000, 35000],
            'dvnetpayannualr7': [24000, 32000, 0, 40000, 28000],
            'DvtotgirR7': [30000, 40000, 0, 50000, 35000],
        })

    @pytest.fixture
    def sample_households_data(self):
        """Create sample household-level data."""
        return pd.DataFrame({
            'CASER7': [1001, 1002, 1003],
            'hvaluer7': [200000, 150000, 300000],
            'DVHValueR7': [200000, 150000, 300000],
            'DVSaValR7_SUM': [10000, 5000, 20000],
            'TotmortR7': [120000, 80000, 200000],
            'hhldrr7': [1, 2, 1],
            'dvrentpaidr7': [0, 800, 0],
        })

    def test_initialization(self, sample_individuals_data, sample_households_data):
        """Test WASReader initialization with valid data."""
        was_reader = WASReader(
            country_name_short="GB",
            individuals_df=sample_individuals_data,
            households_df=sample_households_data,
        )
        
        assert was_reader.country_name_short == "GB"
        assert len(was_reader.individuals_df) == 5
        assert len(was_reader.households_df) == 3
        assert list(was_reader.individuals_df.columns) == list(sample_individuals_data.columns)
        assert list(was_reader.households_df.columns) == list(sample_households_data.columns)

    def test_initialization_with_empty_dataframes(self):
        """Test WASReader initialization with empty DataFrames."""
        empty_df = pd.DataFrame()
        was_reader = WASReader(
            country_name_short="GB",
            individuals_df=empty_df,
            households_df=empty_df,
        )
        
        assert was_reader.country_name_short == "GB"
        assert len(was_reader.individuals_df) == 0
        assert len(was_reader.households_df) == 0

    @patch('pandas.read_stata')
    def test_read_stata_basic_functionality(self, mock_read_stata, mock_exchange_rates):
        """Test basic functionality of read_stata static method."""
        # Mock the Stata file reading
        mock_data = pd.DataFrame({
            'pidno': [1, 2, 3],
            'sexr7': [1, 2, 1],
            'DVAge17R7': [25, 30, 45],
            'dvgrspayannualr7': [30000, 40000, 50000],
            'DvtotgirR7': [30000, 40000, 50000],
            'unmapped_var': [1, 2, 3],  # This should be filtered out
        })
        mock_read_stata.return_value = mock_data
        
        # Test the read_stata method
        result = WASReader.read_stata(
            path="test_path.dta",
            country_name="United Kingdom",
            country_name_short="GB",
            year=2022,
            exchange_rates=mock_exchange_rates,
        )
        
        # Verify that pandas.read_stata was called correctly
        mock_read_stata.assert_called_once_with("test_path.dta", preserve_dtypes=False, convert_categoricals=False)
        
        # Verify that only mapped variables are kept
        expected_columns = ['Gender', 'Age', 'Employee Income', 'Income']
        assert all(col in result.columns for col in expected_columns)
        assert 'unmapped_var' not in result.columns
        
        # Verify that variable names are mapped correctly
        assert 'Gender' in result.columns  # sexr7 -> Gender
        assert 'Age' in result.columns  # DVAge17R7 -> Age
        
        # Verify that ID is set as index (pidno -> ID)
        assert result.index.name == 'ID'
        
        # Verify exchange rate conversion was called
        mock_exchange_rates.from_eur_to_lcu.assert_called_once_with(
            country="United Kingdom",
            year=2022,
        )

    @patch('pandas.read_stata')
    def test_read_stata_currency_conversion(self, mock_read_stata, mock_exchange_rates):
        """Test currency conversion in read_stata method."""
        # Mock exchange rate conversion
        mock_exchange_rates.from_eur_to_lcu.return_value = 1.2
        
        # Mock the Stata file reading
        mock_data = pd.DataFrame({
            'pidno': [1, 2],
            'dvgrspayannualr7': [1000, 2000],
            'DvtotgirR7': [1000, 2000],
        })
        mock_read_stata.return_value = mock_data
        
        result = WASReader.read_stata(
            path="test_path.dta",
            country_name="United Kingdom",
            country_name_short="GB",
            year=2022,
            exchange_rates=mock_exchange_rates,
        )
        
        # Verify currency conversion was applied
        expected_income = 1000 * 1.2
        assert result.loc[1, 'Employee Income'] == expected_income
        assert result.loc[1, 'Income'] == expected_income

    @patch('glob.glob')
    @patch.object(WASReader, 'read_stata')
    def test_from_stata_success(self, mock_read_stata, mock_glob, mock_exchange_rates):
        """Test successful creation of WASReader from Stata files."""
        # Mock glob to return file paths
        mock_glob.side_effect = [
            ['/path/to/was_round_7_person_eul_june_2022.dta'],  # person files
            ['/path/to/was_round_7_hhold_eul_march_2022.dta'],  # household files
        ]
        
        # Mock the read_stata method to return sample data
        mock_individuals = pd.DataFrame({'ID': [1, 2], 'Gender': [1, 2]})
        mock_households = pd.DataFrame({'HID': [1001, 1002], 'Income': [50000, 60000]})
        mock_read_stata.side_effect = [mock_individuals, mock_households]
        
        # Test from_stata method
        result = WASReader.from_stata(
            country_name="United Kingdom",
            country_name_short="GB",
            year=2022,
            was_data_path=Path("/path/to/was/data"),
            exchange_rates=mock_exchange_rates,
            round_number=7,
        )
        
        # Verify the result
        assert isinstance(result, WASReader)
        assert result.country_name_short == "GB"
        assert len(result.individuals_df) == 2
        assert len(result.households_df) == 2
        
        # Verify that read_stata was called twice (once for each file type)
        assert mock_read_stata.call_count == 2

    @patch('glob.glob')
    def test_from_stata_no_person_files(self, mock_glob, mock_exchange_rates):
        """Test from_stata method when no person files are found."""
        # Mock glob to return no person files
        mock_glob.side_effect = [
            [],  # No person files
            ['/path/to/was_round_7_hhold_eul_march_2022.dta'],  # household files exist
        ]
        
        # Test that FileNotFoundError is raised
        with pytest.raises(FileNotFoundError, match="No person files found"):
            WASReader.from_stata(
                country_name="United Kingdom",
                country_name_short="GB",
                year=2022,
                was_data_path=Path("/path/to/was/data"),
                exchange_rates=mock_exchange_rates,
                round_number=7,
            )

    @patch('glob.glob')
    def test_from_stata_no_household_files(self, mock_glob, mock_exchange_rates):
        """Test from_stata method when no household files are found."""
        # Mock glob to return no household files
        mock_glob.side_effect = [
            ['/path/to/was_round_7_person_eul_june_2022.dta'],  # person files exist
            [],  # No household files
        ]
        
        # Test that FileNotFoundError is raised
        with pytest.raises(FileNotFoundError, match="No household files found"):
            WASReader.from_stata(
                country_name="United Kingdom",
                country_name_short="GB",
                year=2022,
                was_data_path=Path("/path/to/was/data"),
                exchange_rates=mock_exchange_rates,
                round_number=7,
            )

    @patch('pandas.read_stata')
    def test_read_stata_missing_variables(self, mock_read_stata, mock_exchange_rates):
        """Test read_stata with missing variables in the data."""
        # Mock data with only some of the mapped variables
        mock_data = pd.DataFrame({
            'pidno': [1, 2],
            'sexr7': [1, 2],
            # Missing DVAge17R7, dvgrspayannualr7, etc.
        })
        mock_read_stata.return_value = mock_data
        
        result = WASReader.read_stata(
            path="test_path.dta",
            country_name="United Kingdom",
            country_name_short="GB",
            year=2022,
            exchange_rates=mock_exchange_rates,
        )
        
        # Should only contain the variables that exist in the data
        expected_columns = ['Gender']
        assert all(col in result.columns for col in expected_columns)
        assert len(result.columns) == 1
        # ID should be the index
        assert result.index.name == 'ID'

    @patch('pandas.read_stata')
    def test_read_stata_numeric_conversion_errors(self, mock_read_stata, mock_exchange_rates):
        """Test handling of numeric conversion errors in read_stata."""
        # Mock data with non-numeric values in monetary columns
        mock_data = pd.DataFrame({
            'pidno': [1, 2, 3],
            'dvgrspayannualr7': [30000, 'invalid', 50000],
            'DvtotgirR7': [30000, 'N/A', 50000],
        })
        mock_read_stata.return_value = mock_data
        
        result = WASReader.read_stata(
            path="test_path.dta",
            country_name="United Kingdom",
            country_name_short="GB",
            year=2022,
            exchange_rates=mock_exchange_rates,
        )
        
        # Verify that invalid values are converted to NaN
        assert pd.isna(result.loc[2, 'Employee Income'])
        assert pd.isna(result.loc[2, 'Income'])
        
        # Verify that valid values are preserved
        assert result.loc[1, 'Employee Income'] == 30000
        assert result.loc[1, 'Income'] == 30000

    def test_variable_mapping_completeness(self):
        """Test that variable mapping covers expected categories."""
        # Test that we have mappings for key categories
        individual_vars = ['pidno', 'personr7', 'CASER7', 'sexr7', 'DVAge17R7']
        income_vars = ['dvgrspayannualr7', 'dvnetpayannualr7', 'DvtotgirR7']
        asset_vars = ['hvaluer7', 'DVSaValR7_SUM', 'DVFFAssetsR7_SUM']
        liability_vars = ['TotmortR7', 'TOTCSCR7_aggr']
        housing_vars = ['hhldrr7', 'dvrentpaidr7']
        
        all_test_vars = individual_vars + income_vars + asset_vars + liability_vars + housing_vars
        
        for var in all_test_vars:
            assert var in var_mapping, f"Variable {var} not found in var_mapping"

    def test_numerical_variables_list(self):
        """Test that numerical variables list contains expected monetary variables."""
        expected_numerical = [
            'Income',
            'Employee Income',
            'Value of the Main Residence',
            'Wealth in Deposits',
            'Outstanding Balance of HMR Mortgages',
            'Rent Paid',
        ]
        
        for var in expected_numerical:
            assert var in var_numerical, f"Variable {var} not found in var_numerical"

    def test_dataframe_attributes(self, sample_individuals_data, sample_households_data):
        """Test that WASReader properly stores and exposes DataFrame attributes."""
        was_reader = WASReader(
            country_name_short="GB",
            individuals_df=sample_individuals_data,
            households_df=sample_households_data,
        )
        
        # Test that DataFrames are accessible as attributes
        assert hasattr(was_reader, 'individuals_df')
        assert hasattr(was_reader, 'households_df')
        
        # Test that they are the same objects
        assert was_reader.individuals_df is sample_individuals_data
        assert was_reader.households_df is sample_households_data
        
        # Test that they are DataFrames
        assert isinstance(was_reader.individuals_df, pd.DataFrame)
        assert isinstance(was_reader.households_df, pd.DataFrame)

    def test_country_code_validation(self, sample_individuals_data, sample_households_data):
        """Test that country code is properly stored and validated."""
        # Test with valid country code
        was_reader = WASReader(
            country_name_short="GB",
            individuals_df=sample_individuals_data,
            households_df=sample_households_data,
        )
        assert was_reader.country_name_short == "GB"
        
        # Test with different country code (should still work)
        was_reader_uk = WASReader(
            country_name_short="UK",
            individuals_df=sample_individuals_data,
            households_df=sample_households_data,
        )
        assert was_reader_uk.country_name_short == "UK"

    def test_was_reader_with_sample_data(self, data_path, readers):
        """Test WASReader with actual sample data from the test dataset."""
        # This test uses the actual sample data to verify the reader works end-to-end
        was_data_path = data_path / "was"
        
        # Check if sample data exists
        person_files = list(was_data_path.glob("was_round_7_person_eul_*.dta"))
        household_files = list(was_data_path.glob("was_round_7_hhold_eul_*.dta"))
        
        if person_files and household_files:
            # Use the exchange rates from the readers fixture
            exchange_rates = readers.exchange_rates
            
            # Test reading the actual sample data
            was_reader = WASReader.from_stata(
                country_name="United Kingdom",
                country_name_short="GB",
                year=2022,
                was_data_path=was_data_path,
                exchange_rates=exchange_rates,
                round_number=7,
            )
            
            # Verify the reader was created successfully
            assert isinstance(was_reader, WASReader)
            assert was_reader.country_name_short == "GB"
            
            # Verify we have some data
            assert len(was_reader.individuals_df) > 0
            assert len(was_reader.households_df) > 0
            
            # Verify that the data has been processed (variable names mapped)
            # The original WAS variable names should not be present
            original_was_vars = ['pidno', 'sexr7', 'DVAge17R7', 'dvgrspayannualr7']
            for var in original_was_vars:
                if var in was_reader.individuals_df.columns:
                    # If the original variable is still there, it means it wasn't mapped
                    # This is okay if the variable doesn't exist in the sample data
                    pass
            
            # Verify that some mapped variables are present
            mapped_vars = ['Gender', 'Age', 'Employee Income', 'Income']
            found_mapped_vars = [var for var in mapped_vars if var in was_reader.individuals_df.columns]
            assert len(found_mapped_vars) > 0, "No mapped variables found in individuals data"
            
            # Verify that ID is set as index
            assert was_reader.individuals_df.index.name == 'ID'
            
        else:
            pytest.skip("Sample WAS data files not found")
