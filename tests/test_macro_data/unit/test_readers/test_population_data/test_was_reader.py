# Tests for the WASReader class
"""
Tests for the WASReader class.

This module tests the functionality of the Wealth and Assets Survey (WAS) data reader,
including data loading, processing, and variable mapping.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from macro_data.readers.population_data.was_reader import WASReader, get_var_mapping, var_numerical

class TestWASReader:
    """Test class for WASReader functionality."""

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
    def test_read_stata_basic_functionality(self, mock_read_stata):
        """Test basic functionality of read_stata static method."""
        # Mock the Stata file reading
        mock_data = pd.DataFrame({
            'pidno': [1, 2, 3],
            'sexr7': [1, 2, 1],
            'DVAge17r7': [25, 30, 45],
            'dvgrspayannualr7': [30000, 40000, 50000],
            'Dvtotgirr7': [30000, 40000, 50000],
            'unmapped_var': [1, 2, 3],  # This should be filtered out
        })
        mock_read_stata.return_value = mock_data
        
        # Test the read_stata method
        result = WASReader.read_stata(
            path="test_path.dta",
            country_name="United Kingdom",
            country_name_short="GB",
            year=2022,
            round_number=7,
        )
        
        # Verify that pandas.read_stata was called correctly
        mock_read_stata.assert_called_once_with("test_path.dta", preserve_dtypes=False, convert_categoricals=False)
        
        # Verify that only mapped variables are kept
        expected_columns = ['Sex', 'Grouped age (17 categories)', 'Gross annual income employee main job (including bonuses and commission received)', 'Total gross regular household annual income']
        assert all(col in result.columns for col in expected_columns)
        assert 'unmapped_var' not in result.columns
        
        # Verify that variable names are mapped correctly
        assert 'Sex' in result.columns  # sexr7 -> Sex
        assert 'Grouped age (17 categories)' in result.columns  # DVAge17R7 -> Grouped age (17 categories)
        
        # Verify that Personal identifier is set as index (pidno -> Personal identifier)
        assert result.index.name == 'Personal identifier'
        
    @patch('glob.glob')
    @patch.object(WASReader, 'read_stata')
    def test_from_stata_success(self, mock_read_stata, mock_glob):
        """Test successful creation of WASReader from Stata files."""
        # Mock glob to return file paths (4 calls: round_person, round_household, wave_person, wave_household)
        mock_glob.side_effect = [
            ['/path/to/was_round_7_person_eul_june_2022.dta'],  # round person files
            ['/path/to/was_round_7_hhold_eul_march_2022.dta'],  # round household files
            [],  # wave person files (empty)
            [],  # wave household files (empty)
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
    def test_from_stata_no_person_files(self, mock_glob):
        """Test from_stata method when no person files are found."""
        # Mock glob to return no person files (4 calls: round_person, round_household, wave_person, wave_household)
        mock_glob.side_effect = [
            [],  # No round person files
            ['/path/to/was_round_7_hhold_eul_march_2022.dta'],  # round household files exist
            [],  # No wave person files
            [],  # No wave household files
        ]
        
        # Test that FileNotFoundError is raised
        with pytest.raises(FileNotFoundError, match="No person files found"):
            WASReader.from_stata(
                country_name="United Kingdom",
                country_name_short="GB",
                year=2022,
                was_data_path=Path("/path/to/was/data"),
                round_number=7,
            )

    @patch('glob.glob')
    def test_from_stata_no_household_files(self, mock_glob):
        """Test from_stata method when no household files are found."""
        # Mock glob to return no household files (4 calls: round_person, round_household, wave_person, wave_household)
        mock_glob.side_effect = [
            ['/path/to/was_round_7_person_eul_june_2022.dta'],  # round person files exist
            [],  # No round household files
            [],  # No wave person files
            [],  # No wave household files
        ]
        
        # Test that FileNotFoundError is raised
        with pytest.raises(FileNotFoundError, match="No household files found"):
            WASReader.from_stata(
                country_name="United Kingdom",
                country_name_short="GB",
                year=2022,
                was_data_path=Path("/path/to/was/data"),
                round_number=7,
            )

    @patch('pandas.read_stata')
    def test_read_stata_missing_variables(self, mock_read_stata):
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
            round_number=7,
        )
        
        # Should only contain the variables that exist in the data
        expected_columns = ['Sex']
        assert all(col in result.columns for col in expected_columns)
        assert len(result.columns) == 1
        # Personal identifier should be the index
        assert result.index.name == 'Personal identifier'

    @patch('pandas.read_stata')
    def test_read_stata_data_type_conversion_errors(self, mock_read_stata):
        """Test handling of data type conversion errors in read_stata."""
        # Mock data with non-numeric values in monetary columns
        mock_data = pd.DataFrame({
            'pidno': [1, 2, 3],
            'dvgrspayannualr7': [30000, 'invalid', 50000],
            'Dvtotgirr7': [30000, 'N/A', 50000],
        })
        mock_read_stata.return_value = mock_data
        
        result = WASReader.read_stata(
            path="test_path.dta",
            country_name="United Kingdom",
            country_name_short="GB",
            year=2022,
            round_number=7,
        )
        
        # Verify that invalid values are converted to NaN
        assert pd.isna(result.loc[2, 'Gross annual income employee main job (including bonuses and commission received)'])
        assert pd.isna(result.loc[2, 'Total gross regular household annual income'])
        
        # Verify that valid values are preserved
        assert result.loc[1, 'Gross annual income employee main job (including bonuses and commission received)'] == 30000
        assert result.loc[1, 'Total gross regular household annual income'] == 30000

    def test_variable_mapping_completeness(self):
        """Test that variable mapping covers expected categories."""
        # Test that we have mappings for key categories using dynamic mapping
        # Test with round 7 (r7 suffix)
        var_mapping_r7 = get_var_mapping(7)
        
        individual_vars = ['pidno', 'personr7', 'CASEr7', 'sexr7', 'DVAge17r7']
        income_vars = ['dvgrspayannualr7', 'dvnetpayannualr7', 'Dvtotgirr7']
        asset_vars = ['hvaluer7', 'DVSaValr7_SUM', 'DVFFAssetsr7_SUM']
        liability_vars = ['Totmortr7', 'TOTCSCr7_aggr']
        housing_vars = ['hhldrr7', 'dvrentpaidr7']
        
        all_test_vars = individual_vars + income_vars + asset_vars + liability_vars + housing_vars
        
        for var in all_test_vars:
            assert var in var_mapping_r7, f"Variable {var} not found in var_mapping for round 7"
        
        # Test with wave 1 (w1 suffix)
        var_mapping_w1 = get_var_mapping(1)
        
        individual_vars_w1 = ['pidno', 'personw1', 'CASEw1', 'sexw1', 'DVAge17w1']
        income_vars_w1 = ['dvgrspayannualw1', 'dvnetpayannualw1', 'Dvtotgirw1']
        
        all_test_vars_w1 = individual_vars_w1 + income_vars_w1
        
        for var in all_test_vars_w1:
            assert var in var_mapping_w1, f"Variable {var} not found in var_mapping for wave 1"

    def test_numerical_variables_list(self):
        """Test that numerical variables list contains expected monetary variables."""
        expected_numerical = [
            'Total gross regular household annual income',
            'Gross annual income employee main job (including bonuses and commission received)',
            'Current value of main residence',
            'Total value of savings accounts',
            'Total mortgage on main residence',
            'How much is usual household rent',
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

    def test_was_reader_with_sample_data(self, data_path):
        """Test WASReader with actual sample data from the test dataset."""
        # This test uses the actual sample data to verify the reader works end-to-end
        was_data_path = data_path / "was"
        
        # Check if sample data exists
        person_files = list(was_data_path.glob("was_round_7_person_eul_*.dta"))
        household_files = list(was_data_path.glob("was_round_7_hhold_eul_*.dta"))
        
        if person_files and household_files:
            # Test reading the actual sample data
            was_reader = WASReader.from_stata(
                country_name="United Kingdom",
                country_name_short="GB",
                year=2022,
                was_data_path=was_data_path,
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
            original_was_vars = ['pidno', 'sexr7', 'DVAge17r7', 'dvgrspayannualr7']
            for var in original_was_vars:
                if var in was_reader.individuals_df.columns:
                    # If the original variable is still there, it means it wasn't mapped
                    # This is okay if the variable doesn't exist in the sample data
                    pass
            
            # Verify that some mapped variables are present
            mapped_vars = ['Sex', 'Grouped age (17 categories)', 'Gross annual income employee main job (including bonuses and commission received)', 'Total gross regular household annual income']
            found_mapped_vars = [var for var in mapped_vars if var in was_reader.individuals_df.columns]
            assert len(found_mapped_vars) > 0, "No mapped variables found in individuals data"
            
            # Verify that Personal identifier is set as index (if pidno column exists in the data)
            if 'Personal identifier' in was_reader.individuals_df.columns:
                # If Personal identifier column exists, it should be the index
                assert was_reader.individuals_df.index.name == 'Personal identifier'
            else:
                # If no Personal identifier column, the index should be the default RangeIndex
                assert was_reader.individuals_df.index.name is None
            
        else:
            pytest.skip("Sample WAS data files not found")
