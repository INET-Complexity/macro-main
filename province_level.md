# Plan for Province-Level Adaptation

## 1. Region Data Structure

### 1.1 Create Region Dataclass
```python
@dataclass
class Region:
    """Container for region information.
    
    This class provides a simple way to identify regions and their relationship
    to parent countries, without changing the core Country enum structure.
    
    Attributes:
        code (str): Region code (e.g., "CAN_AB" for Alberta)
        parent_country (Country): Parent country (e.g., Country.CAN)
        name (str): Full name of the region (e.g., "Alberta")
    """
    code: str
    parent_country: Country
    name: str
```

### 1.2 Extend DataConfiguration
```python
class DataConfiguration(BaseModel):
    # ... existing attributes ...
    region_disaggregation: Optional[dict[Country, list[Region]]] = None
```

## 2. Data Processing Pipeline

### 2.1 Modify DataWrapper
- Keep existing structure but add region support:
  - If a country has region_disaggregation, create SyntheticCountry for each region
  - Use parent country's exchange rates, tax rates, etc.
  - Use region-specific input-output tables and industry data
  - Split country-level data among regions based on GDP share

### 2.2 Modify SyntheticCountry
- Add region support while maintaining existing structure:
  - Add optional region attribute
  - Use region-specific data when available
  - Fall back to parent country data when needed
  - Handle inter-regional trade

## 3. Implementation Steps

### 3.1 Phase 1: Basic Structure
1. Implement Region dataclass
2. Add region_disaggregation to DataConfiguration
3. Modify DataWrapper to handle regions
4. Update SyntheticCountry to support regions

### 3.2 Phase 2: Data Processing
1. Add region-specific data readers
2. Implement data disaggregation methods
3. Handle inter-regional trade
4. Add validation for data consistency

### 3.3 Phase 3: Testing
1. Test with single region
2. Test with multiple regions
3. Validate data consistency
4. Test inter-regional relationships

## 4. Key Considerations

### 4.1 Data Consistency
- Ensure regional GDP sums to national GDP
- Maintain balanced trade flows between regions
- Keep financial flows consistent across regions
- Validate population and employment data

### 4.2 Data Sources
1. Region-specific data:
   - Input-output tables
   - Industry data
   - Employment data
   - GDP data

2. Parent country data:
   - Exchange rates
   - Tax rates
   - Interest rates
   - Inflation data

### 4.3 Data Disaggregation Rules
1. Split by GDP share:
   - Government revenue
   - Tax revenue
   - Social benefits
   - Other country-level data

2. Use actual data where available:
   - Industry structure
   - Employment
   - Trade flows
   - Population statistics

## 5. Next Steps

1. Begin with Region dataclass implementation
2. Add region_disaggregation to DataConfiguration
3. Modify DataWrapper to handle regions
4. Update SyntheticCountry
5. Add region-specific data readers
6. Test with single region
7. Expand to multiple regions
8. Add validation and testing

## 6. Advantages of This Approach

1. Minimal Changes:
   - Maintains existing Country enum
   - Keeps current data structure
   - No need for new wrapper classes

2. Flexibility:
   - Can mix countries and regions
   - Easy to add new regions
   - Supports different disaggregation levels

3. Simplicity:
   - Clear separation of concerns
   - Easy to understand and maintain
   - Minimal new code

4. Compatibility:
   - Works with existing code
   - No breaking changes
   - Easy to test

## 7. Potential Challenges

1. Data Availability:
   - Some data may only be available at country level
   - Need robust disaggregation methods
   - Handle missing regional data

2. Consistency:
   - Ensure regional data sums to national totals
   - Maintain balanced inter-regional flows
   - Handle currency and exchange rates

3. Performance:
   - Increased data volume with multiple regions
   - More complex inter-regional calculations
   - Need for efficient data structures

4. Validation:
   - More complex validation rules
   - Need for region-specific checks
   - Inter-regional consistency validation 