# Test Plan: ML Algorithm Improvements

This document outlines comprehensive tests for the improved ML algorithm and artifact management system, covering all new features implemented.

## Summary

The implementation includes:
- **Dataset validation system**: Checks for duplicates, missing values, invalid values
- **Dataset statistics computation**: Calculates mean/min/max and other metrics
- **Parquet storage support**: Efficient columnar storage format
- **Uncertainty calibration**: Isotonic regression for probability calibration
- **Incremental training**: Updates existing models with new data

## Test Files to Create

### New Test File: `test_ml_improvements.py`

#### 1. Dataset Validation Tests
- **`test_validate_dataset_valid`**: Verify valid datasets pass validation
- **`test_validate_dataset_duplicates`**: Test with duplicate rows
- **`test_validate_dataset_missing_values`**: Test with missing fields
- **`test_validate_dataset_invalid_values`**: Test with invalid numeric values

#### 2. Dataset Statistics Tests
- **`test_compute_dataset_statistics`**: Verify statistics computation (row count, unique builds/scenarios, mean/min/max values)

#### 3. Parquet Storage Tests
- **`test_parquet_storage_roundtrip`**: Test writing and reading Parquet files
- Verify Parquet file creation and manifest metadata
- Test auto-detection of file format

#### 4. Incremental Training Tests
- **`test_incremental_train_basic`**: Test basic incremental training functionality
- Verify metadata includes parent model reference
- Test training on new data with existing model as starting point

#### 5. Uncertainty Calibration Tests
- **`test_calibration_ece_computation`**: Test Expected Calibration Error (ECE) calculation
- Verify calibration with isotonic regression

#### 6. Integration Tests
- **`test_full_pipeline`**: End-to-end test of the entire ML pipeline
- Validates complete workflow from dataset creation to prediction

## Existing Test Files to Update

### 1. `test_dataset_snapshot.py`
- Add test for `dataset_format` parameter
- Verify Parquet format support in snapshot creation
- Update existing tests to accept `dataset_format` parameter

### 2. `test_surrogate_model.py`
- Add test for `incremental_train` function
- Test calibration functionality
- Verify calibration is applied to predictions

## Test Infrastructure

### Test Dependencies
- **pytest**: Test framework
- **pandas**: Data handling for Parquet
- **pyarrow**: Parquet file support
- **scikit-learn**: Isotonic regression for calibration

### Test Data Generation
- Create synthetic build data using existing `_create_build` and `_build_metrics` helpers
- Generate datasets with varying quality to test validation
- Create diverse metrics to test calibration and statistics

## Expected Outcomes

All tests should verify:
1. **Validation system correctly identifies issues**
2. **Statistics accurately describe dataset properties**
3. **Parquet files are created and read correctly**
4. **Incremental training produces valid models**
5. **Calibration improves prediction reliability**

## Running Tests

### Command Line
```bash
# Install test dependencies
cd backend
pip install -e ".[ml,dev]"

# Run specific test file
python -m pytest tests/test_ml_improvements.py -v

# Run all ML-related tests
python -m pytest tests/test_surrogate_model.py tests/test_dataset_snapshot.py tests/test_ml_improvements.py -v

# Run with coverage
python -m pytest tests/test_ml_improvements.py --cov=backend.engine.surrogate --cov-report=html
```

### Using Makefile
```bash
make backend-test
```

## Coverage Targets

- **Dataset validation system**: 85%+ coverage
- **Dataset statistics**: 90%+ coverage  
- **Parquet storage**: 80%+ coverage
- **Incremental training**: 80%+ coverage
- **Calibration**: 75%+ coverage

## Integration with CI/CD

These tests should be added to the existing CI pipeline:
1. Run on PR creation
2. Run on main branch updates
3. Integrate with code coverage tools

This comprehensive test plan ensures the new features are robust, reliable, and correctly integrated with the existing system.
