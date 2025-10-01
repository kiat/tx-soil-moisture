#!/usr/bin/env python3
"""
Unit tests for the new label generation functionality in data_to_X_y.
Tests cover all label types, edge cases, and backward compatibility.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from core.data_helpers import data_to_X_y


def test_backward_compatibility():
    """Test that label_type='point' produces exactly the same results as the original implementation."""
    print("Testing backward compatibility...")
    
    # Create test data
    np.random.seed(42)
    data = np.random.rand(100, 6)  # 100 timesteps, 6 features
    window_size = 10
    offset = 5
    
    # Test with point label type (should match original behavior)
    X_point, y_point = data_to_X_y(data, window_size, offset, label_type="point")
    
    # Manual calculation of what the original function should produce
    rows = len(data) - window_size - offset
    expected_y = data[window_size + offset - 1: window_size + offset - 1 + rows, 0]
    
    assert len(y_point) == len(expected_y), f"Length mismatch: {len(y_point)} vs {len(expected_y)}"
    assert np.allclose(y_point, expected_y), "Point labels don't match original implementation"
    assert X_point.shape == (rows, window_size, 6), f"X shape mismatch: {X_point.shape}"
    
    print("✓ Backward compatibility test passed")


def test_rolling_mean_basic():
    """Test basic rolling mean functionality."""
    print("Testing rolling mean basic functionality...")
    
    # Create simple test data where we can verify the mean manually
    data = np.arange(50).reshape(50, 1).astype(float)  # [0, 1, 2, ..., 49] in column 0
    window_size = 5
    offset = 0  # No original offset
    agg_hours = 3
    offset_hours = 0
    samples_per_hour = 1
    
    X, y = data_to_X_y(data, window_size, offset, 
                       label_type="rolling_mean", agg_hours=agg_hours, 
                       offset_hours=offset_hours, samples_per_hour=samples_per_hour)
    
    # For the first window [0,1,2,3,4], with offset=0, offset_hours=0:
    # target_end_idx = 0 + 5 + 0 = 5
    # target_start_idx = 5 - 3 + 1 = 3  
    # target should be mean of [3,4,5] = 4.0
    expected_first_target = np.mean([3, 4, 5])  # 4.0
    expected_second_target = np.mean([4, 5, 6])  # 5.0
    
    assert np.isclose(y[0], expected_first_target), f"First target: {y[0]} vs {expected_first_target}"
    assert np.isclose(y[1], expected_second_target), f"Second target: {y[1]} vs {expected_second_target}"
    
    print("✓ Rolling mean basic test passed")


def test_rolling_mean_with_offset():
    """Test rolling mean with forecast offset."""
    print("Testing rolling mean with offset...")
    
    data = np.arange(50).reshape(50, 1).astype(float)
    window_size = 5
    offset = 2  # Original offset
    agg_hours = 3
    offset_hours = 1  # Additional forecast offset
    samples_per_hour = 1
    
    X, y = data_to_X_y(data, window_size, offset, 
                       label_type="rolling_mean", agg_hours=agg_hours, 
                       offset_hours=offset_hours, samples_per_hour=samples_per_hour)
    
    # For the first window [0,1,2,3,4], with total offset = 2 + 1 = 3
    # Target end is at index 5 + 3 = 8
    # Rolling mean window is [8-3+1:8+1] = [6,7,8], mean = 7.0
    expected_first_target = np.mean([6, 7, 8])
    
    assert len(y) > 0, "No targets generated"
    assert np.isclose(y[0], expected_first_target), f"First target with offset: {y[0]} vs {expected_first_target}"
    
    print("✓ Rolling mean with offset test passed")


def test_daily_mean():
    """Test daily mean functionality."""
    print("Testing daily mean functionality...")
    
    # Create 72 hours of hourly data (3 days)
    data = np.arange(72).reshape(72, 1).astype(float)
    window_size = 24  # 24 hours
    offset = 0
    offset_hours = 0
    samples_per_hour = 1
    
    X, y = data_to_X_y(data, window_size, offset, 
                       label_type="daily_mean", 
                       offset_hours=offset_hours, samples_per_hour=samples_per_hour)
    
    # For the first window [0-23], target end is at 24 + 0 = 24
    # Daily mean window is [24-24+1:24+1] = [1:25] = [1-24], mean = 12.5
    expected_first_target = np.mean(np.arange(1, 25))
    
    assert len(y) > 0, "No targets generated for daily mean"
    assert np.isclose(y[0], expected_first_target), f"Daily mean target: {y[0]} vs {expected_first_target}"
    
    print("✓ Daily mean test passed")


def test_insufficient_history():
    """Test handling of insufficient history."""
    print("Testing insufficient history handling...")
    
    # Create very short data that can't support the required aggregation
    data = np.arange(10).reshape(10, 1).astype(float)
    window_size = 5
    offset = 0
    agg_hours = 10  # Requires 10 samples, but we only have 10 total
    
    X, y = data_to_X_y(data, window_size, offset, 
                       label_type="rolling_mean", agg_hours=agg_hours)
    
    # Should return empty arrays when insufficient data
    assert len(X) == 0 and len(y) == 0, "Should return empty arrays for insufficient data"
    
    print("✓ Insufficient history test passed")


def test_samples_per_hour():
    """Test non-hourly data with samples_per_hour parameter."""
    print("Testing samples_per_hour functionality...")
    
    # Create 30-minute data (2 samples per hour) for 48 hours
    data = np.arange(96).reshape(96, 1).astype(float)  # 96 samples = 48 hours at 2 samples/hour
    window_size = 48  # 24 hours of 30-min data
    offset = 0
    agg_hours = 12  # 12 hours
    samples_per_hour = 2
    
    X, y = data_to_X_y(data, window_size, offset, 
                       label_type="rolling_mean", agg_hours=agg_hours,
                       samples_per_hour=samples_per_hour)
    
    # agg_hours=12 with samples_per_hour=2 means we need 24 samples for aggregation
    # First window is [0-47], target_end is at 48 + 0 = 48
    # Rolling mean window is [48-24+1:48+1] = [25:49] = [25-48], mean = 36.5
    expected_first_target = np.mean(np.arange(25, 49))
    
    assert len(y) > 0, "No targets generated with samples_per_hour"
    assert np.isclose(y[0], expected_first_target), f"Samples per hour target: {y[0]} vs {expected_first_target}"
    
    print("✓ Samples per hour test passed")


def test_edge_cases():
    """Test various edge cases."""
    print("Testing edge cases...")
    
    # Test with minimum viable data
    data = np.arange(30).reshape(30, 1).astype(float)
    window_size = 5
    offset = 1
    agg_hours = 3
    
    X, y = data_to_X_y(data, window_size, offset, 
                       label_type="rolling_mean", agg_hours=agg_hours)
    
    assert len(X) == len(y), "X and y should have same length"
    assert X.shape[1] == window_size, "Window size should be preserved"
    assert X.shape[2] == 1, "Feature dimension should be preserved"
    
    # Test with invalid label type
    try:
        data_to_X_y(data, window_size, offset, label_type="invalid")
        assert False, "Should raise ValueError for invalid label_type"
    except ValueError:
        pass
    
    print("✓ Edge cases test passed")


def run_all_tests():
    """Run all unit tests."""
    print("Running comprehensive unit tests for label generation...")
    print("=" * 60)
    
    test_backward_compatibility()
    test_rolling_mean_basic()
    test_rolling_mean_with_offset()
    test_daily_mean()
    test_insufficient_history()
    test_samples_per_hour()
    test_edge_cases()
    
    print("=" * 60)
    print("✅ All tests passed successfully!")
    print("\nUnit test coverage:")
    print("- Exact equality with manual np.mean on toy series ✓")
    print("- offset_hours=0 vs offset_hours>0 correctness ✓")
    print("- Masking/skip behavior when insufficient history ✓")
    print("- Model input/target shapes unchanged ✓")
    print("- Backward compatibility with point labels ✓")


if __name__ == "__main__":
    run_all_tests()
