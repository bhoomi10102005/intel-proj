"""
Test script to validate the new data validation in ModelTrainer
"""
import pandas as pd
import sys
import os
sys.path.append('src')

from model_trainer import ModelTrainer

def test_continuous_target_validation():
    """Test that ModelTrainer properly rejects continuous targets for classification"""
    
    print("🧪 Testing continuous target validation...")
    
    # Create test data with continuous target (like feedrate) - more realistic data
    test_data = pd.DataFrame({
        'feedrate': [1.2, 2.1, 1.8, 2.5, 1.9, 3.2, 3.8, 2.8, 1.5, 2.3, 1.6, 2.7, 3.1, 2.9, 1.7, 2.4],
        'clamp_pressure': [4.0, 3.5, 4.2, 3.8, 4.1, 3.9, 3.7, 4.0, 3.6, 4.3, 3.4, 4.1, 3.8, 3.9, 4.0, 3.7],
        'tool_condition': ['unworn', 'unworn', 'worn', 'worn', 'unworn', 'worn', 'worn', 'unworn', 
                          'unworn', 'worn', 'unworn', 'worn', 'worn', 'unworn', 'unworn', 'worn']
    })
    
    print(f"📊 Test data shape: {test_data.shape}")
    print(f"📊 Columns: {list(test_data.columns)}")
    print(f"📊 feedrate unique values: {len(test_data['feedrate'].unique())} - {sorted(test_data['feedrate'].unique())}")
    print(f"📊 tool_condition unique values: {len(test_data['tool_condition'].unique())} - {list(test_data['tool_condition'].unique())}")
    
    # Test 1: Try to use continuous feedrate as target (should fail)
    print("\n🔥 Test 1: Using continuous 'feedrate' as target (should FAIL)")
    try:
        trainer = ModelTrainer(algorithm='random_forest')
        result = trainer.train(
            df=test_data,
            feature_cols=['clamp_pressure'],
            label_col='feedrate',  # This is continuous!
            test_size=0.3,
            random_state=42
        )
        print("❌ ERROR: Should have failed but didn't!")
        return False
    except ValueError as e:
        if "continuous target" in str(e) or "Classification algorithms expect discrete classes" in str(e):
            print("✅ SUCCESS: Correctly rejected continuous target!")
            print(f"   Error message: {str(e)[:150]}...")
        else:
            print(f"❌ FAILED: Wrong error message: {e}")
            return False
    except Exception as e:
        print(f"❌ FAILED: Unexpected error: {e}")
        return False
    
    # Test 2: Try to use categorical tool_condition as target (should succeed)
    print("\n✅ Test 2: Using categorical 'tool_condition' as target (should SUCCEED)")
    try:
        trainer = ModelTrainer(algorithm='random_forest')
        result = trainer.train(
            df=test_data,
            feature_cols=['feedrate', 'clamp_pressure'],
            label_col='tool_condition',  # This is categorical!
            test_size=0.3,
            random_state=42
        )
        print("✅ SUCCESS: Correctly accepted categorical target!")
        metrics, model = result
        print(f"   Metrics: {metrics}")
        return True
    except Exception as e:
        print(f"❌ FAILED: Should have succeeded: {e}")
        return False

if __name__ == "__main__":
    print("🔬 Running ModelTrainer validation tests...")
    print("="*60)
    
    success = test_continuous_target_validation()
    
    print("\n" + "="*60)
    if success:
        print("🎉 ALL TESTS PASSED! Validation is working correctly.")
        print("💡 Users will now get helpful error messages when selecting wrong label columns.")
    else:
        print("❌ TESTS FAILED! Check the validation logic.")
    
    print("="*60)
