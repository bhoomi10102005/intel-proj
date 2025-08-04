#!/usr/bin/env python3

import pandas as pd
import sys
import os

# Add src to path
sys.path.append('src')

from model_trainer import ModelTrainer

def test_single_algorithm():
    print("Testing with detailed logging...")
    
    # Load test data
    df = pd.read_csv('test_data1.csv')
    print(f"Loaded data: {len(df)} rows, {len(df.columns)} columns")
    
    feature_cols = ['feedrate', 'clamp_pressure', 'spindle_speed', 'vibration_x', 'vibration_y', 'vibration_z', 'temperature', 'current_draw']
    label_col = 'tool_condition'
    
    # Test Random Forest first
    algo_name = "Random Forest"
    algo_key = "random_forest"
    
    print(f"\n--- Testing {algo_name} with detailed logging ---")
    try:
        trainer = ModelTrainer(algorithm=algo_key)
        metrics, model = trainer.train(
            df, 
            feature_cols, 
            label_col,
            test_size=0.2,
            random_state=42,
            cross_validation=True
        )
        
        print(f"SUCCESS! Final metrics: {metrics}")
        
        # Test model saving
        model_filename = f"{algo_key}_model.pkl"
        trainer.save_model(model, filename=model_filename)
        print(f"Model saved as: {model_filename}")
        
    except Exception as e:
        print(f"FAILED: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_algorithm()
