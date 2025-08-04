#!/usr/bin/env python3

import pandas as pd
import sys
import os

# Add src to path
sys.path.append('src')

from model_trainer import ModelTrainer

def test_model_training():
    # Load test data
    df = pd.read_csv('test_data1.csv')
    print(f"Loaded data: {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    print(f"Target values: {df['tool_condition'].value_counts()}")
    
    # Test each algorithm
    algorithms = [
        ("Random Forest", "random_forest"),
        ("Decision Tree", "decision_tree"),  
        ("SVM", "svm"),
        ("Logistic Regression", "logistic_regression")
    ]
    
    feature_cols = ['feedrate', 'clamp_pressure', 'spindle_speed', 'vibration_x', 'vibration_y', 'vibration_z', 'temperature', 'current_draw']
    label_col = 'tool_condition'
    
    for algo_name, algo_key in algorithms:
        print(f"\n--- Testing {algo_name} ---")
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
            
            print(f"✅ {algo_name} trained successfully!")
            print(f"   Accuracy: {metrics['accuracy']:.3f}")
            print(f"   Precision: {metrics['precision']:.3f}")
            print(f"   Recall: {metrics['recall']:.3f}")
            print(f"   F1 Score: {metrics['f1_score']:.3f}")
            
            # Test model saving
            model_filename = f"{algo_key}_model.pkl"
            trainer.save_model(model, filename=model_filename)
            print(f"   Model saved as: {model_filename}")
            
        except Exception as e:
            print(f"❌ {algo_name} failed: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_model_training()
