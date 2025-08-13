#!/usr/bin/env python3
"""
Train the model with the new name: random_forest_model_main.pkl
"""

import pandas as pd
import os
import sys

# Add src to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)

from src.model import train_model

def main():
    print("🚀 Training model with new name: random_forest_model_main.pkl")
    
    # Load training data
    train_data_path = 'data/train2.csv'
    if not os.path.exists(train_data_path):
        print(f"❌ Training data not found at {train_data_path}")
        return
    
    print(f"📊 Loading training data from {train_data_path}")
    df = pd.read_csv(train_data_path)
    print(f"📈 Data shape: {df.shape}")
    print(f"📋 Columns: {list(df.columns)}")
    
    # Check if we have the required columns
    if 'tool_condition' not in df.columns:
        print("❌ Error: 'tool_condition' column not found in training data")
        return
    
    # Prepare target variable
    target_map = {'unworn': 0, 'worn': 1}
    y = df['tool_condition'].map(target_map)
    
    print(f"🎯 Target distribution:")
    print(df['tool_condition'].value_counts())
    
    # Select feature columns (numeric columns only, excluding non-feature columns)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in ['No']]
    
    if not feature_cols:
        print("❌ Error: No numeric feature columns found")
        return
    
    print(f"🔧 Selected features: {feature_cols}")
    X = df[feature_cols]
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    print(f"📊 Feature matrix shape: {X.shape}")
    print(f"🎯 Target vector shape: {y.shape}")
    
    # Specify the new model path
    new_model_path = 'models/random_forest_model_main.pkl'
    
    # Train the model with the new name
    try:
        print(f"🎯 Training model and saving as: {new_model_path}")
        model = train_model(X, y, model_path=new_model_path)
        print("✅ Model training completed successfully!")
        print(f"💾 Model saved as: {new_model_path}")
        
        # Test the model loading
        print("\n🧪 Testing model loading...")
        from src.model import load_model
        loaded_model = load_model(path=new_model_path)
        print("✅ Model loading test successful!")
        
        # Test prediction
        print("\n🔮 Testing prediction...")
        import numpy as np
        sample_data = pd.DataFrame({
            col: [X[col].mean()] for col in feature_cols
        })
        prediction = loaded_model.predict(sample_data)
        print(f"✅ Sample prediction: {prediction[0]} ({'worn' if prediction[0] == 1 else 'unworn'})")
        
        print("\n🎉 All tests passed! The new model is ready to use.")
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
