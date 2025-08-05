import joblib
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def train_model(X, y, model_path='models/random_forest_model_main.pkl'):
    """Train a Random Forest model and save it"""
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    # Split data for validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Validate model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"Model trained with accuracy: {accuracy:.3f}")
    
    # Save model
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    return model

def load_model(path='models/random_forest_model_main.pkl'):
    """Load a trained model, create one if it doesn't exist"""
    if not os.path.exists(path):
        print(f"Model not found at {path}. Training a new model...")
        
        # Load training data
        train_data_path = 'data/train.csv'
        if not os.path.exists(train_data_path):
            raise FileNotFoundError(f"Training data not found at {train_data_path}")
        
        # Load and prepare data
        df = pd.read_csv(train_data_path)
        
        # Prepare features and target
        if 'tool_condition' in df.columns:
            # Map target to numeric values
            target_map = {'unworn': 0, 'worn': 1}
            y = df['tool_condition'].map(target_map)
            
            # Select feature columns (exclude target)
            feature_cols = [col for col in df.columns if col != 'tool_condition']
            X = df[feature_cols]
            
            # Train new model
            model = train_model(X, y, path)
            return model
        else:
            raise ValueError("Training data must contain 'tool_condition' column")
    
    return joblib.load(path)

def predict(model, input_df):
    """Make predictions using the trained model"""
    return model.predict(input_df)
