"""
Model Training Module
Provides functionality for training and evaluating machine learning models
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    A class to handle model training and evaluation
    """
    
    def __init__(self, algorithm=None):
        logger.info(f"Initializing ModelTrainer with algorithm: {algorithm}")
        self.available_algorithms = {
            'random_forest': RandomForestClassifier(random_state=42),
            'decision_tree': DecisionTreeClassifier(random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'svm': SVC(random_state=42, probability=True),
            'knn': KNeighborsClassifier()
        }
        
        self.algorithm = algorithm
        self.trained_models = {}
        self.evaluation_results = {}
        
        logger.info(f"Available algorithms: {list(self.available_algorithms.keys())}")
        if algorithm and algorithm not in self.available_algorithms:
            logger.error(f"Algorithm '{algorithm}' not found in available algorithms")
        else:
            logger.info(f"ModelTrainer initialized successfully with algorithm: {algorithm}")
    
    def prepare_data(self, df, target_column, feature_columns=None):
        """
        Prepare data for training
        """
        logger.info(f"Preparing data - DataFrame shape: {df.shape}")
        logger.info(f"Target column: {target_column}")
        logger.info(f"Feature columns: {feature_columns}")
        logger.info(f"DataFrame columns: {list(df.columns)}")
        
        # Handle target column encoding
        if target_column in df.columns:
            logger.info(f"Target column '{target_column}' found in DataFrame")
            logger.info(f"Target column dtype: {df[target_column].dtype}")
            logger.info(f"Target column unique values: {df[target_column].unique()}")
            
            # IMPORTANT: Validate that we're not doing regression with classification algorithms
            y_series = df[target_column]
            if y_series.dtype in ['float64', 'float32', 'int64', 'int32'] and len(y_series.unique()) > 5:
                logger.error("🚨 CLASSIFICATION vs REGRESSION MISMATCH DETECTED! 🚨")
                logger.error(f"Target column '{target_column}' appears to be CONTINUOUS/NUMERIC with {len(y_series.unique())} unique values")
                logger.error(f"Sample values: {sorted(y_series.unique())[:10]}")
                logger.error("Classification algorithms expect DISCRETE CLASSES (like 'worn'/'unworn', 0/1, etc.)")
                logger.error("")
                logger.error("💡 SOLUTION SUGGESTIONS:")
                logger.error("1. If you want to predict tool condition, use 'tool_condition' as the label column")
                logger.error("2. If you want to predict a continuous value, use regression algorithms instead")
                logger.error("3. If this should be classification, convert continuous values to discrete classes")
                logger.error("")
                
                # Check if tool_condition column exists as a better alternative
                if 'tool_condition' in df.columns:
                    tool_condition_values = df['tool_condition'].unique()
                    logger.error(f"📋 RECOMMENDED: Use 'tool_condition' as label instead (values: {tool_condition_values})")
                
                raise ValueError(f"Cannot use classification algorithms with continuous target '{target_column}'. "
                               f"Target has {len(y_series.unique())} unique continuous values: {sorted(y_series.unique())[:10]}... "
                               f"Classification algorithms expect discrete classes like 'worn'/'unworn' or 0/1. "
                               f"Either use 'tool_condition' as label or switch to regression algorithms.")
            
            if df[target_column].dtype == 'object':
                logger.info("Target column is object type, applying mapping")
                # Encode categorical target
                target_mapping = {'unworn': 0, 'worn': 1}
                y = df[target_column].map(target_mapping)
                logger.info(f"Mapped values: {y.value_counts().to_dict()}")
                
                if y.isnull().any():
                    logger.warning("Some values couldn't be mapped, using label encoding")
                    # If mapping doesn't work, use label encoding
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    y = le.fit_transform(df[target_column])
                    logger.info(f"Label encoded values: {np.unique(y)}")
            else:
                logger.info("Target column is numeric with few classes, treating as discrete")
                y = df[target_column]
        else:
            logger.error(f"Target column '{target_column}' not found in dataset")
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        # Prepare features
        if feature_columns is None:
            logger.info("No feature columns specified, selecting numeric columns")
            # Use all numeric columns except target
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            logger.info(f"Found numeric columns: {numeric_cols}")
            if target_column in numeric_cols:
                numeric_cols.remove(target_column)
                logger.info(f"Removed target column, remaining: {numeric_cols}")
            feature_columns = numeric_cols
        else:
            # Validate that all specified feature columns exist in the dataset
            missing_features = [col for col in feature_columns if col not in df.columns]
            if missing_features:
                logger.error(f"❌ Missing feature columns: {missing_features}")
                logger.error(f"Available columns: {list(df.columns)}")
                available_features = [col for col in feature_columns if col in df.columns]
                if available_features:
                    logger.warning(f"⚠️ Using only available features: {available_features}")
                    feature_columns = available_features
                else:
                    raise ValueError(f"None of the specified feature columns {feature_columns} exist in dataset. Available columns: {list(df.columns)}")
            else:
                logger.info(f"✅ All specified feature columns exist in dataset")
        
        # Check if label column is accidentally included in features
        if target_column in feature_columns:
            logger.warning(f"⚠️ Label column '{target_column}' is included in features! Removing it to prevent data leakage.")
            feature_columns = [col for col in feature_columns if col != target_column]
            logger.info(f"Updated feature columns (removed label): {feature_columns}")
        
        if not feature_columns:
            logger.error("No suitable feature columns found")
            raise ValueError("No suitable feature columns found")
        
        logger.info(f"Final feature columns: {feature_columns}")
        X = df[feature_columns]
        logger.info(f"Feature matrix shape: {X.shape}")
        
        # Handle missing values
        missing_count = X.isnull().sum().sum()
        if missing_count > 0:
            logger.warning(f"Found {missing_count} missing values, filling with mean")
            X = X.fillna(X.mean())
        else:
            logger.info("No missing values found in features")
        
        logger.info(f"Data preparation completed - X shape: {X.shape}, y shape: {y.shape}")
        logger.info(f"Target distribution: {pd.Series(y).value_counts().to_dict()}")
        
        # Warn about small dataset size
        n_samples = X.shape[0]
        if n_samples < 50:
            logger.warning(f"⚠️ Very small dataset detected ({n_samples} samples)")
            logger.warning("⚠️ Results may not be reliable with such a small dataset")
            logger.warning("⚠️ Consider collecting more data for better model performance")
        elif n_samples < 100:
            logger.warning(f"⚠️ Small dataset detected ({n_samples} samples)")
            logger.warning("⚠️ Consider collecting more data for more robust results")
        
        return X, y, feature_columns
    
    def train(self, df, feature_cols, label_col, test_size=0.2, random_state=42, cross_validation=True):
        """
        Train method expected by app.py
        """
        logger.info("="*50)
        logger.info("STARTING TRAINING PROCESS")
        logger.info("="*50)
        logger.info(f"Algorithm: {self.algorithm}")
        logger.info(f"DataFrame shape: {df.shape}")
        logger.info(f"Feature columns: {feature_cols}")
        logger.info(f"Label column: {label_col}")
        logger.info(f"Test size: {test_size}")
        logger.info(f"Random state: {random_state}")
        logger.info(f"Cross validation: {cross_validation}")
        
        try:
            # Prepare data
            logger.info("Step 1: Preparing data...")
            X, y, feature_columns = self.prepare_data(df, label_col, feature_cols)
            logger.info("✅ Data preparation successful")
            
            # Use the algorithm specified in constructor
            logger.info("Step 2: Validating algorithm...")
            if self.algorithm not in self.available_algorithms:
                logger.error(f"Algorithm '{self.algorithm}' not available")
                logger.error(f"Available algorithms: {list(self.available_algorithms.keys())}")
                raise ValueError(f"Algorithm '{self.algorithm}' not available")
            logger.info(f"✅ Algorithm '{self.algorithm}' validated")
            
            # Train the model
            logger.info("Step 3: Training model...")
            model, metrics = self.train_model(X, y, self.algorithm, test_size)
            logger.info("✅ Model training successful")
            
            logger.info("="*50)
            logger.info("TRAINING COMPLETED SUCCESSFULLY")
            logger.info("="*50)
            logger.info(f"Final metrics: {metrics}")
            
            return metrics, model
            
        except Exception as e:
            logger.error("="*50)
            logger.error("TRAINING FAILED")
            logger.error("="*50)
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise Exception(f"Training failed: {str(e)}")
    
    def save_model(self, model, filename="model.pkl"):
        """
        Save model to file with given filename
        """
        logger.info(f"Saving model to: {filename}")
        logger.info(f"Model type: {type(model).__name__}")
        logger.info(f"Algorithm: {self.algorithm}")
        
        try:
            model_path = f"models/{filename}"
            logger.info(f"Full model path: {os.path.abspath(model_path)}")
            
            # Ensure models directory exists
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            logger.info("Models directory created/verified")
            
            model_data = {
                'model': model,
                'algorithm': self.algorithm,
                'feature_columns': getattr(self, 'feature_columns', [])
            }
            
            logger.info(f"Saving model data: {list(model_data.keys())}")
            logger.info(f"Feature columns: {model_data['feature_columns']}")
            
            joblib.dump(model_data, model_path)
            logger.info(f"✅ Model saved successfully to {model_path}")
            
            # Verify file was created
            if os.path.exists(model_path):
                file_size = os.path.getsize(model_path)
                logger.info(f"Verified: File exists with size {file_size} bytes")
            else:
                logger.error("❌ File was not created!")
                
            return True
        except Exception as e:
            logger.error(f"Error saving model:")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise Exception(f"Failed to save model: {str(e)}")

    def train_model(self, X, y, algorithm_name, test_size=0.2):
        """
        Train a model with the specified algorithm
        """
        logger.info(f"Training model with algorithm: {algorithm_name}")
        logger.info(f"Input data - X shape: {X.shape}, y shape: {y.shape}")
        logger.info(f"Test size: {test_size}")
        
        try:
            # Split data
            logger.info("Splitting data into train/test sets...")
            
            # Check if we can do stratified split
            unique_values, counts = np.unique(y, return_counts=True)
            min_count = counts.min()
            logger.info(f"Minimum class count: {min_count}")
            
            if min_count >= 2:
                # Can do stratified split
                logger.info("Using stratified train/test split")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
            else:
                # Cannot do stratified split - too few samples per class
                logger.warning("Cannot use stratified split - some classes have only 1 sample")
                logger.warning("Using regular train/test split without stratification")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
            logger.info(f"Train set - X: {X_train.shape}, y: {y_train.shape}")
            logger.info(f"Test set - X: {X_test.shape}, y: {y_test.shape}")
            logger.info(f"Train class distribution: {pd.Series(y_train).value_counts().to_dict()}")
            logger.info(f"Test class distribution: {pd.Series(y_test).value_counts().to_dict()}")
            
            # Get model
            if algorithm_name not in self.available_algorithms:
                logger.error(f"Algorithm '{algorithm_name}' not found in available algorithms")
                logger.error(f"Available: {list(self.available_algorithms.keys())}")
                raise ValueError(f"Algorithm '{algorithm_name}' not found in available algorithms")
            
            model = self.available_algorithms[algorithm_name]
            logger.info(f"Using model: {type(model).__name__}")
            logger.info(f"Model parameters: {model.get_params()}")
            
            # Train model
            logger.info("Training model...")
            model.fit(X_train, y_train)
            logger.info("✅ Model fitting completed")
            
            # Make predictions
            logger.info("Making predictions on test set...")
            y_pred = model.predict(X_test)
            logger.info(f"Predictions shape: {y_pred.shape}")
            logger.info(f"Prediction distribution: {pd.Series(y_pred).value_counts().to_dict()}")
            
            y_pred_proba = None
            if hasattr(model, 'predict_proba'):
                logger.info("Getting prediction probabilities...")
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                logger.info(f"Probabilities shape: {y_pred_proba.shape}")
            else:
                logger.info("Model doesn't support predict_proba")
            
            # Calculate metrics
            logger.info("Calculating metrics...")
            metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
            logger.info(f"Basic metrics calculated: {metrics}")
            
            # Cross-validation
            logger.info("Performing cross-validation...")
            # Adjust CV folds for small datasets
            n_samples = len(X)
            cv_folds = min(5, n_samples // 2)  # Use fewer folds for small datasets
            if cv_folds < 2:
                logger.warning(f"Dataset too small for cross-validation (n={n_samples}), skipping CV")
                metrics['cv_mean'] = metrics['accuracy']  # Use test accuracy as fallback
                metrics['cv_std'] = 0.0
            else:
                logger.info(f"Using {cv_folds}-fold cross-validation for dataset with {n_samples} samples")
                try:
                    cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
                    metrics['cv_mean'] = cv_scores.mean()
                    metrics['cv_std'] = cv_scores.std()
                    logger.info(f"CV scores: {cv_scores}")
                except Exception as cv_error:
                    logger.warning(f"Cross-validation failed: {cv_error}")
                    logger.warning("Using test accuracy as fallback")
                    metrics['cv_mean'] = metrics['accuracy']
                    metrics['cv_std'] = 0.0
            
            logger.info(f"CV mean: {metrics['cv_mean']:.4f}, CV std: {metrics['cv_std']:.4f}")
            
            # Store feature columns for later use
            self.feature_columns = X.columns.tolist()
            logger.info(f"Stored feature columns: {self.feature_columns}")
            
            # Store results
            logger.info("Storing results...")
            self.trained_models[algorithm_name] = {
                'model': model,
                'X_test': X_test,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'feature_columns': X.columns.tolist()
            }
            
            self.evaluation_results[algorithm_name] = metrics
            logger.info("✅ Results stored successfully")
            
            logger.info(f"Final metrics for {algorithm_name}: {metrics}")
            return model, metrics
            
        except Exception as e:
            logger.error(f"Error in train_model for {algorithm_name}:")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        Calculate evaluation metrics
        """
        logger.info("Calculating evaluation metrics...")
        logger.info(f"y_true shape: {y_true.shape}, unique values: {np.unique(y_true)}")
        logger.info(f"y_pred shape: {y_pred.shape}, unique values: {np.unique(y_pred)}")
        
        try:
            # Basic classification metrics
            accuracy = accuracy_score(y_true, y_pred)
            
            # Handle the case where there might be class imbalance
            unique_classes = len(np.unique(y_true))
            if unique_classes == 1:
                logger.warning("Only one class present in y_true - this is unusual!")
                logger.warning("Setting precision, recall, and f1 to accuracy value")
                precision = recall = f1 = accuracy
            else:
                # Use different averaging strategies based on the number of classes
                if unique_classes == 2:
                    # Binary classification - use binary averaging
                    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
                    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
                    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
                else:
                    # Multi-class classification - use weighted averaging
                    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            logger.info(f"Basic metrics calculated: {metrics}")
            
            # ROC AUC if probabilities available
            if y_pred_proba is not None and len(np.unique(y_true)) == 2:
                logger.info("Calculating ROC AUC...")
                try:
                    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                    metrics['roc_auc'] = auc(fpr, tpr)
                    metrics['fpr'] = fpr
                    metrics['tpr'] = tpr
                    logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
                except Exception as e:
                    logger.warning(f"Failed to calculate ROC AUC: {str(e)}")
            else:
                logger.info("Skipping ROC AUC (no probabilities or not binary classification)")
            
            # Confusion matrix
            logger.info("Calculating confusion matrix...")
            metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
            logger.info(f"Confusion matrix:\n{metrics['confusion_matrix']}")
            
            logger.info("✅ All metrics calculated successfully")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics:")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise
    
    def create_confusion_matrix_plot(self, algorithm_name):
        """
        Create confusion matrix visualization
        """
        if algorithm_name not in self.evaluation_results:
            return None
            
        cm = self.evaluation_results[algorithm_name]['confusion_matrix']
        
        fig = px.imshow(
            cm,
            text_auto=True,
            aspect="auto",
            title=f"Confusion Matrix - {algorithm_name}",
            labels=dict(x="Predicted", y="Actual"),
            color_continuous_scale="Blues"
        )
        
        fig.update_layout(
            xaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=['Unworn', 'Worn']),
            yaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=['Unworn', 'Worn'])
        )
        
        return fig
    
    def create_roc_curve_plot(self, algorithm_name):
        """
        Create ROC curve visualization
        """
        if algorithm_name not in self.evaluation_results:
            return None
            
        metrics = self.evaluation_results[algorithm_name]
        
        if 'fpr' not in metrics or 'tpr' not in metrics:
            return None
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=metrics['fpr'],
            y=metrics['tpr'],
            mode='lines',
            name=f'{algorithm_name} (AUC = {metrics["roc_auc"]:.3f})',
            line=dict(width=2)
        ))
        
        # Add diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash', color='gray')
        ))
        
        fig.update_layout(
            title=f'ROC Curve - {algorithm_name}',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            showlegend=True
        )
        
        return fig
    
    def create_metrics_comparison_plot(self):
        """
        Create comparison plot of all trained models
        """
        if not self.evaluation_results:
            return None
        
        algorithms = list(self.evaluation_results.keys())
        metrics_names = ['accuracy', 'precision', 'recall', 'f1']
        
        fig = go.Figure()
        
        for metric in metrics_names:
            values = [self.evaluation_results[alg][metric] for alg in algorithms]
            fig.add_trace(go.Bar(
                name=metric.title(),
                x=algorithms,
                y=values,
                text=[f'{v:.3f}' for v in values],
                textposition='auto'
            ))
        
        fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Algorithm',
            yaxis_title='Score',
            barmode='group',
            yaxis=dict(range=[0, 1])
        )
        
        return fig
    
    def get_feature_importance(self, algorithm_name):
        """
        Get feature importance for tree-based models
        """
        if algorithm_name not in self.trained_models:
            return None
            
        model = self.trained_models[algorithm_name]['model']
        
        if hasattr(model, 'feature_importances_'):
            feature_names = self.trained_models[algorithm_name]['feature_columns']
            importances = model.feature_importances_
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            return importance_df
        
        return None
    
    def create_feature_importance_plot(self, algorithm_name):
        """
        Create feature importance visualization
        """
        importance_df = self.get_feature_importance(algorithm_name)
        
        if importance_df is None:
            return None
        
        fig = px.bar(
            importance_df.head(10),  # Top 10 features
            x='Importance',
            y='Feature',
            orientation='h',
            title=f'Feature Importance - {algorithm_name}',
            labels={'Importance': 'Feature Importance', 'Feature': 'Features'}
        )
        
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        
        return fig
