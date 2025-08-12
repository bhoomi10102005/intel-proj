import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.model import load_model, predict
from src.visualizer import SensorDataVisualizer
from src.model_trainer import ModelTrainer
import os
import glob

st.set_page_config(
    page_title="Machine Sensor Analytics",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dynamic CSS for better styling that adapts to theme
st.markdown("""
<style>
    /* Theme-adaptive variables */
    :root {
        --primary-color: #1f77b4;
        --success-color: #28a745;
        --warning-color: #ffc107;
        --danger-color: #dc3545;
        --info-color: #17a2b8;
    }

    /* Dark theme styles */
    @media (prefers-color-scheme: dark) {
        :root {
            --bg-primary: #0e1117;
            --bg-secondary: #262730;
            --bg-tertiary: #1a1d29;
            --text-primary: #fafafa;
            --text-secondary: #c9c9c9;
            --border-color: #404040;
            --shadow-color: rgba(0, 0, 0, 0.3);
            --card-bg: #1e1e1e;
            --hover-bg: #2a2a2a;
        }
        
        .main-header {
            color: #64b5f6 !important;
            text-shadow: 0 0 10px rgba(100, 181, 246, 0.3);
        }
        
        .metric-card {
            background-color: var(--bg-secondary) !important;
            border: 1px solid var(--border-color) !important;
            color: var(--text-primary) !important;
            box-shadow: 0 4px 12px var(--shadow-color) !important;
        }
        
        .prediction-success {
            background-color: rgba(40, 167, 69, 0.2) !important;
            color: #4caf50 !important;
            border-left-color: #4caf50 !important;
            border: 1px solid rgba(76, 175, 80, 0.3) !important;
        }
        
        .prediction-warning {
            background-color: rgba(255, 193, 7, 0.2) !important;
            color: #ffeb3b !important;
            border-left-color: #ffeb3b !important;
            border: 1px solid rgba(255, 235, 59, 0.3) !important;
        }
        
        .welcome-section {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color) !important;
        }
        
        .feature-card {
            border: 1px solid var(--border-color) !important;
            box-shadow: 0 4px 20px var(--shadow-color) !important;
        }
        
        .capability-card-green {
            background: linear-gradient(135deg, rgba(40, 167, 69, 0.2) 0%, rgba(72, 187, 120, 0.2) 100%) !important;
            border-left-color: #48bb78 !important;
            color: var(--text-primary) !important;
        }
        
        .capability-card-blue {
            background: linear-gradient(135deg, rgba(33, 150, 243, 0.2) 0%, rgba(100, 181, 246, 0.2) 100%) !important;
            border-left-color: #64b5f6 !important;
            color: var(--text-primary) !important;
        }
        
        .workflow-card {
            background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%) !important;
            border: 1px solid var(--border-color) !important;
            color: var(--text-primary) !important;
        }
    }

    /* Light theme styles */
    @media (prefers-color-scheme: light) {
        :root {
            --bg-primary: #ffffff;
            --bg-secondary: #f8f9fa;
            --bg-tertiary: #e9ecef;
            --text-primary: #212529;
            --text-secondary: #6c757d;
            --border-color: #dee2e6;
            --shadow-color: rgba(0, 0, 0, 0.1);
            --card-bg: #ffffff;
            --hover-bg: #f8f9fa;
        }
        
        .main-header {
            color: #1f77b4 !important;
            text-shadow: 0 2px 4px rgba(31, 119, 180, 0.1);
        }
        
        .metric-card {
            background-color: #f0f2f6 !important;
            color: var(--text-primary) !important;
            box-shadow: 0 2px 8px var(--shadow-color) !important;
        }
        
        .prediction-success {
            background-color: #d4edda !important;
            color: #155724 !important;
        }
        
        .prediction-warning {
            background-color: #fff3cd !important;
            color: #856404 !important;
        }
        
        .welcome-section {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%) !important;
            color: var(--text-primary) !important;
        }
        
        .capability-card-green {
            background-color: #e8f5e8 !important;
            color: var(--text-primary) !important;
        }
        
        .capability-card-blue {
            background-color: #e3f2fd !important;
            color: var(--text-primary) !important;
        }
    }

    /* Common styles */
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .metric-card {
        padding: 1rem;
        border-radius: 12px;
        border-left: 4px solid var(--primary-color);
        transition: all 0.3s ease;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px var(--shadow-color) !important;
    }
    
    .prediction-success {
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid var(--success-color);
        font-weight: 500;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .prediction-warning {
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid var(--warning-color);
        font-weight: 500;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .welcome-section {
        text-align: center;
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .feature-card {
        text-align: center;
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .feature-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2) !important;
    }
    
    .capability-card-green, .capability-card-blue {
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid;
        transition: all 0.3s ease;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .capability-card-green:hover, .capability-card-blue:hover {
        transform: translateX(5px);
    }
    
    .workflow-card {
        text-align: center;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .workflow-card:hover {
        transform: translateY(-3px);
    }
    
    /* Streamlit specific overrides */
    .stSelectbox > div > div {
        border-radius: 8px;
    }
    
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }

    /* Auto-detect and apply theme based on Streamlit's theme */
    [data-theme="dark"] .main-header {
        color: #64b5f6 !important;
        text-shadow: 0 0 10px rgba(100, 181, 246, 0.3);
    }
    
    [data-theme="light"] .main-header {
        color: #1f77b4 !important;
        text-shadow: 0 2px 4px rgba(31, 119, 180, 0.1);
    }
    
    /* Additional theme-adaptive classes */
    .info-section {
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        transition: all 0.3s ease;
    }
    
    .warning-section {
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border-left: 4px solid var(--warning-color);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        transition: all 0.3s ease;
    }
    
    .success-section {
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border-left: 4px solid var(--success-color);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        transition: all 0.3s ease;
    }
    
    .dataset-info-card {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Dark theme for additional classes */
    @media (prefers-color-scheme: dark) {
        .info-section {
            background: linear-gradient(135deg, rgba(33, 150, 243, 0.15) 0%, rgba(100, 181, 246, 0.15) 100%) !important;
            color: var(--text-primary) !important;
            border: 1px solid rgba(100, 181, 246, 0.3) !important;
        }
        
        .warning-section {
            background: linear-gradient(135deg, rgba(255, 193, 7, 0.15) 0%, rgba(255, 235, 59, 0.15) 100%) !important;
            color: var(--text-primary) !important;
            border: 1px solid rgba(255, 235, 59, 0.3) !important;
        }
        
        .success-section {
            background: linear-gradient(135deg, rgba(40, 167, 69, 0.15) 0%, rgba(72, 187, 120, 0.15) 100%) !important;
            color: var(--text-primary) !important;
            border: 1px solid rgba(72, 187, 120, 0.3) !important;
        }
        
        .dataset-info-card {
            background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color) !important;
        }
    }
    
    /* Light theme for additional classes */
    @media (prefers-color-scheme: light) {
        .info-section {
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%) !important;
            color: var(--text-primary) !important;
        }
        
        .warning-section {
            background: linear-gradient(135deg, #fff3cd 0%, #ffe082 100%) !important;
            color: var(--text-primary) !important;
        }
        
        .success-section {
            background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c8 100%) !important;
            color: var(--text-primary) !important;
        }
        
        .dataset-info-card {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%) !important;
            color: var(--text-primary) !important;
        }
    }
    
    /* Hover effects */
    .info-section:hover, .warning-section:hover, .success-section:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px var(--shadow-color) !important;
    }
    
    .dataset-info-card:hover {
        transform: translateX(3px);
        box-shadow: 0 4px 15px var(--shadow-color) !important;
    }
    
    /* Ensure text elements have proper contrast */
    .stMarkdown p, .stMarkdown li, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, 
    .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: var(--text-primary) !important;
    }
    
    /* Dark theme text fixes */
    @media (prefers-color-scheme: dark) {
        .stMarkdown, .stText, .stSelectbox label, .stSlider label, 
        .stNumberInput label, .stMultiSelect label, .stCheckbox label {
            color: var(--text-primary) !important;
        }
        
        .stDataFrame {
            color: var(--text-primary) !important;
        }
        
        /* Fix for expander content */
        .streamlit-expanderContent {
            background-color: var(--bg-secondary) !important;
            color: var(--text-primary) !important;
        }
        
        /* Fix for info boxes */
        .stInfo {
            background-color: rgba(23, 162, 184, 0.2) !important;
            color: var(--text-primary) !important;
            border: 1px solid rgba(23, 162, 184, 0.3) !important;
        }
        
        /* Fix for error boxes */
        .stError {
            background-color: rgba(220, 53, 69, 0.2) !important;
            color: #ff6b6b !important;
            border: 1px solid rgba(220, 53, 69, 0.3) !important;
        }
        
        /* Fix for success boxes */
        .stSuccess {
            background-color: rgba(40, 167, 69, 0.2) !important;
            color: #4caf50 !important;
            border: 1px solid rgba(40, 167, 69, 0.3) !important;
        }
        
        /* Fix for warning boxes */
        .stWarning {
            background-color: rgba(255, 193, 7, 0.2) !important;
            color: #ffeb3b !important;
            border: 1px solid rgba(255, 193, 7, 0.3) !important;
        }
    }
    
    /* Light theme text fixes */
    @media (prefers-color-scheme: light) {
        .stMarkdown, .stText, .stSelectbox label, .stSlider label, 
        .stNumberInput label, .stMultiSelect label, .stCheckbox label {
            color: var(--text-primary) !important;
        }
        
        .stDataFrame {
            color: var(--text-primary) !important;
        }
    }
</style>

<script>
// Enhanced theme detection script
(function() {
    function detectAndApplyTheme() {
        // Multiple ways to detect dark mode
        const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
        
        // Check Streamlit's background color
        const bodyBg = getComputedStyle(document.body).backgroundColor;
        const containerBg = getComputedStyle(document.documentElement).getPropertyValue('--background-color');
        
        // Streamlit dark theme detection
        const isStreamlitDark = bodyBg.includes('rgb(14, 17, 23)') || 
                                bodyBg.includes('#0e1117') || 
                                containerBg.includes('rgb(14, 17, 23)') ||
                                containerBg.includes('#0e1117');
        
        // Check for dark theme indicators in the DOM
        const hasStreamlitDarkClass = document.querySelector('[data-testid="stAppViewContainer"]')?.style.backgroundColor.includes('rgb(14, 17, 23)');
        
        const isDark = prefersDark || isStreamlitDark || hasStreamlitDarkClass;
        
        // Apply theme
        document.documentElement.setAttribute('data-theme', isDark ? 'dark' : 'light');
        
        // Force update CSS variables
        if (isDark) {
            document.documentElement.style.setProperty('--text-primary', '#fafafa');
            document.documentElement.style.setProperty('--bg-secondary', '#262730');
            document.documentElement.style.setProperty('--border-color', '#404040');
        } else {
            document.documentElement.style.setProperty('--text-primary', '#212529');
            document.documentElement.style.setProperty('--bg-secondary', '#f8f9fa');
            document.documentElement.style.setProperty('--border-color', '#dee2e6');
        }
        
        console.log('Theme detected:', isDark ? 'dark' : 'light');
    }
    
    // Initial detection
    detectAndApplyTheme();
    
    // Listen for theme changes
    if (window.matchMedia) {
        window.matchMedia('(prefers-color-scheme: dark)').addListener(detectAndApplyTheme);
    }
    
    // Monitor DOM changes for Streamlit's dynamic updates
    const observer = new MutationObserver(function(mutations) {
        let shouldRecheck = false;
        mutations.forEach(function(mutation) {
            if (mutation.type === 'attributes' && 
                (mutation.attributeName === 'style' || mutation.attributeName === 'class')) {
                shouldRecheck = true;
            }
        });
        if (shouldRecheck) {
            setTimeout(detectAndApplyTheme, 100);
        }
    });
    
    // Start observing
    observer.observe(document.body, { 
        attributes: true, 
        childList: true, 
        subtree: true,
        attributeFilter: ['style', 'class']
    });
    
    // Also check on window focus (when user switches themes)
    window.addEventListener('focus', detectAndApplyTheme);
    
    // Periodic check as fallback
    setInterval(detectAndApplyTheme, 5000);
})();
</script>
""", unsafe_allow_html=True)

# Load available data files
@st.cache_data
def load_available_datasets():
    """Load available experiment and training datasets"""
    experiment_files = glob.glob("data/experiment_*.csv")
    train_file = "data/train.csv"
    return experiment_files, train_file

@st.cache_data
def load_dataset(file_path):
    """Load and return dataset"""
    return pd.read_csv(file_path)

# Step-by-step Machine Learning Pipeline
st.markdown('<h1 class="main-header">🛠️ Machine Sensor Analytics Pipeline</h1>', unsafe_allow_html=True)

# Welcome section
st.markdown("""
<div class="welcome-section">
    <h3>Welcome to Advanced Machine Learning Analytics</h3>
    <p>Follow the step-by-step process to analyze sensor data and predict tool wear status using machine learning.</p>
</div>
""", unsafe_allow_html=True)

# Step 1: Model Evaluation Dashboard
st.markdown("---")
st.markdown('<h2 style="color: #1f77b4;">� Step 1: Model Evaluation Dashboard</h2>', unsafe_allow_html=True)
st.markdown("""
<div class="info-section">
    <h4>🎯 Model Performance Evaluation</h4>
    <p>Evaluate model performance with comprehensive metrics: accuracy, confusion matrix, precision, recall, F1 score, and ROC curve analysis.</p>
</div>
""", unsafe_allow_html=True)

# Select test data file from available files (only .csv in data/)
import glob
csv_files = sorted([f for f in glob.glob("data/*.csv") if os.path.basename(f) == "train.csv" or f.startswith("data/experiment_")])
file_options = [os.path.basename(f) for f in csv_files]
st.markdown("### 📂 Select Test Data File")
selected_file = st.selectbox("Select a test CSV file for evaluation", file_options, help="Only files like train.csv or experiment_XX.csv are shown.")

if selected_file:
    test_file_path = f"data/{selected_file}" if not selected_file.startswith("data/") else selected_file
    if os.path.exists(test_file_path):
        test_df = pd.read_csv(test_file_path)
        st.success(f"✅ Loaded {selected_file} with {len(test_df)} samples and {len(test_df.columns)} features.")
        st.markdown("### 👁️ Test Data Preview")
        st.dataframe(test_df.head(10), use_container_width=True)
        
        # Continue with evaluation logic here (keeping existing evaluation code)
        # Auto-detect label column for evaluation
        def detect_label_column_eval(df):
            """Intelligently detect the most likely label column for evaluation"""
            likely_label_names = [
                'tool_condition', 'label', 'target', 'class', 'condition', 
                'status', 'category', 'outcome', 'result', 'y', 'wear',
                'machining_process', 'process', 'state', 'phase', 'actual',
                'true', 'ground_truth', 'gt'
            ]
            
            # First check for exact matches (case insensitive)
            for col in df.columns:
                if col.lower() in likely_label_names:
                    return col
            
            # Check for partial matches (case insensitive)
            for col in df.columns:
                for label_name in likely_label_names:
                    if label_name in col.lower():
                        return col
            
            # Prioritize categorical/string columns with reasonable unique values
            categorical_candidates = []
            for col in df.columns:
                if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                    unique_count = len(df[col].unique())
                    if 2 <= unique_count <= 15:  # Reasonable for classification
                        categorical_candidates.append((col, unique_count, 'categorical'))
            
            # If we found categorical candidates, prefer the one with fewer unique values
            if categorical_candidates:
                categorical_candidates.sort(key=lambda x: x[1])
                return categorical_candidates[0][0]
            
            # Look for integer columns that might be encoded labels
            integer_candidates = []
            for col in df.columns:
                if df[col].dtype in ['int64', 'int32', 'int8', 'int16']:
                    unique_values = df[col].unique()
                    unique_count = len(unique_values)
                    if 2 <= unique_count <= 10 and max(unique_values) - min(unique_values) < 100:
                        if not (unique_count > 50 and max(unique_values) > 1000):
                            integer_candidates.append((col, unique_count, 'integer'))
            
            if integer_candidates:
                integer_candidates.sort(key=lambda x: x[1])
                return integer_candidates[0][0]
            
            # Avoid obvious sensor/measurement columns
            avoid_patterns = [
                'velocity', 'acceleration', 'position', 'current', 'voltage', 
                'power', 'feedback', 'actual', 'command', 'output', 'sequence_number',
                'feedrate', 'pressure', 'number', 'id', 'index'
            ]
            
            # Find the best remaining candidate
            best_candidate = None
            min_unique = float('inf')
            
            for col in df.columns:
                unique_count = len(df[col].unique())
                is_avoid = any(pattern in col.lower() for pattern in avoid_patterns)
                is_continuous = (df[col].dtype == 'float64' and unique_count > 100)
                
                if not is_avoid and not is_continuous and 2 <= unique_count < min_unique:
                    min_unique = unique_count
                    best_candidate = col
            
            return best_candidate if best_candidate else df.columns[-1]
        
        # Auto-select feature columns for evaluation
        def get_suggested_eval_features(df, label_col):
            """Get suggested feature columns for evaluation"""
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Remove the label column if it's numeric
            if label_col in numeric_cols:
                numeric_cols.remove(label_col)
            
            # Remove obviously non-feature columns
            exclude_patterns = [
                'id', 'index', 'no', 'number', 'sequence', 'program_number',
                'timestamp', 'time', 'date', 'row', 'record'
            ]
            
            suggested_features = []
            for col in numeric_cols:
                is_exclude = any(pattern in col.lower() for pattern in exclude_patterns)
                
                if not is_exclude:
                    # Check if it's likely an ID column
                    if df[col].dtype in ['int64', 'int32'] and len(df[col].unique()) > len(df) * 0.8:
                        is_exclude = True
                    elif len(df[col].unique()) == 1:
                        is_exclude = True
                
                if not is_exclude:
                    suggested_features.append(col)
            
            # Limit to reasonable number of features
            if len(suggested_features) > 20:
                suggested_features = suggested_features[:20]
            
            return suggested_features
        
        # Detect label and features
        detected_label = detect_label_column_eval(test_df)
        suggested_features = get_suggested_eval_features(test_df, detected_label)
        
        # Feature and label selection
        st.markdown("### 🎯 Configure Evaluation")
        
        # Show auto-detection info
        unique_values = test_df[detected_label].unique()
        unique_count = len(unique_values)
        
        # Validate the detected label
        is_good_label = True
        warning_message = ""
        
        if test_df[detected_label].dtype in ['float64', 'float32'] and unique_count > 20:
            is_good_label = False
            warning_message = f"⚠️ **Warning**: '{detected_label}' appears to be continuous data with {unique_count} unique values."
        
        avoid_patterns = ['velocity', 'acceleration', 'position', 'current', 'voltage', 'power']
        if any(pattern in detected_label.lower() for pattern in avoid_patterns):
            is_good_label = False
            warning_message = f"⚠️ **Warning**: '{detected_label}' appears to be sensor data, not a label column."
        
        if is_good_label:
            st.info(f"🤖 **Auto-detected label column**: `{detected_label}`\n\n"
                    f"📊 **Unique values**: {list(unique_values)[:10]}"
                    f"{'...' if len(unique_values) > 10 else ''}\n\n"
                    f"✅ **Validation**: Good label column for evaluation!")
        else:
            st.warning(f"🤖 **Auto-detected label column**: `{detected_label}`\n\n"
                      f"📊 **Unique values**: {list(unique_values)[:10]}"
                      f"{'...' if len(unique_values) > 10 else ''}\n\n"
                      f"{warning_message}")
            
            # Suggest better alternatives
            better_options = []
            for col in test_df.columns:
                if col != detected_label:
                    col_unique_count = len(test_df[col].unique())
                    if test_df[col].dtype == 'object' and 2 <= col_unique_count <= 15:
                        better_options.append(f"`{col}` ({col_unique_count} values)")
                    elif test_df[col].dtype in ['int64', 'int32'] and 2 <= col_unique_count <= 10:
                        better_options.append(f"`{col}` ({col_unique_count} values)")
            
            if better_options:
                st.info(f"💡 **Better alternatives**: {', '.join(better_options[:3])}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Select feature columns
            numeric_cols = test_df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = st.multiselect(
                "Select feature columns:", 
                numeric_cols,
                default=suggested_features,  # Auto-select suggested features
                key="eval_features",
                help="Choose the same features used during training. Suggested features are pre-selected."
            )
            
            if suggested_features:
                st.success(f"✅ Auto-selected {len(suggested_features)} feature columns")
        
        with col2:
            # Select label column with auto-detection
            label_options = list(test_df.columns)
            default_label_index = label_options.index(detected_label) if detected_label in label_options else 0
            
            label_col = st.selectbox(
                "Label column (ground truth):", 
                test_df.columns,
                index=default_label_index,
                key="eval_label",
                help="Choose the column with true labels for comparison. Auto-detected based on data patterns."
            )
            
            # Show label column info
            if label_col:
                unique_count = len(test_df[label_col].unique())
                st.info(f"📋 **{label_col}** has {unique_count} unique values")
                if unique_count <= 10:
                    st.write("**Values:**", list(test_df[label_col].unique()))
        
        if feature_cols and label_col:
            # Add validation for evaluation configuration
            eval_errors = []
            eval_warnings = []
            
            # Check for duplicate column selection
            if label_col in feature_cols:
                eval_errors.append("❌ **Configuration Error**: The label column cannot be used as a feature!")
                eval_errors.append("💡 **Solution**: Remove the label column from feature selection.")
            
            # Check if label column is suitable for evaluation
            label_unique_count = len(test_df[label_col].unique())
            if test_df[label_col].dtype in ['float64', 'float32'] and label_unique_count > 20:
                eval_errors.append(f"❌ **Inappropriate Label**: '{label_col}' has {label_unique_count} unique values.")
                eval_errors.append("🔍 **Issue**: Evaluation expects discrete classes, not continuous values.")
            
            # Check for too few features
            if len(feature_cols) < 2:
                eval_warnings.append("⚠️ **Warning**: Using fewer than 2 features may not match training data.")
            
            # Display errors and warnings
            if eval_errors:
                for error in eval_errors:
                    st.error(error)
            
            if eval_warnings:
                for warning in eval_warnings:
                    st.warning(warning)
            
            # Show evaluation configuration summary
            if not eval_errors:
                st.markdown("### 📋 Evaluation Configuration Summary")
                summary_col1, summary_col2 = st.columns(2)
                
                with summary_col1:
                    st.markdown("**🔧 Feature Columns:**")
                    for i, col in enumerate(feature_cols, 1):
                        st.write(f"{i}. `{col}` ({test_df[col].dtype})")
                
                with summary_col2:
                    st.markdown(f"**🎯 Label Column:** `{label_col}` ({test_df[label_col].dtype})")
                    st.write(f"**Classes:** {len(test_df[label_col].unique())}")
                    if len(test_df[label_col].unique()) <= 10:
                        for val in test_df[label_col].unique():
                            count = sum(test_df[label_col] == val)
                            percentage = (count / len(test_df)) * 100
                            st.write(f"  • `{val}`: {count} samples ({percentage:.1f}%)")

            if st.button("📊 Evaluate Model", type="primary", disabled=bool(eval_errors)):
                try:
                    # Load model and make predictions
                    with st.spinner("Loading model and making predictions..."):
                        model = load_model()
                        X = test_df[feature_cols]
                        y_true = test_df[label_col]
                        
                        # Handle different label formats
                        if y_true.dtype == 'object':
                            # Convert string labels to numeric
                            unique_labels = y_true.unique()
                            if len(unique_labels) == 2:
                                # Binary classification - auto-detect mapping
                                sorted_labels = sorted(unique_labels)
                                label_mapping = {sorted_labels[0]: 0, sorted_labels[1]: 1}
                                st.info(f"🔄 **Label Mapping**: {label_mapping}")
                            else:
                                # Multi-class classification
                                label_mapping = {label: idx for idx, label in enumerate(sorted(unique_labels))}
                                st.info(f"🔄 **Label Mapping**: {label_mapping}")
                            
                            y_true_numeric = y_true.map(label_mapping)
                            if y_true_numeric.isnull().any():
                                st.error("❌ Could not map all labels to numeric values. Please check your data.")
                                st.stop()
                        else:
                            y_true_numeric = y_true
                        
                        # Make predictions
                        y_pred = model.predict(X)
                        y_pred_proba = None
                        if hasattr(model, 'predict_proba'):
                            y_pred_proba = model.predict_proba(X)
                    
                    # Calculate metrics
                    from sklearn.metrics import (
                        accuracy_score, confusion_matrix, precision_score, 
                        recall_score, f1_score, roc_curve, auc, classification_report
                    )
                    
                    acc = accuracy_score(y_true_numeric, y_pred)
                    cm = confusion_matrix(y_true_numeric, y_pred)
                    prec = precision_score(y_true_numeric, y_pred, average='weighted', zero_division=0)
                    rec = recall_score(y_true_numeric, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_true_numeric, y_pred, average='weighted', zero_division=0)
                    
                    # Display main metrics
                    st.markdown("### 📊 Model Performance Metrics")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("🎯 Accuracy", f"{acc*100:.2f}%")
                    with col2:
                        st.metric("🔍 Precision", f"{prec:.3f}")
                    with col3:
                        st.metric("📈 Recall", f"{rec:.3f}")
                    with col4:
                        st.metric("⚖️ F1 Score", f"{f1:.3f}")
                    
                    # Confusion Matrix and Classification Report
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 🔄 Confusion Matrix")
                        
                        # Create label names for confusion matrix
                        if y_true.dtype == 'object':
                            class_names = sorted(y_true.unique())
                        else:
                            class_names = [f"Class {i}" for i in sorted(y_true_numeric.unique())]
                        
                        cm_df = pd.DataFrame(
                            cm,
                            index=[f'Actual {name}' for name in class_names],
                            columns=[f'Predicted {name}' for name in class_names]
                        )
                        st.dataframe(cm_df, use_container_width=True)
                        
                        # Confusion matrix heatmap
                        fig_cm = px.imshow(
                            cm,
                            text_auto=True,
                            aspect="auto",
                            title="Confusion Matrix Heatmap",
                            labels=dict(x="Predicted", y="Actual"),
                            x=class_names,
                            y=class_names,
                            color_continuous_scale='Blues'
                        )
                        st.plotly_chart(fig_cm, use_container_width=True)
                    
                    with col2:
                        st.markdown("### 📋 Classification Report")
                        
                        # Generate classification report
                        if y_true.dtype == 'object':
                            target_names = sorted(y_true.unique())
                        else:
                            target_names = [f"Class {i}" for i in sorted(y_true_numeric.unique())]
                        
                        report = classification_report(y_true_numeric, y_pred, 
                                                     target_names=target_names, 
                                                     output_dict=True, 
                                                     zero_division=0)
                        
                        # Convert to DataFrame for better display
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df.round(3), use_container_width=True)
                    
                    # ROC Curve (for binary classification)
                    if len(set(y_true_numeric)) == 2 and y_pred_proba is not None:
                        st.markdown("### 📈 ROC Curve Analysis")
                        
                        # Calculate ROC curve
                        fpr, tpr, _ = roc_curve(y_true_numeric, y_pred_proba[:, 1])
                        roc_auc = auc(fpr, tpr)
                        
                        # Plot ROC curve
                        fig_roc = go.Figure()
                        fig_roc.add_trace(go.Scatter(
                            x=fpr, y=tpr,
                            mode='lines',
                            name=f'ROC Curve (AUC = {roc_auc:.3f})',
                            line=dict(color='blue', width=2)
                        ))
                        fig_roc.add_trace(go.Scatter(
                            x=[0, 1], y=[0, 1],
                            mode='lines',
                            name='Random Classifier',
                            line=dict(color='red', dash='dash')
                        ))
                        fig_roc.update_layout(
                            title='ROC Curve',
                            xaxis_title='False Positive Rate',
                            yaxis_title='True Positive Rate',
                            width=600, height=400
                        )
                        st.plotly_chart(fig_roc, use_container_width=True)
                        
                        # ROC AUC metric
                        st.metric("📊 ROC AUC Score", f"{roc_auc:.3f}")
                    
                    # Prediction vs Actual comparison
                    st.markdown("### 🔍 Prediction Analysis")
                    
                    # Create results dataframe
                    results_df = test_df[feature_cols + [label_col]].copy()
                    results_df['Predicted'] = y_pred
                    
                    if y_true.dtype == 'object':
                        # Map predictions back to original labels
                        reverse_mapping = {v: k for k, v in label_mapping.items()}
                        results_df['Predicted_Label'] = [reverse_mapping[pred] for pred in y_pred]
                        results_df['Correct'] = results_df[label_col] == results_df['Predicted_Label']
                    else:
                        results_df['Correct'] = results_df[label_col] == results_df['Predicted']
                    
                    # Show sample predictions
                    st.markdown("#### 📋 Sample Predictions")
                    sample_size = min(20, len(results_df))
                    st.dataframe(results_df.head(sample_size), use_container_width=True)
                    
                    # Export results option
                    st.markdown("### 📥 Export Results")
                    
                    # Prepare export data
                    export_data = {
                        'Model_Accuracy': acc,
                        'Precision': prec,
                        'Recall': rec,
                        'F1_Score': f1,
                        'Features_Used': ', '.join(feature_cols),
                        'Label_Column': label_col,
                        'Test_Samples': len(test_df),
                        'Correct_Predictions': sum(results_df['Correct']),
                        'Incorrect_Predictions': sum(~results_df['Correct'])
                    }
                    
                    if len(set(y_true_numeric)) == 2 and y_pred_proba is not None:
                        export_data['ROC_AUC'] = roc_auc
                    
                    # Export summary
                    export_summary = pd.DataFrame([export_data])
                    csv_summary = export_summary.to_csv(index=False)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="📥 Download Evaluation Summary",
                            data=csv_summary,
                            file_name="model_evaluation_summary.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        # Export detailed predictions
                        csv_predictions = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Detailed Predictions",
                            data=csv_predictions,
                            file_name="detailed_predictions.csv",
                            mime="text/csv"
                        )
                    
                except Exception as e:
                    st.error(f"❌ Error during evaluation: {str(e)}")
                    
                    # Check if the error is due to feature mismatch
                    if "feature names should match" in str(e).lower() or "feature names unseen" in str(e).lower():
                        st.error("🔍 **Feature Mismatch Detected!**")
                        st.markdown("""
                        **The model was trained on different features than what you selected.**
                        
                        **Common Solutions:**
                        1. **Check your training data** - What features were used originally?
                        2. **Match feature names** - Ensure test data has the same column names
                        3. **Use compatible data** - Upload test data with the same structure as training data
                        """)
                        
                        # Show what features the model expects vs what was provided
                        if "Feature names seen at fit time" in str(e):
                            st.markdown("#### 🔍 Feature Analysis")
                            error_lines = str(e).split('\n')
                            for line in error_lines:
                                if 'Feature names seen at fit time' in line or 'Feature names unseen at fit time' in line:
                                    st.write(f"**{line.strip()}**")
                                elif line.strip().startswith('- '):
                                    st.write(line.strip())
                        
                        st.info("""
                        💡 **Recommended Solution**: Upload train.csv for evaluation (has `feedrate`, `clamp_pressure`), 
                        or use experiment data that has compatible features.
                        
                        **Why This Happens**: The default model expects `feedrate` and `clamp_pressure` but your 
                        experiment data has different sensor features like `X1_ActualAcceleration`, etc.
                        """)
                    
                    # Provide debugging help
                    with st.expander("🔧 Debugging Information"):
                        st.write("**Selected Features:**", feature_cols)
                        st.write("**Label Column:**", label_col)
                        st.write("**Feature Data Types:**")
                        st.write(test_df[feature_cols].dtypes)
                        st.write("**Label Data Type:**", test_df[label_col].dtype)
                        st.write("**Sample Data:**")
                        st.write(test_df[feature_cols + [label_col]].head())
                        st.write("**Full Error:**")
                        st.code(str(e))
    else:
        st.error(f"❌ File not found: {test_file_path}")
else:
    st.info("Select a test data file to begin model evaluation.")

# Step 2: Worn Tool Prediction
st.markdown("---")
st.markdown('<h2 style="color: #1f77b4;">🔧 Step 2: Worn Tool Prediction</h2>', unsafe_allow_html=True)
st.markdown("""
<div class="info-section">
    <h4>🎯 Prediction Engine</h4>
    <p>Select from available experiment datasets or training data to predict tool wear status using our trained Random Forest model.</p>
</div>
""", unsafe_allow_html=True)

# Load available datasets
experiment_files, train_file = load_available_datasets()

# Data source selection
st.markdown("### 📂 Select Data Source")

# Create options for data selection
data_options = []
if os.path.exists(train_file):
    data_options.append("Training Data (train.csv)")

for exp_file in sorted(experiment_files):
    filename = os.path.basename(exp_file)
    data_options.append(f"Experiment Data ({filename})")

prediction_selected_option = st.selectbox("Choose dataset for prediction:", data_options, key="prediction_data")

if prediction_selected_option:
    # Determine which file to load
    if "train.csv" in prediction_selected_option:
        prediction_selected_file = train_file
        prediction_data_type = "training"
    else:
        # Extract experiment filename
        exp_filename = prediction_selected_option.split("(")[1].split(")")[0]
        prediction_selected_file = f"data/{exp_filename}"
        prediction_data_type = "experiment"
    
    # Load selected dataset
    if os.path.exists(prediction_selected_file):
        prediction_df = load_dataset(prediction_selected_file)
        
        st.success(f"✅ Loaded {prediction_selected_option} - {len(prediction_df)} samples")
        
        # Show dataset preview
        st.markdown("### 👁️ Data Preview")
        st.dataframe(prediction_df.head(10), use_container_width=True)
        
        # Prediction configuration
        st.markdown("### ⚙️ Prediction Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Select prediction method
            prediction_method = st.selectbox(
                "📊 Select Prediction Method",
                ["Batch Processing", "Single Sample Prediction"],
                help="Choose how you want to make predictions",
                key="pred_method"
            )
            
            # Select model confidence threshold
            confidence_threshold = st.slider(
                "🎯 Confidence Threshold",
                min_value=0.1, max_value=0.9, value=0.7, step=0.05,
                help="Minimum confidence level for tool wear prediction",
                key="conf_threshold"
            )
        
        with col2:
            # Select output format
            output_format = st.selectbox(
                "📄 Output Format",
                ["Detailed Report", "Summary Only", "Raw Predictions"],
                help="Choose how detailed you want the prediction results",
                key="output_fmt"
            )
            
            # Enable prediction explanations
            show_explanations = st.checkbox(
                "🧠 Show AI Explanations",
                value=True,
                help="Include explanations for each prediction",
                key="show_exp"
            )
        
        # Feature selection for prediction
        numeric_columns = prediction_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter out ID-like columns
        feature_candidates = []
        for col in numeric_columns:
            if not any(pattern in col.lower() for pattern in ['id', 'number', 'index', 'sequence']):
                feature_candidates.append(col)
        
        if feature_candidates:
            st.markdown("### 🔧 Feature Selection for Prediction")
            
            # Default feature selection based on data type
            if prediction_data_type == "training" and all(col in feature_candidates for col in ['feedrate', 'clamp_pressure']):
                default_features = ['feedrate', 'clamp_pressure']
            else:
                # For experiment data, suggest the first few numeric columns
                default_features = feature_candidates[:min(3, len(feature_candidates))]
            
            selected_features = st.multiselect(
                "Select features for prediction:",
                feature_candidates,
                default=default_features,
                help="Choose which columns to use for tool wear prediction",
                key="pred_features"
            )
            
            if selected_features:
                # Show feature statistics
                st.markdown("#### 📊 Feature Statistics")
                feature_stats = prediction_df[selected_features].describe()
                st.dataframe(feature_stats, use_container_width=True)
                
                if prediction_method == "Single Sample Prediction":
                    st.markdown("### 📊 Single Sample Input")
                    
                    # Create input fields for each feature
                    input_values = {}
                    cols = st.columns(min(3, len(selected_features)))
                    
                    for i, feature in enumerate(selected_features):
                        with cols[i % 3]:
                            feature_stats = prediction_df[feature].describe()
                            min_val = float(feature_stats['min'])
                            max_val = float(feature_stats['max'])
                            mean_val = float(feature_stats['mean'])
                            
                            input_values[feature] = st.number_input(
                                f"🔧 {feature}",
                                min_value=min_val,
                                max_value=max_val,
                                value=mean_val,
                                step=(max_val - min_val) / 100,
                                help=f"Range: {min_val:.2f} to {max_val:.2f}",
                                key=f"input_{feature}"
                            )
                    
                    # Single prediction button
                    if st.button("🚀 Predict Single Sample", type="primary", key="single_predict"):
                        try:
                            # Create input data
                            input_data = pd.DataFrame([input_values])
                            
                            # Load model and make prediction
                            with st.spinner("🔄 Analyzing input and predicting tool wear..."):
                                model = load_model()
                                prediction = model.predict(input_data)[0]
                                prediction_proba = None
                                
                                if hasattr(model, 'predict_proba'):
                                    prediction_proba = model.predict_proba(input_data)[0]
                            
                            # Display prediction results
                            st.markdown("### 🎯 Single Sample Prediction Results")
                            
                            # Main prediction result
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                if prediction == 1:
                                    st.error("⚠️ **WORN TOOL DETECTED**")
                                    recommendation = "🔄 **Immediate Action Required**: Replace tool before next operation"
                                    risk_level = "HIGH"
                                else:
                                    st.success("✅ **TOOL IN GOOD CONDITION**")
                                    recommendation = "✅ **Continue Operation**: Tool can be used safely"
                                    risk_level = "LOW"
                            
                            with col2:
                                if prediction_proba is not None:
                                    confidence = max(prediction_proba) * 100
                                    st.metric("🎯 Confidence", f"{confidence:.1f}%")
                                    
                                    if confidence < confidence_threshold * 100:
                                        st.warning(f"⚠️ Confidence below threshold ({confidence_threshold*100:.0f}%)")
                                else:
                                    st.info("ℹ️ Confidence score not available")
                            
                            with col3:
                                st.metric("⚡ Risk Level", risk_level)
                            
                            # Show input summary
                            st.markdown("#### 📋 Input Summary")
                            input_summary_df = pd.DataFrame([input_values]).round(3)
                            st.dataframe(input_summary_df, use_container_width=True)
                            
                            st.info(recommendation)
                            
                        except Exception as e:
                            st.error(f"❌ Single prediction failed: {str(e)}")
                            if "feature names should match" in str(e).lower():
                                st.info("💡 Try using features that match the model training data (e.g., 'feedrate', 'clamp_pressure' for train.csv)")
                
                else:  # Batch Processing
                    if st.button("🚀 Run Batch Prediction", type="primary", key="batch_predict"):
                        try:
                            with st.spinner("🔄 Processing batch predictions..."):
                                # Load model
                                model = load_model()
                                
                                # Prepare data
                                X_batch = prediction_df[selected_features]
                                
                                # Make predictions
                                predictions = model.predict(X_batch)
                                probabilities = None
                                
                                if hasattr(model, 'predict_proba'):
                                    probabilities = model.predict_proba(X_batch)
                            
                            # Create results DataFrame
                            results_df = prediction_df.copy()
                            results_df['Predicted_Tool_Wear'] = predictions
                            results_df['Wear_Status'] = ['WORN' if pred == 1 else 'GOOD' for pred in predictions]
                            
                            if probabilities is not None:
                                results_df['Confidence'] = [max(prob) for prob in probabilities]
                                results_df['Confidence_Percentage'] = results_df['Confidence'] * 100
                            
                            # Display batch results
                            st.markdown("### 📊 Batch Prediction Results")
                            
                            # Summary statistics
                            total_samples = len(results_df)
                            worn_tools = sum(results_df['Predicted_Tool_Wear'])
                            good_tools = total_samples - worn_tools
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("📊 Total Samples", total_samples)
                            with col2:
                                st.metric("⚠️ Worn Tools", worn_tools)
                            with col3:
                                st.metric("✅ Good Tools", good_tools)
                            with col4:
                                wear_rate = (worn_tools / total_samples) * 100 if total_samples > 0 else 0
                                st.metric("📈 Wear Rate", f"{wear_rate:.1f}%")
                            
                            # Results visualization
                            if output_format in ["Detailed Report", "Summary Only"]:
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # Pie chart of tool status
                                    if worn_tools > 0 or good_tools > 0:
                                        fig_pie = px.pie(
                                            values=[worn_tools, good_tools],
                                            names=['Worn Tools', 'Good Tools'],
                                            title='Tool Condition Distribution',
                                            color_discrete_map={'Worn Tools': '#ff6b6b', 'Good Tools': '#51cf66'}
                                        )
                                        st.plotly_chart(fig_pie, use_container_width=True)
                                
                                with col2:
                                    # Confidence distribution
                                    if 'Confidence_Percentage' in results_df.columns:
                                        fig_hist = px.histogram(
                                            results_df,
                                            x='Confidence_Percentage',
                                            title='Prediction Confidence Distribution',
                                            nbins=20,
                                            color_discrete_sequence=['#4ecdc4']
                                        )
                                        fig_hist.update_layout(
                                            xaxis_title='Confidence (%)',
                                            yaxis_title='Number of Predictions'
                                        )
                                        st.plotly_chart(fig_hist, use_container_width=True)
                            
                            # Detailed results table
                            if output_format == "Detailed Report":
                                st.markdown("#### 📋 Detailed Results")
                                
                                # Filter options
                                col1, col2 = st.columns(2)
                                with col1:
                                    filter_status = st.selectbox(
                                        "🔍 Filter by Status:",
                                        ["All", "WORN Only", "GOOD Only"],
                                        key="filter_status"
                                    )
                                
                                with col2:
                                    if 'Confidence_Percentage' in results_df.columns:
                                        min_confidence = st.slider(
                                            "🎯 Minimum Confidence:",
                                            0, 100, 0,
                                            help="Show only predictions above this confidence level",
                                            key="min_conf_filter"
                                        )
                                
                                # Apply filters
                                filtered_df = results_df.copy()
                                
                                if filter_status == "WORN Only":
                                    filtered_df = filtered_df[filtered_df['Wear_Status'] == 'WORN']
                                elif filter_status == "GOOD Only":
                                    filtered_df = filtered_df[filtered_df['Wear_Status'] == 'GOOD']
                                
                                if 'Confidence_Percentage' in filtered_df.columns:
                                    filtered_df = filtered_df[filtered_df['Confidence_Percentage'] >= min_confidence]
                                
                                st.dataframe(filtered_df, use_container_width=True)
                            
                            elif output_format == "Raw Predictions":
                                st.markdown("#### 🔢 Raw Prediction Values")
                                raw_results = pd.DataFrame({
                                    'Sample_Index': range(len(predictions)),
                                    'Prediction': predictions,
                                    'Confidence': [max(prob) for prob in probabilities] if probabilities is not None else ['N/A'] * len(predictions)
                                })
                                st.dataframe(raw_results, use_container_width=True)
                            
                            # Export options
                            st.markdown("### 📥 Export Results")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Download full results
                                csv_results = results_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Full Results",
                                    data=csv_results,
                                    file_name=f"batch_predictions_{os.path.basename(prediction_selected_file)}",
                                    mime="text/csv",
                                    key="download_full"
                                )
                            
                            with col2:
                                # Download summary report
                                summary_data = {
                                    'File_Processed': [os.path.basename(prediction_selected_file)],
                                    'Total_Samples': [total_samples],
                                    'Worn_Tools': [worn_tools],
                                    'Good_Tools': [good_tools],
                                    'Wear_Rate_Percentage': [wear_rate],
                                    'Features_Used': [', '.join(selected_features)],
                                    'Processing_Date': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')]
                                }
                                
                                summary_df = pd.DataFrame(summary_data)
                                csv_summary = summary_df.to_csv(index=False)
                                
                                st.download_button(
                                    label="📥 Download Summary Report",
                                    data=csv_summary,
                                    file_name=f"prediction_summary_{os.path.basename(prediction_selected_file)}",
                                    mime="text/csv",
                                    key="download_summary"
                                )
                            
                            # Show explanations if requested
                            if show_explanations and output_format == "Detailed Report":
                                st.markdown("### 🧠 AI Prediction Explanations")
                                
                                explanation_text = f"""
                                **Prediction Analysis:**
                                
                                The AI model processed {total_samples} samples from {os.path.basename(prediction_selected_file)} and made predictions based on:
                                
                                1. **Features Used**: {', '.join(selected_features)}
                                2. **Model Type**: Random Forest Classifier (ensemble method)
                                3. **Decision Logic**: The model considers feature interactions and patterns learned from training data
                                
                                **Results Summary:**
                                - **{worn_tools}** tools predicted as WORN ({wear_rate:.1f}% of total)
                                - **{good_tools}** tools predicted as GOOD ({100-wear_rate:.1f}% of total)
                                - **Average Confidence**: {results_df['Confidence_Percentage'].mean():.1f}% if 'Confidence_Percentage' in results_df.columns else "N/A"
                                
                                **Interpretation Guidelines:**
                                - High confidence (>80%): Very reliable prediction
                                - Medium confidence (60-80%): Generally reliable, consider additional verification
                                - Low confidence (<60%): Uncertain prediction, manual inspection recommended
                                """
                                
                                st.markdown(explanation_text)
                        
                        except Exception as e:
                            st.error(f"❌ Batch prediction failed: {str(e)}")
                            
                            with st.expander("🔧 Debugging Information"):
                                st.write("**Selected Features:**", selected_features)
                                st.write("**Data Shape:**", prediction_df.shape)
                                st.write("**Data Types:**")
                                st.write(prediction_df[selected_features].dtypes)
                                st.write("**Sample Data:**")
                                st.write(prediction_df[selected_features].head())
                                st.write("**Full Error:**")
                                st.code(str(e))
        else:
            st.warning("⚠️ No suitable numeric features found for prediction. Please check your data.")
    else:
        st.error(f"❌ Could not load {prediction_selected_option}. Please check if the file exists.")

# Step 3: Sensor Data Visualizer
st.markdown("---")
st.markdown('<h2 style="color: #1f77b4;">📈 Step 3: Sensor Data Visualizer</h2>', unsafe_allow_html=True)
st.markdown("""
<div class="success-section">
    <h4>🔍 Interactive Data Exploration</h4>
    <p>Explore sensor data patterns, compare worn vs unworn tools, and identify key insights through interactive visualizations.</p>
</div>
""", unsafe_allow_html=True)

# Dataset selection for visualization
st.markdown("### 📂 Select Dataset for Visualization")

viz_selected_option = st.selectbox("Choose dataset for visualization:", data_options, key="viz_data")

if viz_selected_option:
    # Determine which file to load
    if "train.csv" in viz_selected_option:
        viz_selected_file = train_file
        viz_data_type = "training"
    else:
        # Extract experiment filename
        exp_filename = viz_selected_option.split("(")[1].split(")")[0]
        viz_selected_file = f"data/{exp_filename}"
        viz_data_type = "experiment"
    
    # Load selected dataset
    if os.path.exists(viz_selected_file):
        viz_df = load_dataset(viz_selected_file)
        
        st.success(f"✅ Loaded {viz_selected_option} - {len(viz_df)} samples, {len(viz_df.columns)} features")
        
        # Show dataset preview
        st.markdown("### 👁️ Data Preview")
        st.dataframe(viz_df.head(10), use_container_width=True)
        
        # Visualization configuration
        st.markdown("### ⚙️ Visualization Configuration")
        
        # Get numeric columns for visualization
        numeric_columns = viz_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter out ID-like columns
        viz_feature_candidates = []
        for col in numeric_columns:
            if not any(pattern in col.lower() for pattern in ['id', 'number', 'index', 'sequence']):
                viz_feature_candidates.append(col)
        
        if viz_feature_candidates:
            col1, col2 = st.columns(2)
            
            with col1:
                # Select visualization type
                viz_type = st.selectbox(
                    "📈 Select Visualization Type",
                    [
                        "Distribution Analysis",
                        "Correlation Heatmap", 
                        "Time Series Analysis",
                        "Feature Comparison",
                        "Statistical Summary",
                        "Pattern Detection"
                    ],
                    help="Choose the type of visualization to generate",
                    key="viz_type"
                )
                
                # Select features for visualization
                if viz_type in ["Feature Comparison", "Pattern Detection"]:
                    max_features = 6
                else:
                    max_features = len(viz_feature_candidates)
                
                selected_viz_features = st.multiselect(
                    "🔧 Select Features to Visualize:",
                    viz_feature_candidates,
                    default=viz_feature_candidates[:min(4, len(viz_feature_candidates))],
                    max_selections=max_features,
                    help="Choose which features to include in the visualization",
                    key="viz_features"
                )
            
            with col2:
                # Additional visualization options
                color_scheme = st.selectbox(
                    "🎨 Color Scheme",
                    ["Default", "Viridis", "Plasma", "Blues", "Reds", "Greens"],
                    help="Choose color scheme for visualizations",
                    key="color_scheme"
                )
                
                # Chart size option
                chart_size = st.selectbox(
                    "📏 Chart Size",
                    ["Medium", "Large", "Small"],
                    help="Select the size of generated charts",
                    key="chart_size"
                )
                
                # Export options
                enable_export = st.checkbox(
                    "📥 Enable Chart Export",
                    value=True,
                    help="Allow downloading charts as images",
                    key="enable_export"
                )
            
            # Check if we have a label column for advanced analysis
            potential_label_cols = []
            for col in viz_df.columns:
                unique_vals = len(viz_df[col].unique())
                if 2 <= unique_vals <= 20:  # Potential categorical label
                    potential_label_cols.append(col)
            
            label_column = None
            if potential_label_cols:
                st.markdown("### 🎯 Advanced Analysis Options")
                label_column = st.selectbox(
                    "🏷️ Select Label Column (Optional):",
                    ["None"] + potential_label_cols,
                    help="Choose a categorical column for group-based analysis",
                    key="label_col_viz"
                )
                if label_column == "None":
                    label_column = None
            
            if st.button("📊 Generate Visualizations", type="primary", key="generate_viz"):
                if selected_viz_features:
                    try:
                        # Set color scheme
                        color_map = {
                            "Default": px.colors.qualitative.Set1,
                            "Viridis": px.colors.sequential.Viridis,
                            "Plasma": px.colors.sequential.Plasma,
                            "Blues": px.colors.sequential.Blues,
                            "Reds": px.colors.sequential.Reds,
                            "Greens": px.colors.sequential.Greens
                        }
                        
                        # Set chart dimensions
                        size_config = {
                            "Small": {"width": 400, "height": 300},
                            "Medium": {"width": 600, "height": 400},
                            "Large": {"width": 800, "height": 500}
                        }
                        chart_config = size_config[chart_size]
                        
                        with st.spinner("🔄 Generating visualizations..."):
                            st.markdown(f"### 📊 {viz_type} Results")
                            
                            if viz_type == "Distribution Analysis":
                                # Create distribution plots for each selected feature
                                for i, feature in enumerate(selected_viz_features):
                                    st.markdown(f"#### 📈 Distribution of {feature}")
                                    
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        # Histogram
                                        fig_hist = px.histogram(
                                            viz_df, 
                                            x=feature,
                                            nbins=30,
                                            title=f"Histogram: {feature}",
                                            color_discrete_sequence=color_map[color_scheme]
                                        )
                                        fig_hist.update_layout(**chart_config)
                                        st.plotly_chart(fig_hist, use_container_width=True)
                                    
                                    with col2:
                                        # Box plot
                                        if label_column and label_column in viz_df.columns:
                                            fig_box = px.box(
                                                viz_df,
                                                y=feature,
                                                color=label_column,
                                                title=f"Box Plot: {feature} by {label_column}",
                                                color_discrete_sequence=color_map[color_scheme]
                                            )
                                        else:
                                            fig_box = px.box(
                                                viz_df,
                                                y=feature,
                                                title=f"Box Plot: {feature}"
                                            )
                                        fig_box.update_layout(**chart_config)
                                        st.plotly_chart(fig_box, use_container_width=True)
                                    
                                    # Statistical summary for this feature
                                    st.markdown(f"**📊 {feature} Statistics:**")
                                    stats = viz_df[feature].describe()
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("Mean", f"{stats['mean']:.3f}")
                                    with col2:
                                        st.metric("Std Dev", f"{stats['std']:.3f}")
                                    with col3:
                                        st.metric("Min", f"{stats['min']:.3f}")
                                    with col4:
                                        st.metric("Max", f"{stats['max']:.3f}")
                            
                            elif viz_type == "Correlation Heatmap":
                                # Calculate correlation matrix
                                corr_matrix = viz_df[selected_viz_features].corr()
                                
                                # Create heatmap
                                fig_corr = px.imshow(
                                    corr_matrix,
                                    text_auto=True,
                                    aspect="auto",
                                    title="Feature Correlation Heatmap",
                                    color_continuous_scale=color_scheme.lower() if color_scheme != "Default" else "RdBu_r"
                                )
                                fig_corr.update_layout(**chart_config)
                                st.plotly_chart(fig_corr, use_container_width=True)
                                
                                # Show correlation insights
                                st.markdown("#### 🔍 Correlation Insights")
                                
                                # Find high correlations
                                high_corr_pairs = []
                                for i in range(len(corr_matrix.columns)):
                                    for j in range(i+1, len(corr_matrix.columns)):
                                        corr_val = corr_matrix.iloc[i, j]
                                        if abs(corr_val) > 0.7:
                                            high_corr_pairs.append((
                                                corr_matrix.columns[i],
                                                corr_matrix.columns[j],
                                                corr_val
                                            ))
                                
                                if high_corr_pairs:
                                    st.markdown("**🔗 Strong Correlations (|r| > 0.7):**")
                                    for feat1, feat2, corr_val in high_corr_pairs:
                                        corr_type = "Positive" if corr_val > 0 else "Negative"
                                        st.write(f"• **{feat1}** ↔ **{feat2}**: {corr_val:.3f} ({corr_type})")
                                else:
                                    st.info("No strong correlations (|r| > 0.7) found between selected features.")
                            
                            elif viz_type == "Time Series Analysis":
                                # Check if there's a time-like column or create index-based series
                                time_col = None
                                for col in viz_df.columns:
                                    if 'time' in col.lower() or 'date' in col.lower():
                                        time_col = col
                                        break
                                
                                if time_col is None:
                                    # Use index as time proxy
                                    viz_df_ts = viz_df.copy()
                                    viz_df_ts['Sample_Index'] = range(len(viz_df_ts))
                                    time_col = 'Sample_Index'
                                    st.info("📍 No time column detected. Using sample index as time proxy.")
                                else:
                                    viz_df_ts = viz_df.copy()
                                
                                # Create time series plots
                                for feature in selected_viz_features:
                                    st.markdown(f"#### ⏱️ Time Series: {feature}")
                                    
                                    fig_ts = px.line(
                                        viz_df_ts,
                                        x=time_col,
                                        y=feature,
                                        title=f"Time Series: {feature}",
                                        color_discrete_sequence=color_map[color_scheme]
                                    )
                                    
                                    if label_column and label_column in viz_df.columns:
                                        fig_ts = px.line(
                                            viz_df_ts,
                                            x=time_col,
                                            y=feature,
                                            color=label_column,
                                            title=f"Time Series: {feature} by {label_column}",
                                            color_discrete_sequence=color_map[color_scheme]
                                        )
                                    
                                    fig_ts.update_layout(**chart_config)
                                    st.plotly_chart(fig_ts, use_container_width=True)
                                    
                                    # Trend analysis
                                    if len(viz_df_ts) > 10:
                                        # Simple trend calculation
                                        x_vals = range(len(viz_df_ts))
                                        y_vals = viz_df_ts[feature].values
                                        
                                        # Linear regression for trend
                                        from sklearn.linear_model import LinearRegression
                                        trend_model = LinearRegression()
                                        trend_model.fit(np.array(x_vals).reshape(-1, 1), y_vals)
                                        trend_slope = trend_model.coef_[0]
                                        
                                        trend_direction = "📈 Increasing" if trend_slope > 0 else "📉 Decreasing" if trend_slope < 0 else "➡️ Stable"
                                        st.write(f"**Trend**: {trend_direction} (slope: {trend_slope:.6f})")
                            
                            elif viz_type == "Feature Comparison":
                                # Create scatter plots and pair plots
                                if len(selected_viz_features) >= 2:
                                    # Scatter plot matrix
                                    st.markdown("#### 🔍 Feature Pair Comparisons")
                                    
                                    for i in range(len(selected_viz_features)):
                                        for j in range(i+1, min(i+3, len(selected_viz_features))):  # Limit pairs to avoid too many plots
                                            feat1, feat2 = selected_viz_features[i], selected_viz_features[j]
                                            
                                            st.markdown(f"##### {feat1} vs {feat2}")
                                            
                                            if label_column and label_column in viz_df.columns:
                                                fig_scatter = px.scatter(
                                                    viz_df,
                                                    x=feat1,
                                                    y=feat2,
                                                    color=label_column,
                                                    title=f"{feat1} vs {feat2} (colored by {label_column})",
                                                    color_discrete_sequence=color_map[color_scheme]
                                                )
                                            else:
                                                fig_scatter = px.scatter(
                                                    viz_df,
                                                    x=feat1,
                                                    y=feat2,
                                                    title=f"{feat1} vs {feat2}",
                                                    color_discrete_sequence=color_map[color_scheme]
                                                )
                                            
                                            fig_scatter.update_layout(**chart_config)
                                            st.plotly_chart(fig_scatter, use_container_width=True)
                                else:
                                    st.warning("Please select at least 2 features for comparison analysis.")
                            
                            elif viz_type == "Statistical Summary":
                                # Comprehensive statistical analysis
                                st.markdown("#### 📊 Comprehensive Statistical Summary")
                                
                                # Basic statistics table
                                stats_df = viz_df[selected_viz_features].describe()
                                st.dataframe(stats_df, use_container_width=True)
                                
                                # Additional statistics
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("##### 📈 Additional Metrics")
                                    additional_stats = {}
                                    for feature in selected_viz_features:
                                        additional_stats[feature] = {
                                            'Variance': viz_df[feature].var(),
                                            'Skewness': viz_df[feature].skew(),
                                            'Kurtosis': viz_df[feature].kurtosis(),
                                            'Missing Values': viz_df[feature].isnull().sum()
                                        }
                                    
                                    additional_df = pd.DataFrame(additional_stats).T
                                    st.dataframe(additional_df, use_container_width=True)
                                
                                with col2:
                                    st.markdown("##### 🎯 Data Quality Metrics")
                                    quality_metrics = {}
                                    for feature in selected_viz_features:
                                        quality_metrics[feature] = {
                                            'Completeness': f"{(1 - viz_df[feature].isnull().sum() / len(viz_df)) * 100:.1f}%",
                                            'Unique Values': len(viz_df[feature].unique()),
                                            'Most Frequent': viz_df[feature].mode().iloc[0] if not viz_df[feature].mode().empty else 'N/A',
                                            'Range': f"{viz_df[feature].max() - viz_df[feature].min():.3f}"
                                        }
                                    
                                    quality_df = pd.DataFrame(quality_metrics).T
                                    st.dataframe(quality_df, use_container_width=True)
                                
                                # Correlation analysis
                                if len(selected_viz_features) > 1:
                                    st.markdown("##### 🔗 Quick Correlation Summary")
                                    corr_matrix = viz_df[selected_viz_features].corr()
                                    
                                    # Find strongest correlations
                                    corr_pairs = []
                                    for i in range(len(selected_viz_features)):
                                        for j in range(i+1, len(selected_viz_features)):
                                            corr_val = corr_matrix.iloc[i, j]
                                            corr_pairs.append((
                                                selected_viz_features[i],
                                                selected_viz_features[j],
                                                abs(corr_val),
                                                corr_val
                                            ))
                                    
                                    # Sort by absolute correlation value
                                    corr_pairs.sort(key=lambda x: x[2], reverse=True)
                                    
                                    st.write("**Top 3 Strongest Correlations:**")
                                    for i, (feat1, feat2, abs_corr, corr) in enumerate(corr_pairs[:3]):
                                        st.write(f"{i+1}. **{feat1}** ↔ **{feat2}**: {corr:.3f}")
                            
                            elif viz_type == "Pattern Detection":
                                # Advanced pattern detection
                                st.markdown("#### 🔍 Pattern Detection Analysis")
                                
                                # Outlier detection
                                st.markdown("##### 🎯 Outlier Detection")
                                
                                outlier_results = {}
                                for feature in selected_viz_features:
                                    Q1 = viz_df[feature].quantile(0.25)
                                    Q3 = viz_df[feature].quantile(0.75)
                                    IQR = Q3 - Q1
                                    lower_bound = Q1 - 1.5 * IQR
                                    upper_bound = Q3 + 1.5 * IQR
                                    
                                    outliers = viz_df[(viz_df[feature] < lower_bound) | (viz_df[feature] > upper_bound)]
                                    outlier_results[feature] = {
                                        'count': len(outliers),
                                        'percentage': (len(outliers) / len(viz_df)) * 100,
                                        'lower_bound': lower_bound,
                                        'upper_bound': upper_bound
                                    }
                                
                                # Display outlier results
                                for feature, result in outlier_results.items():
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric(f"Outliers in {feature}", result['count'])
                                    with col2:
                                        st.metric("Percentage", f"{result['percentage']:.1f}%")
                                    with col3:
                                        st.write(f"**Bounds**: [{result['lower_bound']:.3f}, {result['upper_bound']:.3f}]")
                                
                                # Pattern visualization
                                if len(selected_viz_features) >= 2:
                                    st.markdown("##### 📊 Pattern Visualization")
                                    
                                    # Create a combined plot showing patterns
                                    fig_pattern = go.Figure()
                                    
                                    for i, feature in enumerate(selected_viz_features[:4]):  # Limit to 4 features
                                        # Normalize the feature for comparison
                                        normalized_values = (viz_df[feature] - viz_df[feature].min()) / (viz_df[feature].max() - viz_df[feature].min())
                                        
                                        fig_pattern.add_trace(go.Scatter(
                                            x=list(range(len(normalized_values))),
                                            y=normalized_values,
                                            mode='lines',
                                            name=feature,
                                            line=dict(width=2)
                                        ))
                                    
                                    fig_pattern.update_layout(
                                        title="Normalized Feature Patterns Comparison",
                                        xaxis_title="Sample Index",
                                        yaxis_title="Normalized Value (0-1)",
                                        **chart_config
                                    )
                                    st.plotly_chart(fig_pattern, use_container_width=True)
                        
                        # Export functionality
                        if enable_export:
                            st.markdown("### 📥 Export Options")
                            st.info("💡 You can right-click on any chart above and select 'Save as image' to download it.")
                            
                            # Create summary report
                            summary_report = f"""
                            # Visualization Summary Report
                            
                            **Dataset**: {viz_selected_option}
                            **Visualization Type**: {viz_type}
                            **Features Analyzed**: {', '.join(selected_viz_features)}
                            **Total Samples**: {len(viz_df)}
                            **Generated On**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
                            
                            ## Key Insights:
                            - {len(selected_viz_features)} features were analyzed
                            - Dataset contains {len(viz_df)} samples
                            - Color scheme used: {color_scheme}
                            """
                            
                            if viz_type == "Statistical Summary":
                                summary_report += f"\n\n## Statistical Highlights:\n"
                                for feature in selected_viz_features:
                                    mean_val = viz_df[feature].mean()
                                    std_val = viz_df[feature].std()
                                    summary_report += f"- **{feature}**: Mean = {mean_val:.3f}, Std = {std_val:.3f}\n"
                            
                            st.download_button(
                                label="📥 Download Analysis Report",
                                data=summary_report,
                                file_name=f"visualization_report_{viz_type.lower().replace(' ', '_')}.md",
                                mime="text/markdown",
                                key="download_viz_report"
                            )
                        
                        st.success("✅ Visualizations generated successfully!")
                    
                    except Exception as e:
                        st.error(f"❌ Visualization generation failed: {str(e)}")
                        
                        with st.expander("🔧 Debugging Information"):
                            st.write("**Selected Features:**", selected_viz_features)
                            st.write("**Visualization Type:**", viz_type)
                            st.write("**Data Shape:**", viz_df.shape)
                            st.write("**Data Types:**")
                            st.write(viz_df[selected_viz_features].dtypes)
                            st.write("**Full Error:**")
                            st.code(str(e))
                else:
                    st.warning("⚠️ Please select at least one feature for visualization.")
        else:
            st.warning("⚠️ No suitable numeric features found for visualization. Please check your data.")
    else:
        st.error(f"❌ Could not load {viz_selected_option}. Please check if the file exists.")

# Step 4: Data Analysis
st.markdown("---")
st.markdown('<h2 style="color: #1f77b4;">📊 Step 4: Data Analysis</h2>', unsafe_allow_html=True)
st.markdown("""
<div class="warning-section">
    <h4>🔬 Comprehensive Data Analysis</h4>
    <p>Perform in-depth analysis of training data, experiment datasets, and model performance insights.</p>
</div>
""", unsafe_allow_html=True)

# Analysis configuration
st.markdown("### ⚙️ Analysis Configuration")

col1, col2 = st.columns(2)

with col1:
    # Select analysis type
    analysis_type = st.selectbox(
        "🔍 Select Analysis Type",
        [
            "Training Data Analysis",
            "Experiment Data Comparison", 
            "Feature Importance Analysis",
            "Model Performance Analysis",
            "Data Quality Assessment",
            "Comprehensive Report"
        ],
        help="Choose the type of analysis to perform",
        key="analysis_type"
    )

with col2:
    # Analysis depth
    analysis_depth = st.selectbox(
        "📊 Analysis Depth",
        ["Quick Overview", "Standard Analysis", "Deep Dive"],
        help="Choose how detailed the analysis should be",
        key="analysis_depth"
    )

# Load available datasets for analysis
experiment_files, train_file = load_available_datasets()

if analysis_type == "Training Data Analysis":
    st.markdown("### 🔍 Training Data Analysis")
    
    if os.path.exists(train_file):
        analysis_df = load_dataset(train_file)
        
        st.success(f"✅ Loaded training data - {len(analysis_df)} samples, {len(analysis_df.columns)} features")
        
        # Basic statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Data Overview")
            st.write(f"**Total Samples:** {len(analysis_df)}")
            st.write(f"**Features:** {len(analysis_df.columns)}")
            st.write(f"**Memory Usage:** {analysis_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Check for label column
            if 'tool_condition' in analysis_df.columns:
                condition_counts = analysis_df['tool_condition'].value_counts()
                st.write("**Tool Condition Distribution:**")
                for condition, count in condition_counts.items():
                    percentage = count/len(analysis_df)*100
                    st.write(f"- {condition.title()}: {count} ({percentage:.1f}%)")
            
            # Data types summary
            st.markdown("**Data Types:**")
            dtype_counts = analysis_df.dtypes.value_counts()
            for dtype, count in dtype_counts.items():
                st.write(f"- {str(dtype)}: {count} columns")
        
        with col2:
            if 'tool_condition' in analysis_df.columns:
                # Pie chart for tool condition
                condition_counts = analysis_df['tool_condition'].value_counts()
                fig_pie = px.pie(
                    values=condition_counts.values,
                    names=[name.title() for name in condition_counts.index],
                    title="Tool Condition Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                # Show feature distribution if no label
                numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    sample_feature = numeric_cols[0]
                    fig_hist = px.histogram(
                        analysis_df,
                        x=sample_feature,
                        title=f"Distribution of {sample_feature}",
                        nbins=30
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
        
        # Detailed analysis based on depth
        if analysis_depth in ["Standard Analysis", "Deep Dive"]:
            st.markdown("### 📊 Detailed Statistical Analysis")
            
            # Get numeric columns
            numeric_columns = analysis_df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_columns:
                # Statistical summary
                st.markdown("#### 📈 Statistical Summary")
                stats_df = analysis_df[numeric_columns].describe()
                st.dataframe(stats_df, use_container_width=True)
                
                # Missing values analysis
                st.markdown("#### 🔍 Data Quality Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    missing_data = analysis_df.isnull().sum()
                    missing_percentage = (missing_data / len(analysis_df)) * 100
                    
                    if missing_data.sum() > 0:
                        missing_df = pd.DataFrame({
                            'Missing_Count': missing_data,
                            'Missing_Percentage': missing_percentage
                        }).sort_values('Missing_Count', ascending=False)
                        
                        st.write("**Missing Values Summary:**")
                        st.dataframe(missing_df[missing_df['Missing_Count'] > 0], use_container_width=True)
                    else:
                        st.success("✅ No missing values found!")
                
                with col2:
                    # Duplicate analysis
                    duplicate_count = analysis_df.duplicated().sum()
                    duplicate_percentage = (duplicate_count / len(analysis_df)) * 100
                    
                    st.metric("🔄 Duplicate Rows", duplicate_count)
                    st.metric("📊 Duplicate Percentage", f"{duplicate_percentage:.2f}%")
                    
                    # Unique values per column
                    st.write("**Unique Values per Column:**")
                    for col in analysis_df.columns[:5]:  # Show first 5 columns
                        unique_count = len(analysis_df[col].unique())
                        unique_percentage = (unique_count / len(analysis_df)) * 100
                        st.write(f"- **{col}**: {unique_count} ({unique_percentage:.1f}%)")
                
                # Correlation analysis
                if len(numeric_columns) > 1:
                    st.markdown("#### 🔗 Feature Correlation Analysis")
                    
                    corr_matrix = analysis_df[numeric_columns].corr()
                    
                    # Heatmap
                    fig_corr = px.imshow(
                        corr_matrix,
                        text_auto=True,
                        aspect="auto",
                        title="Feature Correlation Matrix",
                        color_continuous_scale="RdBu_r"
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                    
                    # Strong correlations
                    strong_corr = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            corr_val = corr_matrix.iloc[i, j]
                            if abs(corr_val) > 0.7:
                                strong_corr.append((
                                    corr_matrix.columns[i],
                                    corr_matrix.columns[j],
                                    corr_val
                                ))
                    
                    if strong_corr:
                        st.markdown("**🎯 Strong Correlations (|r| > 0.7):**")
                        for feat1, feat2, corr_val in strong_corr:
                            st.write(f"• **{feat1}** ↔ **{feat2}**: {corr_val:.3f}")
                    else:
                        st.info("No strong correlations found.")
        
        if analysis_depth == "Deep Dive":
            st.markdown("### 🔬 Deep Dive Analysis")
            
            # Feature distribution analysis
            if numeric_columns:
                st.markdown("#### 📊 Feature Distribution Analysis")
                
                for feature in numeric_columns[:4]:  # Limit to first 4 features
                    st.markdown(f"##### {feature} Analysis")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # Basic stats
                        feature_stats = analysis_df[feature].describe()
                        st.write("**Statistics:**")
                        st.write(f"Mean: {feature_stats['mean']:.4f}")
                        st.write(f"Std: {feature_stats['std']:.4f}")
                        st.write(f"Skewness: {analysis_df[feature].skew():.4f}")
                        st.write(f"Kurtosis: {analysis_df[feature].kurtosis():.4f}")
                    
                    with col2:
                        # Distribution plot
                        fig_dist = px.histogram(
                            analysis_df,
                            x=feature,
                            nbins=30,
                            title=f"Distribution of {feature}",
                            marginal="box"
                        )
                        st.plotly_chart(fig_dist, use_container_width=True)
                    
                    with col3:
                        # Outlier detection
                        Q1 = analysis_df[feature].quantile(0.25)
                        Q3 = analysis_df[feature].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        outliers = analysis_df[(analysis_df[feature] < lower_bound) | (analysis_df[feature] > upper_bound)]
                        
                        st.write("**Outlier Analysis:**")
                        st.write(f"Outliers: {len(outliers)}")
                        st.write(f"Percentage: {len(outliers)/len(analysis_df)*100:.2f}%")
                        st.write(f"Lower bound: {lower_bound:.4f}")
                        st.write(f"Upper bound: {upper_bound:.4f}")
        
        # Raw data view
        st.markdown("### 📋 Raw Data Preview")
        
        # Data filtering options
        col1, col2 = st.columns(2)
        
        with col1:
            show_rows = st.slider("Number of rows to display", 5, min(100, len(analysis_df)), 20)
        
        with col2:
            if 'tool_condition' in analysis_df.columns:
                filter_condition = st.selectbox(
                    "Filter by tool condition:",
                    ["All"] + list(analysis_df['tool_condition'].unique())
                )
            else:
                filter_condition = "All"
        
        # Apply filters
        if filter_condition != "All":
            filtered_df = analysis_df[analysis_df['tool_condition'] == filter_condition]
        else:
            filtered_df = analysis_df
        
        st.dataframe(filtered_df.head(show_rows), use_container_width=True)
        
        # Export training data analysis
        st.markdown("### 📥 Export Analysis Results")
        
        # Create analysis summary
        analysis_summary = {
            'Dataset': 'Training Data (train.csv)',
            'Total_Samples': len(analysis_df),
            'Total_Features': len(analysis_df.columns),
            'Numeric_Features': len(numeric_columns),
            'Missing_Values': analysis_df.isnull().sum().sum(),
            'Duplicate_Rows': analysis_df.duplicated().sum(),
            'Memory_Usage_MB': analysis_df.memory_usage(deep=True).sum() / 1024**2,
            'Analysis_Date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        if 'tool_condition' in analysis_df.columns:
            condition_counts = analysis_df['tool_condition'].value_counts()
            for condition, count in condition_counts.items():
                analysis_summary[f'Tool_Condition_{condition}'] = count
        
        summary_df = pd.DataFrame([analysis_summary])
        csv_summary = summary_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Training Data Analysis",
            data=csv_summary,
            file_name="training_data_analysis.csv",
            mime="text/csv",
            key="download_training_analysis"
        )
    
    else:
        st.error("❌ Training data (train.csv) not found!")

elif analysis_type == "Experiment Data Comparison":
    st.markdown("### 🔬 Experiment Data Comparison")
    
    if experiment_files:
        # Select experiments to compare
        selected_experiments = st.multiselect(
            "Select experiment files to compare:",
            [os.path.basename(f) for f in experiment_files],
            default=[os.path.basename(f) for f in experiment_files[:3]],  # Default to first 3
            key="selected_experiments"
        )
        
        if selected_experiments:
            if st.button("🔍 Compare Experiments", type="primary", key="compare_experiments"):
                try:
                    comparison_data = {}
                    
                    with st.spinner("Loading and analyzing experiment data..."):
                        for exp_file in selected_experiments:
                            exp_path = f"data/{exp_file}"
                            if os.path.exists(exp_path):
                                exp_df = load_dataset(exp_path)
                                
                                # Calculate statistics for each experiment
                                numeric_cols = exp_df.select_dtypes(include=[np.number]).columns
                                
                                comparison_data[exp_file] = {
                                    'Total_Samples': len(exp_df),
                                    'Total_Features': len(exp_df.columns),
                                    'Numeric_Features': len(numeric_cols),
                                    'Missing_Values': exp_df.isnull().sum().sum(),
                                    'Memory_MB': exp_df.memory_usage(deep=True).sum() / 1024**2
                                }
                                
                                # Add feature means if available
                                if len(numeric_cols) > 0:
                                    for col in numeric_cols[:5]:  # First 5 numeric columns
                                        comparison_data[exp_file][f'Mean_{col}'] = exp_df[col].mean()
                    
                    # Create comparison DataFrame
                    comparison_df = pd.DataFrame(comparison_data).T
                    
                    st.markdown("### 📊 Experiment Comparison Results")
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    # Visualize comparisons
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Sample count comparison
                        fig_samples = px.bar(
                            x=list(comparison_data.keys()),
                            y=[data['Total_Samples'] for data in comparison_data.values()],
                            title="Sample Count Comparison",
                            labels={'x': 'Experiment', 'y': 'Number of Samples'}
                        )
                        st.plotly_chart(fig_samples, use_container_width=True)
                    
                    with col2:
                        # Feature count comparison
                        fig_features = px.bar(
                            x=list(comparison_data.keys()),
                            y=[data['Total_Features'] for data in comparison_data.values()],
                            title="Feature Count Comparison",
                            labels={'x': 'Experiment', 'y': 'Number of Features'},
                            color_discrete_sequence=['orange']
                        )
                        st.plotly_chart(fig_features, use_container_width=True)
                    
                    # Feature mean comparison (if available)
                    mean_columns = [col for col in comparison_df.columns if col.startswith('Mean_')]
                    if mean_columns:
                        st.markdown("### 📈 Feature Mean Comparison")
                        
                        for mean_col in mean_columns[:3]:  # Show first 3 mean columns
                            feature_name = mean_col.replace('Mean_', '')
                            
                            fig_mean = px.bar(
                                x=list(comparison_data.keys()),
                                y=[data.get(mean_col, 0) for data in comparison_data.values()],
                                title=f"Mean {feature_name} Comparison",
                                labels={'x': 'Experiment', 'y': f'Mean {feature_name}'}
                            )
                            st.plotly_chart(fig_mean, use_container_width=True)
                    
                    # Export comparison results
                    csv_comparison = comparison_df.to_csv()
                    st.download_button(
                        label="📥 Download Comparison Results",
                        data=csv_comparison,
                        file_name="experiment_comparison.csv",
                        mime="text/csv",
                        key="download_comparison"
                    )
                
                except Exception as e:
                    st.error(f"❌ Comparison failed: {str(e)}")
        else:
            st.warning("Please select at least one experiment file to compare.")
    else:
        st.warning("No experiment files found for comparison.")

elif analysis_type == "Feature Importance Analysis":
    st.markdown("### 🎯 Feature Importance Analysis")
    
    # Check if we have a trained model
    try:
        model = load_model()
        
        if hasattr(model, 'feature_importances_'):
            # Load training data to get feature names
            if os.path.exists(train_file):
                train_df = load_dataset(train_file)
                
                # Try to identify features used in training
                numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
                
                # Remove label column if present
                if 'tool_condition' in numeric_cols:
                    numeric_cols.remove('tool_condition')
                
                # Check if model expects these features
                expected_features = len(model.feature_importances_)
                
                if len(numeric_cols) >= expected_features:
                    feature_names = numeric_cols[:expected_features]
                    importances = model.feature_importances_
                    
                    # Create importance DataFrame
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importances,
                        'Importance_Percentage': importances * 100
                    }).sort_values('Importance', ascending=False)
                    
                    st.markdown("### 📊 Feature Importance Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Feature importance table
                        st.dataframe(importance_df, use_container_width=True)
                        
                        # Key insights
                        st.markdown("#### 🔍 Key Insights")
                        most_important = importance_df.iloc[0]
                        least_important = importance_df.iloc[-1]
                        
                        st.write(f"**Most Important Feature**: {most_important['Feature']} ({most_important['Importance_Percentage']:.1f}%)")
                        st.write(f"**Least Important Feature**: {least_important['Feature']} ({least_important['Importance_Percentage']:.1f}%)")
                        
                        # Top 3 features
                        top_3 = importance_df.head(3)
                        total_top_3 = top_3['Importance_Percentage'].sum()
                        st.write(f"**Top 3 Features Account for**: {total_top_3:.1f}% of importance")
                    
                    with col2:
                        # Feature importance bar chart
                        fig_importance = px.bar(
                            importance_df,
                            x='Importance_Percentage',
                            y='Feature',
                            orientation='h',
                            title="Feature Importance (%)",
                            color='Importance_Percentage',
                            color_continuous_scale='viridis'
                        )
                        fig_importance.update_layout(height=400)
                        st.plotly_chart(fig_importance, use_container_width=True)
                    
                    # Feature importance pie chart
                    st.markdown("### 🥧 Feature Importance Distribution")
                    
                    # Group small importances together
                    threshold = 5.0  # 5% threshold
                    pie_data = importance_df.copy()
                    
                    small_features = pie_data[pie_data['Importance_Percentage'] < threshold]
                    large_features = pie_data[pie_data['Importance_Percentage'] >= threshold]
                    
                    if len(small_features) > 1:
                        # Combine small features
                        other_importance = small_features['Importance_Percentage'].sum()
                        
                        # Create new dataframe with combined small features
                        pie_df = large_features.copy()
                        pie_df = pd.concat([pie_df, pd.DataFrame({
                            'Feature': ['Other Features'],
                            'Importance': [other_importance / 100],
                            'Importance_Percentage': [other_importance]
                        })], ignore_index=True)
                    else:
                        pie_df = pie_data
                    
                    fig_pie = px.pie(
                        pie_df,
                        values='Importance_Percentage',
                        names='Feature',
                        title="Feature Importance Distribution"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                    # Export feature importance
                    csv_importance = importance_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Feature Importance",
                        data=csv_importance,
                        file_name="feature_importance_analysis.csv",
                        mime="text/csv",
                        key="download_importance"
                    )
                
                else:
                    st.error(f"❌ Feature mismatch: Model expects {expected_features} features, but found {len(numeric_cols)}")
            else:
                st.error("❌ Training data not found for feature analysis")
        else:
            st.error("❌ The loaded model doesn't support feature importance analysis")
    
    except Exception as e:
        st.error(f"❌ Feature importance analysis failed: {str(e)}")

elif analysis_type == "Model Performance Analysis":
    st.markdown("### 🎯 Model Performance Analysis")
    
    # This would typically require validation data or cross-validation results
    st.info("📊 Model performance analysis requires evaluation results from Step 1.")
    
    # Load model and show basic info
    try:
        model = load_model()
        
        st.markdown("#### 🤖 Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Model Type**: {type(model).__name__}")
            
            if hasattr(model, 'n_estimators'):
                st.write(f"**Number of Estimators**: {model.n_estimators}")
            
            if hasattr(model, 'max_depth'):
                st.write(f"**Max Depth**: {model.max_depth}")
            
            if hasattr(model, 'random_state'):
                st.write(f"**Random State**: {model.random_state}")
        
        with col2:
            if hasattr(model, 'feature_importances_'):
                n_features = len(model.feature_importances_)
                st.write(f"**Expected Features**: {n_features}")
                st.write(f"**Feature Importance Available**: ✅ Yes")
            else:
                st.write(f"**Feature Importance Available**: ❌ No")
            
            if hasattr(model, 'classes_'):
                st.write(f"**Classes**: {list(model.classes_)}")
        
        # Model size estimation
        import pickle
        import io
        
        buffer = io.BytesIO()
        pickle.dump(model, buffer)
        model_size = len(buffer.getvalue()) / 1024  # Size in KB
        st.write(f"**Model Size**: {model_size:.2f} KB")
        
        # Recommendations for model evaluation
        st.markdown("#### 💡 Performance Analysis Recommendations")
        st.info("""
        To get comprehensive model performance metrics:
        1. **Go to Step 1** - Model Evaluation Dashboard
        2. **Upload test data** with known labels
        3. **Run evaluation** to get accuracy, precision, recall, F1-score
        4. **Review confusion matrix** and ROC curves
        5. **Export results** for detailed analysis
        """)
    
    except Exception as e:
        st.error(f"❌ Could not load model for analysis: {str(e)}")

elif analysis_type == "Data Quality Assessment":
    st.markdown("### 🔍 Data Quality Assessment")
    
    # Assess all available datasets
    quality_results = {}
    
    # Check training data
    if os.path.exists(train_file):
        train_df = load_dataset(train_file)
        quality_results['train.csv'] = assess_data_quality(train_df)
    
    # Check experiment data
    for exp_file in experiment_files[:5]:  # Limit to first 5 experiments
        exp_df = load_dataset(exp_file)
        filename = os.path.basename(exp_file)
        quality_results[filename] = assess_data_quality(exp_df)
    
    if quality_results:
        st.markdown("### 📊 Data Quality Assessment Results")
        
        # Create quality summary DataFrame
        quality_df = pd.DataFrame(quality_results).T
        st.dataframe(quality_df, use_container_width=True)
        
        # Quality visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Completeness score
            fig_completeness = px.bar(
                x=list(quality_results.keys()),
                y=[result['Completeness_Score'] for result in quality_results.values()],
                title="Data Completeness Score",
                labels={'x': 'Dataset', 'y': 'Completeness (%)'},
                color_discrete_sequence=['green']
            )
            st.plotly_chart(fig_completeness, use_container_width=True)
        
        with col2:
            # Duplicate percentage
            fig_duplicates = px.bar(
                x=list(quality_results.keys()),
                y=[result['Duplicate_Percentage'] for result in quality_results.values()],
                title="Duplicate Data Percentage",
                labels={'x': 'Dataset', 'y': 'Duplicates (%)'},
                color_discrete_sequence=['orange']
            )
            st.plotly_chart(fig_duplicates, use_container_width=True)
        
        # Quality recommendations
        st.markdown("### 💡 Data Quality Recommendations")
        
        for dataset, quality in quality_results.items():
            st.write(f"**{dataset}**:")
            
            if quality['Completeness_Score'] < 95:
                st.warning(f"⚠️ Low completeness ({quality['Completeness_Score']:.1f}%) - Consider data cleaning")
            
            if quality['Duplicate_Percentage'] > 5:
                st.warning(f"⚠️ High duplicates ({quality['Duplicate_Percentage']:.1f}%) - Remove duplicate rows")
            
            if quality['Completeness_Score'] >= 95 and quality['Duplicate_Percentage'] <= 5:
                st.success(f"✅ Good data quality")
        
        # Export quality assessment
        csv_quality = quality_df.to_csv()
        st.download_button(
            label="📥 Download Quality Assessment",
            data=csv_quality,
            file_name="data_quality_assessment.csv",
            mime="text/csv",
            key="download_quality"
        )
    
    else:
        st.error("❌ No datasets found for quality assessment")

elif analysis_type == "Comprehensive Report":
    st.markdown("### 📋 Comprehensive Analysis Report")
    
    if st.button("🚀 Generate Comprehensive Report", type="primary", key="generate_report"):
        with st.spinner("Generating comprehensive analysis report..."):
            
            # Collect all analysis data
            report_data = {
                'report_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'datasets_analyzed': [],
                'total_samples': 0,
                'total_features': 0,
                'model_info': {},
                'quality_summary': {},
                'feature_importance': {}
            }
            
            # Analyze training data
            if os.path.exists(train_file):
                train_df = load_dataset(train_file)
                report_data['datasets_analyzed'].append('train.csv')
                report_data['total_samples'] += len(train_df)
                report_data['total_features'] = len(train_df.columns)
                
                # Training data summary
                train_summary = {
                    'samples': len(train_df),
                    'features': len(train_df.columns),
                    'missing_values': train_df.isnull().sum().sum(),
                    'duplicates': train_df.duplicated().sum()
                }
                
                if 'tool_condition' in train_df.columns:
                    train_summary['label_distribution'] = train_df['tool_condition'].value_counts().to_dict()
                
                report_data['training_data'] = train_summary
            
            # Analyze experiment data
            experiment_summaries = {}
            for exp_file in experiment_files[:3]:  # First 3 experiments
                exp_df = load_dataset(exp_file)
                filename = os.path.basename(exp_file)
                
                experiment_summaries[filename] = {
                    'samples': len(exp_df),
                    'features': len(exp_df.columns),
                    'missing_values': exp_df.isnull().sum().sum(),
                    'duplicates': exp_df.duplicated().sum()
                }
                
                report_data['datasets_analyzed'].append(filename)
                report_data['total_samples'] += len(exp_df)
            
            report_data['experiments'] = experiment_summaries
            
            # Model information
            try:
                model = load_model()
                report_data['model_info'] = {
                    'type': type(model).__name__,
                    'feature_importance_available': hasattr(model, 'feature_importances_'),
                    'classes': list(model.classes_) if hasattr(model, 'classes_') else None
                }
                
                if hasattr(model, 'n_estimators'):
                    report_data['model_info']['n_estimators'] = model.n_estimators
                
            except Exception as e:
                report_data['model_info'] = {'error': str(e)}
            
            # Display comprehensive report
            st.markdown("### 📊 Comprehensive Analysis Report")
            
            # Executive Summary
            st.markdown("#### 📋 Executive Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📊 Datasets Analyzed", len(report_data['datasets_analyzed']))
            with col2:
                st.metric("📈 Total Samples", f"{report_data['total_samples']:,}")
            with col3:
                st.metric("🔧 Features (Training)", report_data['total_features'])
            with col4:
                st.metric("🤖 Model Type", report_data['model_info'].get('type', 'Unknown'))
            
            # Training Data Analysis
            if 'training_data' in report_data:
                st.markdown("#### 🎯 Training Data Analysis")
                
                train_data = report_data['training_data']
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Samples**: {train_data['samples']:,}")
                    st.write(f"**Features**: {train_data['features']}")
                    st.write(f"**Missing Values**: {train_data['missing_values']}")
                    st.write(f"**Duplicates**: {train_data['duplicates']}")
                
                with col2:
                    if 'label_distribution' in train_data:
                        st.write("**Label Distribution**:")
                        for label, count in train_data['label_distribution'].items():
                            percentage = (count / train_data['samples']) * 100
                            st.write(f"- {label}: {count} ({percentage:.1f}%)")
            
            # Experiment Data Summary
            if report_data['experiments']:
                st.markdown("#### 🔬 Experiment Data Summary")
                
                exp_df = pd.DataFrame(report_data['experiments']).T
                st.dataframe(exp_df, use_container_width=True)
            
            # Model Summary
            st.markdown("#### 🤖 Model Summary")
            
            model_info = report_data['model_info']
            if 'error' not in model_info:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Model Type**: {model_info.get('type', 'Unknown')}")
                    st.write(f"**Feature Importance**: {'✅ Available' if model_info.get('feature_importance_available') else '❌ Not Available'}")
                
                with col2:
                    if model_info.get('classes'):
                        st.write(f"**Classes**: {model_info['classes']}")
                    if model_info.get('n_estimators'):
                        st.write(f"**Estimators**: {model_info['n_estimators']}")
            else:
                st.error(f"Model analysis error: {model_info['error']}")
            
            # Recommendations
            st.markdown("#### 💡 Recommendations")
            
            recommendations = []
            
            # Data quality recommendations
            if 'training_data' in report_data:
                train_data = report_data['training_data']
                if train_data['missing_values'] > 0:
                    recommendations.append("🔍 Address missing values in training data")
                if train_data['duplicates'] > 0:
                    recommendations.append("🔄 Remove duplicate rows from training data")
            
            # Model recommendations
            if model_info.get('feature_importance_available'):
                recommendations.append("📊 Analyze feature importance to understand model decisions")
            
            recommendations.append("🎯 Run model evaluation with test data to assess performance")
            recommendations.append("📈 Consider cross-validation for more robust performance estimates")
            
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
            
            # Generate downloadable report
            report_text = f"""
# Comprehensive Analysis Report
Generated on: {report_data['report_date']}

## Executive Summary
- Datasets Analyzed: {len(report_data['datasets_analyzed'])}
- Total Samples: {report_data['total_samples']:,}
- Model Type: {report_data['model_info'].get('type', 'Unknown')}

## Training Data Summary
"""
            
            if 'training_data' in report_data:
                train_data = report_data['training_data']
                report_text += f"""
- Samples: {train_data['samples']:,}
- Features: {train_data['features']}
- Missing Values: {train_data['missing_values']}
- Duplicates: {train_data['duplicates']}
"""
                
                if 'label_distribution' in train_data:
                    report_text += "\n### Label Distribution:\n"
                    for label, count in train_data['label_distribution'].items():
                        percentage = (count / train_data['samples']) * 100
                        report_text += f"- {label}: {count} ({percentage:.1f}%)\n"
            
            if report_data['experiments']:
                report_text += "\n## Experiment Data Summary:\n"
                for exp_name, exp_data in report_data['experiments'].items():
                    report_text += f"\n### {exp_name}:\n"
                    report_text += f"- Samples: {exp_data['samples']:,}\n"
                    report_text += f"- Features: {exp_data['features']}\n"
                    report_text += f"- Missing Values: {exp_data['missing_values']}\n"
                    report_text += f"- Duplicates: {exp_data['duplicates']}\n"
            
            report_text += f"\n## Model Information:\n"
            if 'error' not in model_info:
                report_text += f"- Type: {model_info.get('type', 'Unknown')}\n"
                report_text += f"- Feature Importance Available: {model_info.get('feature_importance_available', False)}\n"
                if model_info.get('classes'):
                    report_text += f"- Classes: {model_info['classes']}\n"
            
            report_text += f"\n## Recommendations:\n"
            for i, rec in enumerate(recommendations, 1):
                # Remove emoji for text report
                clean_rec = rec.split(' ', 1)[1] if ' ' in rec else rec
                report_text += f"{i}. {clean_rec}\n"
            
            st.download_button(
                label="📥 Download Comprehensive Report",
                data=report_text,
                file_name=f"comprehensive_analysis_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown",
                key="download_comprehensive"
            )
            
            st.success("✅ Comprehensive report generated successfully!")

# Helper function for data quality assessment
def assess_data_quality(df):
    """Assess the quality of a dataset"""
    total_cells = len(df) * len(df.columns)
    missing_cells = df.isnull().sum().sum()
    completeness = ((total_cells - missing_cells) / total_cells) * 100
    
    duplicates = df.duplicated().sum()
    duplicate_percentage = (duplicates / len(df)) * 100
    
    return {
        'Total_Samples': len(df),
        'Total_Features': len(df.columns),
        'Missing_Values': missing_cells,
        'Completeness_Score': round(completeness, 2),
        'Duplicate_Count': duplicates,
        'Duplicate_Percentage': round(duplicate_percentage, 2),
        'Memory_Usage_MB': round(df.memory_usage(deep=True).sum() / 1024**2, 2)
    }

# Completion message
st.markdown("---")
st.markdown('<h2 style="color: #28a745;">🎉 Pipeline Complete!</h2>', unsafe_allow_html=True)
st.success("You have completed all steps of the Machine Learning Pipeline!")
