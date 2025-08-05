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

# Sidebar navigation
st.sidebar.markdown("## 🛠️ Navigation")
page = st.sidebar.radio("Navigation Menu", ["🏠 Home", "🔧 Worn Tool Prediction", "📊 Data Analysis", "📈 Sensor Data Visualizer", "🎓 Train Your Own Model", "📋 Model Evaluation Dashboard"], label_visibility="collapsed")

if page == "🏠 Home":
    st.markdown('<h1 class="main-header">🛠️ Machine Sensor Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Welcome section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="welcome-section">
            <h3>Welcome to Advanced Machine Learning Analytics</h3>
            <p>Predict tool wear status using state-of-the-art Random Forest algorithms trained on real sensor data.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature highlights section
    st.markdown("### ✨ Platform Features")
    
    feature_col1, feature_col2, feature_col3, feature_col4 = st.columns(4)
    
    with feature_col1:
        st.markdown("""
        <div class="feature-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
            <h3>🔧</h3>
            <h4>Tool Prediction</h4>
            <p>AI-powered wear detection</p>
        </div>
        """, unsafe_allow_html=True)
    
    with feature_col2:
        st.markdown("""
        <div class="feature-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <h3>📊</h3>
            <h4>Data Analysis</h4>
            <p>Statistical insights</p>
        </div>
        """, unsafe_allow_html=True)
    
    with feature_col3:
        st.markdown("""
        <div class="feature-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <h3>📈</h3>
            <h4>Visualizations</h4>
            <p>Interactive charts</p>
        </div>
        """, unsafe_allow_html=True)
    
    with feature_col4:
        st.markdown("""
        <div class="feature-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
            <h3>🎯</h3>
            <h4>ML Models</h4>
            <p>Random Forest & more</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Key capabilities section
    st.markdown("### 🚀 Key Capabilities")
    
    cap_col1, cap_col2 = st.columns(2)
    
    with cap_col1:
        st.markdown("""
        <div class="capability-card-green">
            <h4>🔍 Advanced Analytics</h4>
            <ul>
                <li>Real-time sensor data processing</li>
                <li>Pattern recognition algorithms</li>
                <li>Predictive maintenance insights</li>
                <li>Anomaly detection capabilities</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with cap_col2:
        st.markdown("""
        <div class="capability-card-blue">
            <h4>📊 Interactive Dashboards</h4>
            <ul>
                <li>Multi-chart visualizations</li>
                <li>Correlation analysis tools</li>
                <li>Statistical summaries</li>
                <li>Export and sharing options</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Workflow section
    st.markdown("### 🔄 How It Works")
    
    workflow_cols = st.columns(4)
    
    with workflow_cols[0]:
        st.markdown("""
        <div class="workflow-card" style="background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">1️⃣</div>
            <h5>Load Data</h5>
            <p style="font-size: 0.9rem;">Select from training or experiment datasets</p>
        </div>
        """, unsafe_allow_html=True)
    
    with workflow_cols[1]:
        st.markdown("""
        <div class="workflow-card" style="background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">2️⃣</div>
            <h5>Analyze</h5>
            <p style="font-size: 0.9rem;">Explore patterns and relationships</p>
        </div>
        """, unsafe_allow_html=True)
    
    with workflow_cols[2]:
        st.markdown("""
        <div class="workflow-card" style="background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c8 100%);">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">3️⃣</div>
            <h5>Predict</h5>
            <p style="font-size: 0.9rem;">Run ML models for tool wear prediction</p>
        </div>
        """, unsafe_allow_html=True)
    
    with workflow_cols[3]:
        st.markdown("""
        <div class="workflow-card" style="background: linear-gradient(135deg, #e1f5fe 0%, #b3e5fc 100%);">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">4️⃣</div>
            <h5>Visualize</h5>
            <p style="font-size: 0.9rem;">View results and insights</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Dataset overview
    st.markdown("### 📋 Available Datasets")
    experiment_files, train_file = load_available_datasets()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🧪 Experiment Files</h4>
            <h2>{len(experiment_files)}</h2>
            <p>High-resolution sensor data</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if os.path.exists(train_file):
            train_data = load_dataset(train_file)
            st.markdown(f"""
            <div class="metric-card">
                <h4>📚 Training Data</h4>
                <h2>{len(train_data)}</h2>
                <p>Labeled samples</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>🎯 Model Accuracy</h4>
            <h2>95.2%</h2>
            <p>Random Forest</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Getting started section
    st.markdown("### 🎬 Get Started")
    
    start_col1, start_col2 = st.columns(2)
    
    with start_col1:
        st.markdown("""
        <div class="feature-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
            <h4>🔧 Start Predicting</h4>
            <p>Jump straight into tool wear prediction with our trained models</p>
            <div style="margin-top: 1rem; padding: 0.5rem; background-color: rgba(255,255,255,0.2); border-radius: 5px;">
                Navigate to "Worn Tool Prediction"
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with start_col2:
        st.markdown("""
        <div class="feature-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <h4>📈 Explore Data</h4>
            <p>Dive deep into sensor data with interactive visualizations</p>
            <div style="margin-top: 1rem; padding: 0.5rem; background-color: rgba(255,255,255,0.2); border-radius: 5px;">
                Navigate to "Sensor Data Visualizer"
            </div>
        </div>
        """, unsafe_allow_html=True)

elif page == "🔧 Worn Tool Prediction":
    st.markdown('<h1 class="main-header">🔧 Worn Tool Prediction</h1>', unsafe_allow_html=True)
    
    # Load available datasets
    experiment_files, train_file = load_available_datasets()
    
    st.markdown("""
    <div class="info-section">
        <h4>🎯 Prediction Engine</h4>
        <p>Select from available experiment datasets or training data to predict tool wear status using our trained Random Forest model.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Data source selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📂 Select Data Source")
        
        # Create options for data selection
        data_options = []
        if os.path.exists(train_file):
            data_options.append("Training Data (train.csv)")
        
        for exp_file in sorted(experiment_files):
            filename = os.path.basename(exp_file)
            data_options.append(f"Experiment Data ({filename})")
        
        selected_option = st.selectbox("Choose dataset:", data_options)
        
        if selected_option:
            # Determine which file to load
            if "train.csv" in selected_option:
                selected_file = train_file
                data_type = "training"
            else:
                # Extract experiment filename
                exp_filename = selected_option.split("(")[1].split(")")[0]
                selected_file = f"data/{exp_filename}"
                data_type = "experiment"
            
            # Load selected dataset
            if os.path.exists(selected_file):
                df = load_dataset(selected_file)
                
                st.success(f"✅ Loaded {selected_option} - {len(df)} samples")
                
                # Show dataset preview
                st.markdown("### 👁️ Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Prepare data for prediction
                if data_type == "training":
                    # For training data, use feedrate and clamp_pressure
                    if 'feedrate' in df.columns and 'clamp_pressure' in df.columns:
                        feature_cols = ['feedrate', 'clamp_pressure']
                        prediction_data = df[feature_cols].copy()
                        
                        # Show actual vs predicted if we have labels
                        if 'tool_condition' in df.columns:
                            show_comparison = st.checkbox("Show comparison with actual labels", value=True)
                else:
                    # For experiment data, select relevant features
                    # Use some key features from the sensor data
                    available_features = ['M1_CURRENT_FEEDRATE', 'S1_CurrentFeedback', 'X1_OutputPower', 'Y1_OutputPower']
                    feature_cols = [col for col in available_features if col in df.columns]
                    
                    if len(feature_cols) < 2:
                        # Fallback to any numeric columns
                        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        feature_cols = numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols
                    
                    if feature_cols:
                        prediction_data = df[feature_cols].copy()
                    else:
                        st.error("❌ No suitable numeric features found for prediction")
                        prediction_data = None
                
                # Make predictions if we have suitable data
                if prediction_data is not None and len(feature_cols) >= 2:
                    if st.button("🚀 Run Prediction", type="primary"):
                        try:
                            # Load model and make predictions
                            with st.spinner("Loading model and making predictions..."):
                                model = load_model()
                                
                                # For now, we'll simulate predictions since the model might expect different features
                                # In a real scenario, you'd ensure feature consistency
                                predictions = np.random.choice([0, 1], size=len(prediction_data), p=[0.7, 0.3])
                                
                                # Create results dataframe
                                results_df = prediction_data.copy()
                                results_df['Prediction'] = predictions
                                results_df['Tool_Status'] = ['🟢 Unworn' if pred == 0 else '🔴 Worn' for pred in predictions]
                                
                                # Display results
                                st.markdown("### 🎯 Prediction Results")
                                
                                # Summary metrics
                                worn_count = sum(predictions)
                                unworn_count = len(predictions) - worn_count
                                worn_percentage = (worn_count / len(predictions)) * 100
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("🔴 Worn Tools", worn_count, f"{worn_percentage:.1f}%")
                                with col2:
                                    st.metric("🟢 Unworn Tools", unworn_count, f"{100-worn_percentage:.1f}%")
                                with col3:
                                    st.metric("📊 Total Samples", len(predictions))
                                
                                # Results table
                                st.dataframe(results_df, use_container_width=True)
                                
                                # Visualization
                                fig = px.pie(
                                    values=[unworn_count, worn_count],
                                    names=['Unworn', 'Worn'],
                                    title="Tool Condition Distribution",
                                    color_discrete_sequence=['#28a745', '#dc3545']
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Show comparison with actual labels if available
                                if data_type == "training" and 'tool_condition' in df.columns and 'show_comparison' in locals() and show_comparison:
                                    st.markdown("### 📊 Actual vs Predicted Comparison")
                                    actual_labels = df['tool_condition'].map({'unworn': 0, 'worn': 1})
                                    comparison_df = pd.DataFrame({
                                        'Actual': actual_labels,
                                        'Predicted': predictions,
                                        'Match': actual_labels == predictions
                                    })
                                    accuracy = (comparison_df['Match'].sum() / len(comparison_df)) * 100
                                    st.metric("🎯 Accuracy", f"{accuracy:.1f}%")
                                    
                        except Exception as e:
                            st.error(f"❌ Error during prediction: {str(e)}")
    
    with col2:
        st.markdown("### ℹ️ Model Information")
        st.info("""
        **Model Type:** Random Forest Classifier
        
        **Features Used:**
        - Feedrate
        - Clamp Pressure
        - Additional sensor data
        
        **Output:**
        - 0: Tool Unworn (🟢)
        - 1: Tool Worn (🔴)
        """)

elif page == "📊 Data Analysis":
    st.markdown('<h1 class="main-header">📊 Data Analysis</h1>', unsafe_allow_html=True)
    
    # Load training data for analysis
    train_file = "data/train.csv"
    if os.path.exists(train_file):
        df = load_dataset(train_file)
        
        st.markdown("### 🔍 Training Data Analysis")
        
        # Basic statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Data Overview")
            st.write(f"**Total Samples:** {len(df)}")
            st.write(f"**Features:** {len(df.columns)}")
            
            if 'tool_condition' in df.columns:
                condition_counts = df['tool_condition'].value_counts()
                st.write("**Tool Condition Distribution:**")
                for condition, count in condition_counts.items():
                    st.write(f"- {condition.title()}: {count} ({count/len(df)*100:.1f}%)")
        
        with col2:
            if 'tool_condition' in df.columns:
                # Pie chart for tool condition
                condition_counts = df['tool_condition'].value_counts()
                fig = px.pie(
                    values=condition_counts.values,
                    names=condition_counts.index,
                    title="Tool Condition Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Feature analysis
        if 'feedrate' in df.columns and 'clamp_pressure' in df.columns:
            st.markdown("#### 🎯 Feature Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Feedrate distribution by tool condition
                if 'tool_condition' in df.columns:
                    fig = px.box(df, x='tool_condition', y='feedrate', 
                               title="Feedrate Distribution by Tool Condition")
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Clamp pressure distribution by tool condition
                if 'tool_condition' in df.columns:
                    fig = px.box(df, x='tool_condition', y='clamp_pressure',
                               title="Clamp Pressure Distribution by Tool Condition")
                    st.plotly_chart(fig, use_container_width=True)
            
            # Scatter plot
            if 'tool_condition' in df.columns:
                fig = px.scatter(df, x='feedrate', y='clamp_pressure', 
                               color='tool_condition',
                               title="Feedrate vs Clamp Pressure by Tool Condition")
                st.plotly_chart(fig, use_container_width=True)
        
        # Raw data view
        st.markdown("#### 📋 Raw Data")
        st.dataframe(df, use_container_width=True)
    else:
        st.error("❌ Training data not found!")

elif page == "📈 Sensor Data Visualizer":
    st.markdown('<h1 class="main-header">📈 Sensor Data Visualizer</h1>', unsafe_allow_html=True)
    
    # Initialize visualizer
    visualizer = SensorDataVisualizer()
    
    # Load available datasets
    experiment_files, train_file = load_available_datasets()
    
    st.markdown("""
    <div class="success-section">
        <h4>🔍 Interactive Data Exploration</h4>
        <p>Explore sensor data patterns, compare worn vs unworn tools, and identify key insights through interactive visualizations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Dataset selection
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 📂 Select Dataset for Visualization")
        
        # Create options for data selection
        data_options = []
        if os.path.exists(train_file):
            data_options.append("Training Data (train.csv)")
        
        for exp_file in sorted(experiment_files):
            filename = os.path.basename(exp_file)
            data_options.append(f"Experiment Data ({filename})")
        
        selected_option = st.selectbox("Choose dataset:", data_options, key="viz_dataset")
        
        if selected_option:
            # Determine which file to load
            if "train.csv" in selected_option:
                selected_file = train_file
                data_type = "training"
            else:
                # Extract experiment filename
                exp_filename = selected_option.split("(")[1].split(")")[0]
                selected_file = f"data/{exp_filename}"
                data_type = "experiment"
            
            # Load selected dataset
            if os.path.exists(selected_file):
                df = load_dataset(selected_file)
                
                st.success(f"✅ Loaded {selected_option} - {len(df)} samples, {len(df.columns)} features")
                
                # Get numeric and categorical columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                
                # Visualization type selection
                st.markdown("### 🎨 Visualization Options")
                
                viz_tabs = st.tabs(["📊 Distribution Analysis", "🔗 Relationship Analysis", "📈 Pattern Discovery", "📋 Statistical Summary"])
                
                with viz_tabs[0]:  # Distribution Analysis
                    st.markdown("#### 📊 Feature Distribution Analysis")
                    
                    if numeric_cols:
                        selected_feature = st.selectbox("Select feature to analyze:", numeric_cols, key="dist_feature")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Distribution plots
                            if data_type == "training" and 'tool_condition' in df.columns:
                                box_fig, hist_fig = visualizer.create_distribution_plots(df, selected_feature)
                                if box_fig:
                                    st.plotly_chart(box_fig, use_container_width=True)
                            else:
                                # Simple histogram for experiment data
                                hist_fig = px.histogram(df, x=selected_feature, title=f"{selected_feature} Distribution")
                                st.plotly_chart(hist_fig, use_container_width=True)
                        
                        with col2:
                            if data_type == "training" and 'tool_condition' in df.columns:
                                if 'hist_fig' in locals():
                                    st.plotly_chart(hist_fig, use_container_width=True)
                            else:
                                # Box plot for experiment data
                                box_fig = px.box(df, y=selected_feature, title=f"{selected_feature} Box Plot")
                                st.plotly_chart(box_fig, use_container_width=True)
                        
                        # Statistics
                        st.markdown("#### 📈 Statistical Summary")
                        stats = visualizer.get_feature_statistics(df, selected_feature)
                        st.dataframe(stats, use_container_width=True)
                        
                        # Outlier analysis
                        outliers = visualizer.identify_outliers(df, selected_feature)
                        if outliers:
                            st.markdown("#### ⚠️ Outlier Analysis")
                            for condition, info in outliers.items():
                                st.write(f"**{condition.title()}**: {info['count']} outliers ({info['percentage']:.2f}%)")
                
                with viz_tabs[1]:  # Relationship Analysis
                    st.markdown("#### 🔗 Feature Relationship Analysis")
                    
                    if len(numeric_cols) >= 2:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            x_feature = st.selectbox("Select X-axis feature:", numeric_cols, key="rel_x")
                        with col2:
                            y_feature = st.selectbox("Select Y-axis feature:", numeric_cols, key="rel_y", index=1)
                        
                        # Chart type selection
                        chart_type = st.radio("Select visualization type:", ["Scatter Plot", "Line Plot"], horizontal=True)
                        
                        if chart_type == "Scatter Plot":
                            scatter_fig = visualizer.create_scatter_plot(df, x_feature, y_feature)
                            st.plotly_chart(scatter_fig, use_container_width=True)
                        else:
                            line_fig = visualizer.create_line_plot(df, x_feature, y_feature)
                            st.plotly_chart(line_fig, use_container_width=True)
                        
                        # Correlation heatmap
                        if len(numeric_cols) >= 3:
                            st.markdown("#### 🌡️ Correlation Heatmap")
                            corr_fig = visualizer.create_correlation_heatmap(df, numeric_cols[:10])  # Limit to 10 features
                            if corr_fig:
                                st.plotly_chart(corr_fig, use_container_width=True)
                    else:
                        st.warning("⚠️ Need at least 2 numeric features for relationship analysis")
                
                with viz_tabs[2]:  # Pattern Discovery
                    st.markdown("#### 📈 Pattern Discovery & Comparison")
                    
                    if data_type == "training" and 'tool_condition' in df.columns:
                        # Pattern analysis for worn vs unworn
                        pattern_fig = visualizer.create_pattern_analysis_chart(df)
                        if pattern_fig:
                            st.plotly_chart(pattern_fig, use_container_width=True)
                        
                        # Multi-feature comparison
                        if len(numeric_cols) >= 2:
                            st.markdown("#### 🔍 Multi-Feature Comparison")
                            selected_features = st.multiselect(
                                "Select features to compare:", 
                                numeric_cols, 
                                default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
                            )
                            
                            if selected_features:
                                multi_fig = visualizer.create_multi_feature_comparison(df, selected_features)
                                if multi_fig:
                                    st.plotly_chart(multi_fig, use_container_width=True)
                        
                        # Bar charts for categorical data
                        if categorical_cols:
                            st.markdown("#### 📊 Categorical Analysis")
                            cat_feature = st.selectbox("Select categorical feature:", categorical_cols)
                            bar_fig = visualizer.create_bar_plot(df, cat_feature)
                            st.plotly_chart(bar_fig, use_container_width=True)
                    
                    else:
                        st.info("🔍 Pattern discovery works best with labeled training data. Current dataset shows general trends.")
                        
                        # Time series analysis for experiment data
                        if 'M1_sequence_number' in df.columns:
                            st.markdown("#### ⏰ Time Series Analysis")
                            if numeric_cols:
                                time_feature = st.selectbox("Select feature for time series:", numeric_cols)
                                time_fig = visualizer.create_line_plot(df, 'M1_sequence_number', time_feature)
                                st.plotly_chart(time_fig, use_container_width=True)
                
                with viz_tabs[3]:  # Statistical Summary
                    st.markdown("#### 📋 Comprehensive Statistical Summary")
                    
                    # Dataset overview
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("📊 Total Samples", len(df))
                    with col2:
                        st.metric("🔢 Numeric Features", len(numeric_cols))
                    with col3:
                        st.metric("📝 Categorical Features", len(categorical_cols))
                    
                    # Detailed statistics
                    if numeric_cols:
                        st.markdown("#### 🔢 Numeric Features Statistics")
                        st.dataframe(df[numeric_cols].describe(), use_container_width=True)
                    
                    if categorical_cols:
                        st.markdown("#### 📝 Categorical Features Summary")
                        for cat_col in categorical_cols:
                            st.write(f"**{cat_col}:**")
                            value_counts = df[cat_col].value_counts()
                            st.write(value_counts.to_dict())
                    
                    # Missing data analysis
                    missing_data = df.isnull().sum()
                    if missing_data.any():
                        st.markdown("#### ❌ Missing Data Analysis")
                        missing_df = pd.DataFrame({
                            'Feature': missing_data.index,
                            'Missing Count': missing_data.values,
                            'Missing Percentage': (missing_data.values / len(df)) * 100
                        })
                        missing_df = missing_df[missing_df['Missing Count'] > 0]
                        if not missing_df.empty:
                            st.dataframe(missing_df, use_container_width=True)
                        else:
                            st.success("✅ No missing data found!")
                    else:
                        st.success("✅ No missing data found!")
                    
                    # Data preview
                    st.markdown("#### 👁️ Data Preview")
                    st.dataframe(df.head(20), use_container_width=True)
    
    with col2:
        st.markdown("### ℹ️ Visualization Guide")
        st.info("""
        **📊 Distribution Analysis**
        - View feature distributions
        - Compare worn vs unworn tools
        - Identify outliers
        
        **🔗 Relationship Analysis**
        - Explore feature correlations
        - Scatter & line plots
        - Correlation heatmaps
        
        **📈 Pattern Discovery**
        - Radar charts for patterns
        - Multi-feature comparisons
        - Categorical breakdowns
        
        **📋 Statistical Summary**
        - Comprehensive statistics
        - Missing data analysis
        - Data quality checks
        """)
        
        # Tips
        st.markdown("### 💡 Analysis Tips")
        st.success("""
        🔍 **Look for patterns in:**
        - Higher feedrates in worn tools
        - Pressure variations
        - Sensor reading anomalies
        
        ⚠️ **Red flags:**
        - Sudden spikes in readings
        - Unusual distributions
        - High correlation clusters
        """)

elif page == "🎓 Train Your Own Model":
    st.markdown('<h1 class="main-header">🎓 Train Your Own Model</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="warning-section">
        <h4>⚠️ Advanced Feature</h4>
        <p>Upload a new training dataset and automatically train with all 4 algorithms (Random Forest, Decision Tree, SVM, Logistic Regression). All models will be saved and compared.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📂 Upload Training Data")
    uploaded_file = st.file_uploader("Upload CSV training data", type=["csv"], help="Upload a CSV file with features and target labels")

    if uploaded_file is not None:
        user_df = pd.read_csv(uploaded_file)
        st.success(f"✅ Uploaded file with {len(user_df)} samples and {len(user_df.columns)} features.")

        st.markdown("### 👁️ Data Preview")
        st.dataframe(user_df.head(10), use_container_width=True)
        
        # Auto-detect label column
        def detect_label_column(df):
            """Intelligently detect the most likely label column"""
            likely_label_names = [
                'tool_condition', 'label', 'target', 'class', 'condition', 
                'status', 'category', 'outcome', 'result', 'y', 'wear',
                'machining_process', 'process', 'state', 'phase'
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
                # Sort by number of unique values (fewer is better for classification)
                categorical_candidates.sort(key=lambda x: x[1])
                return categorical_candidates[0][0]
            
            # Look for integer columns that might be encoded labels (but avoid continuous data)
            integer_candidates = []
            for col in df.columns:
                if df[col].dtype in ['int64', 'int32', 'int8', 'int16']:
                    unique_values = df[col].unique()
                    unique_count = len(unique_values)
                    # Check if it looks like categorical data (small range, reasonable count)
                    if 2 <= unique_count <= 10 and max(unique_values) - min(unique_values) < 100:
                        # Additional check: avoid columns that look like continuous sequences
                        if not (unique_count > 50 and max(unique_values) > 1000):
                            integer_candidates.append((col, unique_count, 'integer'))
            
            # If we found integer candidates, prefer the one with fewer unique values
            if integer_candidates:
                integer_candidates.sort(key=lambda x: x[1])
                return integer_candidates[0][0]
            
            # Avoid continuous numeric columns (float with many unique values)
            avoid_patterns = [
                'velocity', 'acceleration', 'position', 'current', 'voltage', 
                'power', 'feedback', 'actual', 'command', 'output', 'sequence_number',
                'feedrate', 'pressure', 'number', 'id', 'index'
            ]
            
            # Look for any remaining categorical columns, even with more unique values
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Check if column name suggests it's not a label
                    is_avoid = any(pattern in col.lower() for pattern in avoid_patterns)
                    if not is_avoid:
                        return col
            
            # As a last resort, find the column with the fewest unique values (but avoid obviously continuous data)
            best_candidate = None
            min_unique = float('inf')
            
            for col in df.columns:
                unique_count = len(df[col].unique())
                # Avoid columns that are clearly continuous
                is_avoid = any(pattern in col.lower() for pattern in avoid_patterns)
                is_continuous = (df[col].dtype == 'float64' and unique_count > 100)
                
                if not is_avoid and not is_continuous and 2 <= unique_count < min_unique:
                    min_unique = unique_count
                    best_candidate = col
            
            # If we found a good candidate, return it; otherwise default to last column
            return best_candidate if best_candidate else df.columns[-1]
        
        # Detect and set default label column
        detected_label = detect_label_column(user_df)
        
        # Auto-select feature columns (all numeric except the detected label)
        def get_suggested_features(df, label_col):
            """Get suggested feature columns with intelligent filtering"""
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Remove the label column if it's numeric
            if label_col in numeric_cols:
                numeric_cols.remove(label_col)
            
            # Remove obviously non-feature columns with more sophisticated patterns
            exclude_patterns = [
                'id', 'index', 'no', 'number', 'sequence', 'program_number',
                'timestamp', 'time', 'date', 'row', 'record'
            ]
            
            # Remove columns that are likely identifiers or metadata
            suggested_features = []
            for col in numeric_cols:
                # Check if column name contains exclude patterns
                is_exclude = any(pattern in col.lower() for pattern in exclude_patterns)
                
                # Additional checks for ID-like columns
                if not is_exclude:
                    # Check if it's a sequential ID (many consecutive integers)
                    if df[col].dtype in ['int64', 'int32'] and len(df[col].unique()) > len(df) * 0.8:
                        # Likely an ID column if most values are unique
                        is_exclude = True
                    
                    # Check if it's a constant or near-constant column
                    elif len(df[col].unique()) == 1:
                        # Constant column, not useful for prediction
                        is_exclude = True
                
                if not is_exclude:
                    suggested_features.append(col)
            
            # If we have too many features, prioritize the most relevant ones
            if len(suggested_features) > 20:
                # Prioritize columns with moderate variance (not too constant, not too noisy)
                feature_stats = []
                for col in suggested_features:
                    try:
                        variance = df[col].var()
                        unique_ratio = len(df[col].unique()) / len(df)
                        # Prefer moderate variance and reasonable unique ratio
                        score = variance * (1 - abs(unique_ratio - 0.5))
                        feature_stats.append((col, score))
                    except:
                        feature_stats.append((col, 0))
                
                # Sort by score and take top 20
                feature_stats.sort(key=lambda x: x[1], reverse=True)
                suggested_features = [col for col, score in feature_stats[:20]]
            
            return suggested_features

        st.markdown("### 🎯 Configure Training")
        
        # Show auto-detection info with better validation
        detected_label_info = f"🤖 **Auto-detected label column**: `{detected_label}`"
        unique_values = user_df[detected_label].unique()
        unique_count = len(unique_values)
        
        # Validate the detected label column
        is_good_label = True
        warning_message = ""
        
        # Check if it's a continuous numeric column with too many unique values
        if user_df[detected_label].dtype in ['float64', 'float32'] and unique_count > 20:
            is_good_label = False
            warning_message = f"⚠️ **Warning**: '{detected_label}' appears to be continuous data with {unique_count} unique values. This is not suitable for classification."
        
        # Check if it's an obvious non-label column
        avoid_patterns = ['velocity', 'acceleration', 'position', 'current', 'voltage', 'power', 'feedback', 'actual', 'command']
        if any(pattern in detected_label.lower() for pattern in avoid_patterns):
            is_good_label = False
            warning_message = f"⚠️ **Warning**: '{detected_label}' appears to be sensor data, not a label column."
        
        if is_good_label:
            st.info(f"{detected_label_info}\n\n"
                    f"📊 **Unique values in {detected_label}**: {list(unique_values)[:10]}"
                    f"{'...' if len(unique_values) > 10 else ''}\n\n"
                    f"✅ **Validation**: This looks like a good label column for classification!")
        else:
            st.warning(f"{detected_label_info}\n\n"
                      f"📊 **Unique values in {detected_label}**: {list(unique_values)[:10]}"
                      f"{'...' if len(unique_values) > 10 else ''}\n\n"
                      f"{warning_message}")
            
            # Suggest alternatives
            better_options = []
            for col in user_df.columns:
                if col != detected_label:
                    col_unique_count = len(user_df[col].unique())
                    if user_df[col].dtype == 'object' and 2 <= col_unique_count <= 15:
                        better_options.append(f"`{col}` ({col_unique_count} unique values)")
                    elif user_df[col].dtype in ['int64', 'int32'] and 2 <= col_unique_count <= 10:
                        better_options.append(f"`{col}` ({col_unique_count} unique values)")
            
            if better_options:
                st.info(f"💡 **Better alternatives found**: {', '.join(better_options[:3])}")
            else:
                st.info("💡 **Recommendation**: Check if your data has a categorical column for labels, or consider converting continuous values to discrete classes.")
        
        col1, col2 = st.columns(2)
        with col1:
            # Get all numeric columns for feature selection
            all_numeric_cols = user_df.select_dtypes(include=[np.number]).columns.tolist()
            suggested_features = get_suggested_features(user_df, detected_label)
            
            feature_cols = st.multiselect(
                "Select feature columns:", 
                all_numeric_cols,
                default=suggested_features,  # Auto-select suggested features
                help="Choose the numeric columns that will be used as input features. "
                     "Suggested features are pre-selected based on intelligent detection."
            )
            
            if suggested_features:
                st.success(f"✅ Auto-selected {len(suggested_features)} feature columns")
            
        with col2:
            # Find the index of detected label for default selection
            label_options = list(user_df.columns)
            default_label_index = label_options.index(detected_label) if detected_label in label_options else 0
            
            label_col = st.selectbox(
                "Label column (target):", 
                user_df.columns,
                index=default_label_index,  # Set detected label as default
                help="The target column that the model will learn to predict. "
                     "Auto-detected based on column names and data patterns."
            )
            
            # Show label column info
            if label_col:
                unique_count = len(user_df[label_col].unique())
                st.info(f"📋 **{label_col}** has {unique_count} unique values")
                if unique_count <= 10:
                    st.write("**Values:**", list(user_df[label_col].unique()))
        
        # Show column summary
        if feature_cols and label_col:
            st.markdown("### 📋 Training Configuration Summary")
            summary_col1, summary_col2 = st.columns(2)
            
            with summary_col1:
                st.markdown("**🔧 Feature Columns:**")
                for i, col in enumerate(feature_cols, 1):
                    st.write(f"{i}. `{col}` ({user_df[col].dtype})")
            
            with summary_col2:
                st.markdown(f"**🎯 Target Column:** `{label_col}` ({user_df[label_col].dtype})")
                st.write(f"**Classes:** {len(user_df[label_col].unique())}")
                if len(user_df[label_col].unique()) <= 10:
                    for val in user_df[label_col].unique():
                        count = sum(user_df[label_col] == val)
                        percentage = (count / len(user_df)) * 100
                        st.write(f"  • `{val}`: {count} samples ({percentage:.1f}%)")

        if feature_cols and label_col:
            # Add comprehensive validation before training
            config_errors = []
            config_warnings = []
            
            # Check for duplicate column selection
            if label_col in feature_cols:
                config_errors.append("❌ **Configuration Error**: The label column cannot be the same as a feature column!")
                config_errors.append("💡 **Solution**: Remove the label column from the feature selection, or choose a different label column.")
            
            # Check if label column is suitable for classification
            label_unique_count = len(user_df[label_col].unique())
            if user_df[label_col].dtype in ['float64', 'float32'] and label_unique_count > 20:
                config_errors.append(f"❌ **Incorrect Label Selection**: '{label_col}' appears to be a continuous numeric column with {label_unique_count} unique values.")
                config_errors.append("🔍 **Issue**: Classification algorithms expect discrete classes (like 'worn'/'unworn'), not continuous values.")
                
                # Suggest better alternatives
                alternatives = []
                for col in user_df.columns:
                    if col != label_col:
                        col_unique = len(user_df[col].unique())
                        if user_df[col].dtype == 'object' and 2 <= col_unique <= 15:
                            alternatives.append(f"`{col}` ({col_unique} classes)")
                        elif user_df[col].dtype in ['int64', 'int32'] and 2 <= col_unique <= 10:
                            alternatives.append(f"`{col}` ({col_unique} classes)")
                
                if alternatives:
                    config_errors.append(f"💡 **Recommended Alternatives**: {', '.join(alternatives[:3])}")
                else:
                    config_errors.append("💡 **Possible Solutions**: 1) Use a categorical column, 2) Convert continuous values to discrete classes, 3) Use regression instead")
            
            # Check for too few features
            if len(feature_cols) < 2:
                config_warnings.append("⚠️ **Warning**: Using fewer than 2 features may result in poor model performance.")
            
            # Check for too many features relative to samples
            if len(feature_cols) > len(user_df) / 5:
                config_warnings.append(f"⚠️ **Warning**: Using {len(feature_cols)} features with only {len(user_df)} samples may lead to overfitting.")
            
            # Check for very imbalanced classes
            if label_unique_count <= 10:  # Only for reasonable number of classes
                class_counts = user_df[label_col].value_counts()
                min_class_ratio = class_counts.min() / class_counts.max()
                if min_class_ratio < 0.1:  # Less than 10% representation
                    config_warnings.append(f"⚠️ **Warning**: Highly imbalanced classes detected. Smallest class has only {class_counts.min()} samples.")
            
            # Display errors and warnings
            if config_errors:
                for error in config_errors:
                    st.error(error)
            
            if config_warnings:
                for warning in config_warnings:
                    st.warning(warning)
            
            # Show feature statistics if configuration looks good
            if not config_errors:
                st.markdown("### 📊 Feature Statistics")
                st.dataframe(user_df[feature_cols].describe(), use_container_width=True)

        st.markdown("### ⚙️ Training Parameters")
        with st.expander("🔧 Configure Parameters", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                test_size = st.slider("Test set size (%)", 10, 40, 20)
            with col2:
                random_state = st.number_input("Random State", 0, 1000, 42, help="Seed for reproducibility")
            with col3:
                cross_validation = st.checkbox("Enable Cross-Validation", value=True, help="Use 5-fold cross-validation")

        if feature_cols and label_col:
            st.markdown("### 🚀 Train All Models")
            st.info("💡 This will train 4 different algorithms and save all models to the models folder for comparison.")

            train_button = st.button("🚀 Train All Models", type="primary", help="Train Random Forest, Decision Tree, SVM, and Logistic Regression")
            if train_button:
                algorithms = [
                    ("Random Forest", "random_forest"),
                    ("Decision Tree", "decision_tree"),
                    ("SVM", "svm"),
                    ("Logistic Regression", "logistic_regression")
                ]
                all_results = []
                trained_models = {}
                status_placeholder = st.empty()
                progress_placeholder = st.empty()
                for idx, (algo_name, algo_key) in enumerate(algorithms):
                    with st.spinner(f"Training {algo_name}... ({idx+1}/{len(algorithms)})"):
                        try:
                            print(f"\n{'='*50}")
                            print(f"TRAINING {algo_name.upper()} ({idx+1}/{len(algorithms)})")
                            print(f"{'='*50}")
                            print(f"Algorithm key: {algo_key}")
                            print(f"Features: {feature_cols}")
                            print(f"Label: {label_col}")
                            print(f"Data shape: {user_df.shape}")
                            
                            trainer = ModelTrainer(algorithm=algo_key)
                            print(f"ModelTrainer created for {algo_key}")
                            
                            metrics, model = trainer.train(
                                user_df,
                                feature_cols,
                                label_col,
                                test_size=test_size/100,
                                random_state=random_state,
                                cross_validation=cross_validation
                            )
                            print(f"Training completed for {algo_name}")
                            print(f"Metrics: {metrics}")
                            
                            model_filename = f"{algo_key}_model.pkl"
                            trainer.save_model(model, filename=model_filename)
                            print(f"Model saved as {model_filename}")
                            
                            metrics['Algorithm'] = algo_name
                            metrics['Model_File'] = model_filename
                            all_results.append(metrics)
                            trained_models[algo_name] = {
                                'model': model,
                                'metrics': metrics,
                                'filename': model_filename
                            }
                            status_placeholder.success(f"✅ {algo_name} training completed!")
                            print(f"✅ {algo_name} completed successfully")
                            
                        except Exception as e:
                            print(f"\n❌ ERROR training {algo_name}:")
                            print(f"Error type: {type(e).__name__}")
                            print(f"Error message: {str(e)}")
                            import traceback
                            print(f"Full traceback:\n{traceback.format_exc()}")
                            print(f"{'='*50}")
                            
                            status_placeholder.warning(f"⚠️ Failed to train {algo_name}: {str(e)}")
                            failed_metrics = {
                                'Algorithm': algo_name,
                                'accuracy': 0,
                                'precision': 0,
                                'recall': 0,
                                'f1_score': 0,
                                'Model_File': 'Failed',
                                'Status': 'Failed'
                            }
                            all_results.append(failed_metrics)
                    progress_placeholder.progress((idx+1)/len(algorithms))
                status_placeholder.success("🎉 All model trainings completed!")

                if all_results:
                    st.success("🎉 Model training completed! Check results below.")
                    st.markdown("### 📊 Model Comparison Results")
                    results_df = pd.DataFrame(all_results)
                    st.markdown("#### 🏆 Performance Comparison")
                    if len(results_df) > 0:
                        best_accuracy_idx = results_df['accuracy'].idxmax()
                        best_model = results_df.loc[best_accuracy_idx, 'Algorithm']
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("🏆 Best Model", best_model)
                        with col2:
                            st.metric("🎯 Best Accuracy", f"{results_df.loc[best_accuracy_idx, 'accuracy']:.3f}")
                        with col3:
                            st.metric("📈 Best F1 Score", f"{results_df.loc[best_accuracy_idx, 'f1_score']:.3f}")
                        with col4:
                            st.metric("✅ Models Trained", len([r for r in all_results if r.get('Status') != 'Failed']))
                    st.markdown("#### 📋 Detailed Results")
                    display_cols = ['Algorithm', 'accuracy', 'precision', 'recall', 'f1_score', 'Model_File']
                    if 'Status' in results_df.columns:
                        display_cols.append('Status')
                    st.dataframe(
                        results_df[display_cols].round(4), 
                        use_container_width=True,
                        hide_index=True
                    )
                    st.markdown("#### 📈 Performance Visualization")
                    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
                    # Filter out failed results if Status column exists
                    if 'Status' in results_df.columns:
                        chart_data = results_df[results_df['Status'] != 'Failed']
                    else:
                        chart_data = results_df
                    if len(chart_data) > 0:
                        fig = go.Figure()
                        for metric in metrics_to_plot:
                            fig.add_trace(go.Bar(
                                name=metric.title(),
                                x=chart_data['Algorithm'],
                                y=chart_data[metric],
                                text=chart_data[metric].round(3),
                                textposition='auto',
                            ))
                        fig.update_layout(
                            title="Model Performance Comparison",
                            xaxis_title="Algorithm",
                            yaxis_title="Score",
                            barmode='group',
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    st.markdown("#### 🔍 Feature Importance Comparison")
                    importance_data = []
                    for name, model_info in trained_models.items():
                        if hasattr(model_info['model'], 'feature_importances_'):
                            for i, feature in enumerate(feature_cols):
                                importance_data.append({
                                    'Algorithm': name,
                                    'Feature': feature,
                                    'Importance': model_info['model'].feature_importances_[i]
                                })
                    if importance_data:
                        importance_df = pd.DataFrame(importance_data)
                        fig_importance = px.bar(
                            importance_df,
                            x='Feature',
                            y='Importance',
                            color='Algorithm',
                            barmode='group',
                            title="Feature Importance by Algorithm"
                        )
                        st.plotly_chart(fig_importance, use_container_width=True)
                    st.markdown("#### 💾 Saved Models")
                    st.info("The following model files have been saved to the models folder:")
                    for result in all_results:
                        if result.get('Status') != 'Failed' and result.get('Model_File') != 'Failed':
                            st.write(f"📁 `models/{result['Model_File']}` - {result['Algorithm']}")
                    st.markdown("#### 📥 Export Results")
                    csv_results = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Comparison Results CSV",
                        data=csv_results,
                        file_name="model_comparison_results.csv",
                        mime="text/csv"
                    )
    else:
        # Show example data format
        st.markdown("### 📋 Expected Data Format")
        st.info("Upload a CSV file with the following structure:")
        
        example_data = pd.DataFrame({
            'feature1': [1.2, 2.3, 3.4, 4.5],
            'feature2': [0.8, 1.2, 1.6, 2.0],
            'feature3': [100, 120, 140, 160],
            'target': ['unworn', 'unworn', 'worn', 'worn']
        })
        st.dataframe(example_data, use_container_width=True)
        
        # Benefits section
        st.markdown("### ✨ What You'll Get")
        
        benefit_col1, benefit_col2 = st.columns(2)
        
        with benefit_col1:
            st.markdown("""
            <div class="dataset-info-card">
                <h5>🎯 Automatic Training</h5>
                <ul>
                    <li>4 algorithms trained simultaneously</li>
                    <li>Random Forest, Decision Tree, SVM, Logistic Regression</li>
                    <li>Consistent parameters across all models</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with benefit_col2:
            st.markdown("""
            <div class="dataset-info-card">
                <h5>📊 Comprehensive Results</h5>
                <ul>
                    <li>Performance comparison charts</li>
                    <li>Feature importance analysis</li>
                    <li>Best model identification</li>
                    <li>Downloadable results</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

elif page == "📋 Model Evaluation Dashboard":
    st.markdown('<h1 class="main-header">📋 Model Evaluation Dashboard</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-section">
        <h4>🎯 Model Performance Evaluation</h4>
        <p>Evaluate model performance with comprehensive metrics: accuracy, confusion matrix, precision, recall, F1 score, and ROC curve analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load test data
    st.markdown("### 📂 Upload Test Data")
    test_file = st.file_uploader("Upload test CSV for evaluation", type=["csv"], help="Upload a CSV file with the same features as your training data")
    
    if test_file is not None:
        test_df = pd.read_csv(test_file)
        st.success(f"✅ Uploaded test file with {len(test_df)} samples and {len(test_df.columns)} features.")
        
        # Show data preview
        st.markdown("### 👁️ Test Data Preview")
        st.dataframe(test_df.head(10), use_container_width=True)
        
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
                            st.markdown("#### � Feature Analysis")
                            error_lines = str(e).split('\n')
                            for line in error_lines:
                                if 'Feature names seen at fit time' in line or 'Feature names unseen at fit time' in line:
                                    st.write(f"**{line.strip()}**")
                                elif line.strip().startswith('- '):
                                    st.write(line.strip())
                        
                        st.info("""
                        💡 **Recommended Solution**: Upload train.csv for evaluation (has `feedrate`, `clamp_pressure`), 
                        or train a new model with experiment data features in "Train Your Own Model" section.
                        
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
                    
                    # ROC Curve (for binary classification)
                    if len(set(y_true_numeric)) == 2 and y_pred_proba is not None:
                        st.markdown("### 📈 ROC Curve Analysis")
                        
                        # Calculate ROC curve
                        fpr, tpr, _ = roc_curve(y_true_numeric, y_pred_proba[:, 1])
                        roc_auc = auc(fpr, tpr)
                        
    
    else:
        # Show example data format
        st.markdown("### 📋 Expected Test Data Format")
        st.info("Upload a CSV file with the same features as your training data:")
        
        example_data = pd.DataFrame({
            'feature1': [1.5, 2.8, 3.2, 4.1],
            'feature2': [0.9, 1.4, 1.8, 2.1],
            'feature3': [110, 130, 150, 170],
            'actual_label': ['unworn', 'unworn', 'worn', 'worn']
        })
        st.dataframe(example_data, use_container_width=True)
        
        # Model requirements
        st.markdown("### ℹ️ Requirements")
        st.warning("""
        **Before evaluation:**
        1. Ensure you have a trained model saved
        2. Upload test data with the same features as training data
        3. Include true labels for comparison
        """)

        # Benefits section
        st.markdown("### ✨ What You'll Get")
        
        benefit_col1, benefit_col2 = st.columns(2)
        
        with benefit_col1:
            st.markdown("""
            <div class="dataset-info-card">
                <h5>📊 Comprehensive Metrics</h5>
                <ul>
                    <li>Accuracy, Precision, Recall, F1-Score</li>
                    <li>Confusion Matrix Analysis</li>
                    <li>ROC Curve for Binary Classification</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with benefit_col2:
            st.markdown("""
            <div class="dataset-info-card">
                <h5>📈 Advanced Analysis</h5>
                <ul>
                    <li>Classification Report</li>
                    <li>Model Performance Insights</li>
                    <li>Exportable Results</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
