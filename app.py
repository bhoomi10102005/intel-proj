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

# Enhanced Dynamic CSS for comprehensive dark/light mode support
st.markdown("""
<style>
    /* Base theme-adaptive variables */
    :root {
        --primary-color: #1f77b4;
        --primary-color-dark: #64b5f6;
        --success-color: #28a745;
        --success-color-dark: #4caf50;
        --warning-color: #ffc107;
        --warning-color-dark: #ffeb3b;
        --danger-color: #dc3545;
        --danger-color-dark: #ff6b6b;
        --info-color: #17a2b8;
        --info-color-dark: #29b6f6;
        
        /* Animation variables */
        --transition-fast: 0.15s ease;
        --transition-normal: 0.3s ease;
        --transition-slow: 0.5s ease;
        --border-radius-sm: 6px;
        --border-radius-md: 8px;
        --border-radius-lg: 12px;
        --border-radius-xl: 16px;
        
        /* Shadow variables */
        --shadow-sm: 0 2px 4px rgba(0, 0, 0, 0.1);
        --shadow-md: 0 4px 8px rgba(0, 0, 0, 0.12);
        --shadow-lg: 0 8px 16px rgba(0, 0, 0, 0.15);
        --shadow-xl: 0 12px 24px rgba(0, 0, 0, 0.18);
    }

    /* Dark theme styles */
    @media (prefers-color-scheme: dark) {
        :root {
            --bg-primary: #0e1117;
            --bg-secondary: #262730;
            --bg-tertiary: #1a1d29;
            --bg-quaternary: #2f3349;
            --text-primary: #fafafa;
            --text-secondary: #c9c9c9;
            --text-tertiary: #8b949e;
            --border-color: #404040;
            --border-color-light: #525562;
            --shadow-color: rgba(0, 0, 0, 0.4);
            --card-bg: #1e1e1e;
            --hover-bg: #2a2a2a;
            --input-bg: #21262d;
            --input-border: #484f58;
            --input-focus: #0969da;
            --overlay-bg: rgba(0, 0, 0, 0.8);
        }
        
        .main-header {
            color: var(--primary-color-dark) !important;
            text-shadow: 0 0 10px rgba(100, 181, 246, 0.3);
        }
        
        .metric-card {
            background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%) !important;
            border: 1px solid var(--border-color) !important;
            color: var(--text-primary) !important;
            box-shadow: var(--shadow-lg) !important;
        }
        
        .prediction-success {
            background: linear-gradient(135deg, rgba(76, 175, 80, 0.15) 0%, rgba(129, 199, 132, 0.15) 100%) !important;
            color: var(--success-color-dark) !important;
            border-left-color: var(--success-color-dark) !important;
            border: 1px solid rgba(76, 175, 80, 0.3) !important;
        }
        
        .prediction-warning {
            background: linear-gradient(135deg, rgba(255, 235, 59, 0.15) 0%, rgba(255, 241, 118, 0.15) 100%) !important;
            color: var(--warning-color-dark) !important;
            border-left-color: var(--warning-color-dark) !important;
            border: 1px solid rgba(255, 235, 59, 0.3) !important;
        }
        
        .welcome-section {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color) !important;
        }
        
        .feature-card {
            background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-quaternary) 100%) !important;
            border: 1px solid var(--border-color) !important;
            box-shadow: var(--shadow-xl) !important;
        }
        
        .capability-card-green {
            background: linear-gradient(135deg, rgba(76, 175, 80, 0.2) 0%, rgba(129, 199, 132, 0.2) 100%) !important;
            border-left-color: var(--success-color-dark) !important;
            color: var(--text-primary) !important;
        }
        
        .capability-card-blue {
            background: linear-gradient(135deg, rgba(100, 181, 246, 0.2) 0%, rgba(144, 202, 249, 0.2) 100%) !important;
            border-left-color: var(--primary-color-dark) !important;
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
            --bg-quaternary: #dee2e6;
            --text-primary: #212529;
            --text-secondary: #6c757d;
            --text-tertiary: #868e96;
            --border-color: #dee2e6;
            --border-color-light: #e9ecef;
            --shadow-color: rgba(0, 0, 0, 0.1);
            --card-bg: #ffffff;
            --hover-bg: #f8f9fa;
            --input-bg: #ffffff;
            --input-border: #ced4da;
            --input-focus: #0d6efd;
            --overlay-bg: rgba(0, 0, 0, 0.5);
        }
        
        .main-header {
            color: var(--primary-color) !important;
            text-shadow: 0 2px 4px rgba(31, 119, 180, 0.1);
        }
        
        .metric-card {
            background: linear-gradient(135deg, #f0f2f6 0%, #e9ecef 100%) !important;
            color: var(--text-primary) !important;
            box-shadow: var(--shadow-md) !important;
            border: 1px solid var(--border-color-light) !important;
        }
        
        .prediction-success {
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%) !important;
            color: #155724 !important;
            border: 1px solid #c3e6cb !important;
        }
        
        .prediction-warning {
            background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%) !important;
            color: #856404 !important;
            border: 1px solid #ffeaa7 !important;
        }
        
        .welcome-section {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color) !important;
        }
        
        .feature-card {
            box-shadow: var(--shadow-lg) !important;
            border: 1px solid var(--border-color-light) !important;
        }
        
        .capability-card-green {
            background: linear-gradient(135deg, #e8f5e8 0%, #d4edda 100%) !important;
            color: var(--text-primary) !important;
            border: 1px solid #c3e6cb !important;
        }
        
        .capability-card-blue {
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%) !important;
            color: var(--text-primary) !important;
            border: 1px solid #90caf9 !important;
        }
        
        .workflow-card {
            background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%) !important;
            border: 1px solid var(--border-color) !important;
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
        background: linear-gradient(45deg, var(--primary-color), var(--info-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: headerGlow 3s ease-in-out infinite alternate;
    }
    
    @keyframes headerGlow {
        from { filter: brightness(1); }
        to { filter: brightness(1.2); }
    }
    
    .metric-card {
        padding: 1.5rem;
        border-radius: var(--border-radius-lg);
        border-left: 4px solid var(--primary-color);
        transition: var(--transition-normal);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 30%, rgba(255, 255, 255, 0.1) 50%, transparent 70%);
        transform: translateX(-100%);
        transition: transform 0.6s;
    }
    
    .metric-card:hover::before {
        transform: translateX(100%);
    }
    
    .metric-card:hover {
        transform: translateY(-4px) scale(1.02);
        box-shadow: var(--shadow-xl) !important;
        border-left-color: var(--info-color);
    }
    
    .prediction-success {
        padding: 1.25rem;
        border-radius: var(--border-radius-md);
        border-left: 4px solid var(--success-color);
        font-weight: 500;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        position: relative;
        overflow: hidden;
        transition: var(--transition-normal);
    }
    
    .prediction-warning {
        padding: 1.25rem;
        border-radius: var(--border-radius-md);
        border-left: 4px solid var(--warning-color);
        font-weight: 500;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        position: relative;
        overflow: hidden;
        transition: var(--transition-normal);
    }
    
    .welcome-section {
        text-align: center;
        padding: 2.5rem;
        border-radius: var(--border-radius-xl);
        margin: 2rem 0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        position: relative;
        overflow: hidden;
    }
    
    .feature-card {
        text-align: center;
        padding: 2rem;
        border-radius: var(--border-radius-lg);
        color: white;
        margin-bottom: 1rem;
        transition: var(--transition-normal);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        position: relative;
        overflow: hidden;
    }
    
    .feature-card:hover {
        transform: translateY(-8px) scale(1.03);
        box-shadow: var(--shadow-xl) !important;
    }
    
    .capability-card-green, .capability-card-blue {
        padding: 1.75rem;
        border-radius: var(--border-radius-lg);
        border-left: 4px solid;
        transition: var(--transition-normal);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        position: relative;
        overflow: hidden;
    }
    
    .capability-card-green:hover, .capability-card-blue:hover {
        transform: translateX(8px) scale(1.01);
    }
    
    .workflow-card {
        text-align: center;
        padding: 1.75rem;
        border-radius: var(--border-radius-lg);
        margin: 0.75rem 0;
        transition: var(--transition-normal);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        position: relative;
        overflow: hidden;
    }
    
    .workflow-card:hover {
        transform: translateY(-4px) scale(1.01);
    }
    
    /* Enhanced Streamlit Widget Styling */
    
    /* Text Input Fields */
    .stTextInput > div > div > input {
        background-color: var(--input-bg) !important;
        border: 2px solid var(--input-border) !important;
        border-radius: var(--border-radius-md) !important;
        color: var(--text-primary) !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
        font-size: 14px !important;
        padding: 12px 16px !important;
        transition: var(--transition-normal) !important;
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.1) !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--input-focus) !important;
        box-shadow: 0 0 0 3px rgba(13, 110, 253, 0.25) !important;
        outline: none !important;
        transform: translateY(-1px) !important;
    }
    
    .stTextInput > div > div > input:hover:not(:focus) {
        border-color: var(--border-color-light) !important;
        transform: translateY(-1px) !important;
    }
    
    /* Number Input Fields */
    .stNumberInput > div > div > input {
        background-color: var(--input-bg) !important;
        border: 2px solid var(--input-border) !important;
        border-radius: var(--border-radius-md) !important;
        color: var(--text-primary) !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
        font-size: 14px !important;
        padding: 12px 16px !important;
        transition: var(--transition-normal) !important;
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.1) !important;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: var(--input-focus) !important;
        box-shadow: 0 0 0 3px rgba(13, 110, 253, 0.25) !important;
        outline: none !important;
        transform: translateY(-1px) !important;
    }
    
    /* Select Boxes */
    .stSelectbox > div > div {
        background-color: var(--input-bg) !important;
        border: 2px solid var(--input-border) !important;
        border-radius: var(--border-radius-md) !important;
        transition: var(--transition-normal) !important;
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.1) !important;
    }
    
    .stSelectbox > div > div:hover {
        border-color: var(--border-color-light) !important;
        transform: translateY(-1px) !important;
    }
    
    .stSelectbox > div > div > div {
        color: var(--text-primary) !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
        padding: 12px 16px !important;
    }
    
    /* Multi-Select */
    .stMultiSelect > div > div {
        background-color: var(--input-bg) !important;
        border: 2px solid var(--input-border) !important;
        border-radius: var(--border-radius-md) !important;
        transition: var(--transition-normal) !important;
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.1) !important;
    }
    
    .stMultiSelect > div > div:hover {
        border-color: var(--border-color-light) !important;
        transform: translateY(-1px) !important;
    }
    
    /* Sliders */
    .stSlider > div > div > div > div {
        background-color: var(--input-bg) !important;
    }
    
    .stSlider .thumb {
        background-color: var(--primary-color) !important;
        border: 3px solid var(--input-bg) !important;
        box-shadow: var(--shadow-md) !important;
        transition: var(--transition-normal) !important;
    }
    
    .stSlider .thumb:hover {
        transform: scale(1.1) !important;
        box-shadow: var(--shadow-lg) !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--info-color) 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--border-radius-md) !important;
        font-weight: 600 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
        padding: 12px 24px !important;
        transition: var(--transition-normal) !important;
        box-shadow: var(--shadow-md) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.02) !important;
        box-shadow: var(--shadow-lg) !important;
        filter: brightness(1.1) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0) scale(0.98) !important;
    }
    
    /* File Uploader */
    .stFileUploader > div > div {
        background-color: var(--input-bg) !important;
        border: 2px dashed var(--input-border) !important;
        border-radius: var(--border-radius-lg) !important;
        transition: var(--transition-normal) !important;
        padding: 2rem !important;
    }
    
    .stFileUploader > div > div:hover {
        border-color: var(--primary-color) !important;
        background-color: var(--hover-bg) !important;
        transform: scale(1.01) !important;
    }
    
    /* Checkbox */
    .stCheckbox > label {
        color: var(--text-primary) !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    }
    
    /* Radio */
    .stRadio > label {
        color: var(--text-primary) !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    }
    
    /* Text Area */
    .stTextArea > div > div > textarea {
        background-color: var(--input-bg) !important;
        border: 2px solid var(--input-border) !important;
        border-radius: var(--border-radius-md) !important;
        color: var(--text-primary) !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
        font-size: 14px !important;
        padding: 12px 16px !important;
        transition: var(--transition-normal) !important;
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.1) !important;
        resize: vertical !important;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: var(--input-focus) !important;
        box-shadow: 0 0 0 3px rgba(13, 110, 253, 0.25) !important;
        outline: none !important;
        transform: translateY(-1px) !important;
    }
    
    /* Date Input */
    .stDateInput > div > div > input {
        background-color: var(--input-bg) !important;
        border: 2px solid var(--input-border) !important;
        border-radius: var(--border-radius-md) !important;
        color: var(--text-primary) !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
        padding: 12px 16px !important;
        transition: var(--transition-normal) !important;
    }
    
    /* Time Input */
    .stTimeInput > div > div > input {
        background-color: var(--input-bg) !important;
        border: 2px solid var(--input-border) !important;
        border-radius: var(--border-radius-md) !important;
        color: var(--text-primary) !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
        padding: 12px 16px !important;
        transition: var(--transition-normal) !important;
    }
    
    /* Labels */
    .stSelectbox label, .stSlider label, .stNumberInput label, 
    .stMultiSelect label, .stCheckbox label, .stRadio label,
    .stTextInput label, .stTextArea label, .stFileUploader label,
    .stDateInput label, .stTimeInput label {
        color: var(--text-primary) !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        margin-bottom: 8px !important;
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
        
        /* Dark theme input overrides */
        .stTextInput > div > div > input:focus,
        .stNumberInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus {
            border-color: var(--primary-color-dark) !important;
            box-shadow: 0 0 0 3px rgba(100, 181, 246, 0.25) !important;
        }
        
        .stButton > button {
            background: linear-gradient(135deg, var(--primary-color-dark) 0%, var(--info-color-dark) 100%) !important;
        }
        
        /* Dark theme navigation */
        .stSidebar {
            background-color: var(--bg-secondary) !important;
        }
        
        .stSidebar .sidebar-content {
            background-color: var(--bg-secondary) !important;
        }
        
        /* Dark theme for dropdowns */
        .stSelectbox div[data-baseweb="select"] div {
            background-color: var(--input-bg) !important;
            color: var(--text-primary) !important;
        }
        
        /* Dark theme for plotly charts */
        .js-plotly-plot {
            background-color: var(--bg-secondary) !important;
        }
    }
    
    /* Light theme for additional classes */
    @media (prefers-color-scheme: light) {
        .info-section {
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%) !important;
            color: var(--text-primary) !important;
            border: 1px solid #90caf9 !important;
        }
        
        .warning-section {
            background: linear-gradient(135deg, #fff3cd 0%, #ffe082 100%) !important;
            color: var(--text-primary) !important;
            border: 1px solid #ffcc02 !important;
        }
        
        .success-section {
            background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c8 100%) !important;
            color: var(--text-primary) !important;
            border: 1px solid #81c784 !important;
        }
        
        .dataset-info-card {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color) !important;
        }
        
        /* Light theme input overrides */
        .stTextInput > div > div > input:focus,
        .stNumberInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus {
            border-color: var(--primary-color) !important;
            box-shadow: 0 0 0 3px rgba(13, 110, 253, 0.25) !important;
        }
        
        .stButton > button {
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--info-color) 100%) !important;
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
        
        /* Enhanced dark mode text color fixes */
        .stSelectbox div[data-baseweb="select"] > div {
            color: #fafafa !important;
        }
        
        .stSelectbox div[data-baseweb="select"] span {
            color: #fafafa !important;
        }
        
        .stSelectbox div[data-baseweb="select"] div[role="option"] {
            color: #fafafa !important;
            background-color: var(--bg-secondary) !important;
        }
        
        .stSelectbox div[data-baseweb="select"] div[role="option"]:hover {
            background-color: var(--hover-bg) !important;
            color: #ffffff !important;
        }
        
        /* Fix for all text elements */
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
            color: #fafafa !important;
        }
        
        .stMarkdown p, .stMarkdown span, .stMarkdown div {
            color: #fafafa !important;
        }
        
        /* Fix for selectbox dropdown */
        .stSelectbox > div > div > div {
            color: #fafafa !important;
        }
        
        /* Fix for selectbox placeholder and selected value */
        .stSelectbox > div > div > div > div {
            color: #fafafa !important;
        }
        
        /* Fix for all form labels */
        label, .stSelectbox label, .stTextInput label, .stNumberInput label,
        .stTextArea label, .stSlider label, .stMultiSelect label,
        .stCheckbox label, .stRadio label, .stFileUploader label {
            color: #fafafa !important;
        }
        
        /* Fix for help text */
        .stSelectbox small, .stTextInput small, .stNumberInput small,
        .stTextArea small, .stSlider small, .stMultiSelect small {
            color: #c9c9c9 !important;
        }
        
        /* Fix for caption text */
        .stCaption, .caption {
            color: #c9c9c9 !important;
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
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        
        .welcome-section {
            padding: 1.5rem;
        }
        
        .metric-card, .feature-card, .capability-card-green, .capability-card-blue, .workflow-card {
            padding: 1rem;
        }
        
        .stButton > button {
            padding: 10px 20px !important;
        }
    }
    
    @media (max-width: 480px) {
        .main-header {
            font-size: 1.5rem;
        }
        
        .welcome-section {
            padding: 1rem;
        }
        
        .metric-card, .feature-card, .capability-card-green, .capability-card-blue, .workflow-card {
            padding: 0.75rem;
        }
    }
    
    /* Loading Animation */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .loading {
        animation: pulse 1.5s ease-in-out infinite;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
        border-radius: 6px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border-color);
        border-radius: 6px;
        transition: var(--transition-normal);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-color);
    }
    
    /* Focus Management */
    *:focus {
        outline: 2px solid var(--primary-color) !important;
        outline-offset: 2px !important;
    }
    
    /* Print Styles */
    @media print {
        .stButton, .stSelectbox, .stSlider, .stFileUploader {
            display: none !important;
        }
        
        .main-header, .metric-card, .welcome-section {
            background: white !important;
            color: black !important;
            box-shadow: none !important;
        }
    }
    
    /* High Contrast Mode Support */
    @media (prefers-contrast: high) {
        :root {
            --border-color: #000000 !important;
            --shadow-color: rgba(0, 0, 0, 0.5) !important;
        }
        
        .metric-card, .feature-card, .capability-card-green, .capability-card-blue {
            border: 2px solid var(--border-color) !important;
        }
    }
    
    /* Reduced Motion Support */
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }
        
        .main-header {
            animation: none !important;
        }
        
        .metric-card:hover, .feature-card:hover, .workflow-card:hover {
            transform: none !important;
        }
    }
    
    /* Additional Streamlit Element Fixes */
    .stAlert {
        border-radius: var(--border-radius-md) !important;
        border: 1px solid var(--border-color) !important;
    }
    
    .stExpander {
        border: 1px solid var(--border-color) !important;
        border-radius: var(--border-radius-md) !important;
        background-color: var(--bg-secondary) !important;
    }
    
    .stProgress .stProgress-bar {
        background-color: var(--primary-color) !important;
        border-radius: var(--border-radius-sm) !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background-color: var(--bg-secondary) !important;
        border-radius: var(--border-radius-md) !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: var(--text-primary) !important;
        border-radius: var(--border-radius-sm) !important;
    }
    
    .stDataFrame {
        border: 1px solid var(--border-color) !important;
        border-radius: var(--border-radius-md) !important;
        background-color: var(--bg-primary) !important;
    }
    
    /* Ensure proper color inheritance */
    .stMarkdown, .stText, .stCaption {
        color: var(--text-primary) !important;
    }
    
    .stCode {
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: var(--border-radius-sm) !important;
    }
    
    /* Dark theme attribute-based styling */
    [data-theme="dark"] .stMarkdown,
    [data-theme="dark"] .stText,
    [data-theme="dark"] .stSelectbox label,
    [data-theme="dark"] .stTextInput label,
    [data-theme="dark"] .stNumberInput label,
    [data-theme="dark"] .stTextArea label,
    [data-theme="dark"] .stSlider label,
    [data-theme="dark"] .stMultiSelect label,
    [data-theme="dark"] .stCheckbox label,
    [data-theme="dark"] .stRadio label,
    [data-theme="dark"] .stFileUploader label {
        color: #fafafa !important;
    }
    
    [data-theme="dark"] .stMarkdown h1,
    [data-theme="dark"] .stMarkdown h2,
    [data-theme="dark"] .stMarkdown h3,
    [data-theme="dark"] .stMarkdown h4,
    [data-theme="dark"] .stMarkdown h5,
    [data-theme="dark"] .stMarkdown h6,
    [data-theme="dark"] .stMarkdown p,
    [data-theme="dark"] .stMarkdown span,
    [data-theme="dark"] .stMarkdown div {
        color: #fafafa !important;
    }
    
    [data-theme="dark"] .stSelectbox > div > div > div {
        color: #fafafa !important;
    }
    
    [data-theme="dark"] .stSelectbox div[data-baseweb="select"] > div,
    [data-theme="dark"] .stSelectbox div[data-baseweb="select"] span {
        color: #fafafa !important;
    }
    
    /* Light theme attribute-based styling */
    [data-theme="light"] .stMarkdown,
    [data-theme="light"] .stText,
    [data-theme="light"] .stSelectbox label,
    [data-theme="light"] .stTextInput label,
    [data-theme="light"] .stNumberInput label,
    [data-theme="light"] .stTextArea label,
    [data-theme="light"] .stSlider label,
    [data-theme="light"] .stMultiSelect label,
    [data-theme="light"] .stCheckbox label,
    [data-theme="light"] .stRadio label,
    [data-theme="light"] .stFileUploader label {
        color: #212529 !important;
    }
    
    [data-theme="light"] .stSelectbox > div > div > div {
        color: #212529 !important;
    }
    
    /* Theme transition animation */
    html, body, .stApp, [data-testid="stAppViewContainer"] {
        transition: background-color 0.3s ease, color 0.3s ease !important;
    }
    
    /* Success indicator for theme switching */
    .theme-indicator {
        position: fixed;
        top: 10px;
        right: 10px;
        background: var(--success-color);
        color: white;
        padding: 8px 12px;
        border-radius: var(--border-radius-md);
        font-size: 12px;
        font-weight: 500;
        z-index: 10000;
        opacity: 0;
        transform: translateY(-20px);
        transition: all 0.3s ease;
        pointer-events: none;
    }
    
    .theme-indicator.show {
        opacity: 1;
        transform: translateY(0);
    }
</style>

<script>
// Enhanced comprehensive theme detection and switching script
(function() {
    let currentTheme = 'light';
    let themeCheckInterval;
    
    function detectAndApplyTheme() {
        // Multiple ways to detect dark mode
        const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
        
        // Check Streamlit's background color
        const bodyBg = getComputedStyle(document.body).backgroundColor;
        const containerBg = getComputedStyle(document.documentElement).getPropertyValue('--background-color');
        
        // Check for Streamlit's main container
        const mainContainer = document.querySelector('[data-testid="stAppViewContainer"]');
        const mainContainerBg = mainContainer ? getComputedStyle(mainContainer).backgroundColor : '';
        
        // Check sidebar for theme
        const sidebar = document.querySelector('.css-1d391kg') || document.querySelector('[data-testid="stSidebar"]');
        const sidebarBg = sidebar ? getComputedStyle(sidebar).backgroundColor : '';
        
        // Streamlit dark theme detection (more comprehensive)
        const darkIndicators = [
            bodyBg.includes('rgb(14, 17, 23)') || bodyBg.includes('#0e1117'),
            containerBg.includes('rgb(14, 17, 23)') || containerBg.includes('#0e1117'),
            mainContainerBg.includes('rgb(14, 17, 23)') || mainContainerBg.includes('#0e1117'),
            sidebarBg.includes('rgb(38, 39, 48)') || sidebarBg.includes('#262730'),
            document.documentElement.style.backgroundColor?.includes('#0e1117'),
            document.body.style.backgroundColor?.includes('#0e1117')
        ];
        
        const isStreamlitDark = darkIndicators.some(indicator => indicator);
        
        // Check for dark theme class indicators
        const hasStreamlitDarkClass = document.querySelector('[data-testid="stAppViewContainer"]')?.style.backgroundColor.includes('rgb(14, 17, 23)');
        
        // Final determination
        const isDark = prefersDark || isStreamlitDark || hasStreamlitDarkClass;
        const newTheme = isDark ? 'dark' : 'light';
        
        // Only update if theme changed
        if (newTheme !== currentTheme) {
            currentTheme = newTheme;
            applyTheme(isDark);
            console.log('🎨 Theme switched to:', newTheme);
        }
    }
    
    function applyTheme(isDark) {
        // Apply theme attribute
        document.documentElement.setAttribute('data-theme', isDark ? 'dark' : 'light');
        document.body.setAttribute('data-theme', isDark ? 'dark' : 'light');
        
        // Force update CSS variables
        const root = document.documentElement;
        
        if (isDark) {
            // Dark theme variables
            root.style.setProperty('--text-primary', '#fafafa');
            root.style.setProperty('--text-secondary', '#c9c9c9');
            root.style.setProperty('--text-tertiary', '#8b949e');
            root.style.setProperty('--bg-primary', '#0e1117');
            root.style.setProperty('--bg-secondary', '#262730');
            root.style.setProperty('--bg-tertiary', '#1a1d29');
            root.style.setProperty('--bg-quaternary', '#2f3349');
            root.style.setProperty('--border-color', '#404040');
            root.style.setProperty('--border-color-light', '#525562');
            root.style.setProperty('--shadow-color', 'rgba(0, 0, 0, 0.4)');
            root.style.setProperty('--card-bg', '#1e1e1e');
            root.style.setProperty('--hover-bg', '#2a2a2a');
            root.style.setProperty('--input-bg', '#21262d');
            root.style.setProperty('--input-border', '#484f58');
            root.style.setProperty('--input-focus', '#0969da');
            root.style.setProperty('--overlay-bg', 'rgba(0, 0, 0, 0.8)');
            root.style.setProperty('--primary-color', '#64b5f6');
            root.style.setProperty('--success-color', '#4caf50');
            root.style.setProperty('--warning-color', '#ffeb3b');
            root.style.setProperty('--danger-color', '#ff6b6b');
            root.style.setProperty('--info-color', '#29b6f6');
            
            // Force update text colors for selectbox and other elements
            setTimeout(() => {
                const selectboxElements = document.querySelectorAll('.stSelectbox > div > div > div');
                selectboxElements.forEach(el => {
                    el.style.color = '#fafafa';
                });
                
                const selectboxLabels = document.querySelectorAll('.stSelectbox label');
                selectboxLabels.forEach(el => {
                    el.style.color = '#fafafa';
                });
                
                const markdownElements = document.querySelectorAll('.stMarkdown');
                markdownElements.forEach(el => {
                    el.style.color = '#fafafa';
                });
                
                const allLabels = document.querySelectorAll('label');
                allLabels.forEach(el => {
                    el.style.color = '#fafafa';
                });
            }, 100);
            
        } else {
            // Light theme variables
            root.style.setProperty('--text-primary', '#212529');
            root.style.setProperty('--text-secondary', '#6c757d');
            root.style.setProperty('--text-tertiary', '#868e96');
            root.style.setProperty('--bg-primary', '#ffffff');
            root.style.setProperty('--bg-secondary', '#f8f9fa');
            root.style.setProperty('--bg-tertiary', '#e9ecef');
            root.style.setProperty('--bg-quaternary', '#dee2e6');
            root.style.setProperty('--border-color', '#dee2e6');
            root.style.setProperty('--border-color-light', '#e9ecef');
            root.style.setProperty('--shadow-color', 'rgba(0, 0, 0, 0.1)');
            root.style.setProperty('--card-bg', '#ffffff');
            root.style.setProperty('--hover-bg', '#f8f9fa');
            root.style.setProperty('--input-bg', '#ffffff');
            root.style.setProperty('--input-border', '#ced4da');
            root.style.setProperty('--input-focus', '#0d6efd');
            root.style.setProperty('--overlay-bg', 'rgba(0, 0, 0, 0.5)');
            root.style.setProperty('--primary-color', '#1f77b4');
            root.style.setProperty('--success-color', '#28a745');
            root.style.setProperty('--warning-color', '#ffc107');
            root.style.setProperty('--danger-color', '#dc3545');
            root.style.setProperty('--info-color', '#17a2b8');
            
            // Force update text colors for light theme
            setTimeout(() => {
                const selectboxElements = document.querySelectorAll('.stSelectbox > div > div > div');
                selectboxElements.forEach(el => {
                    el.style.color = '#212529';
                });
                
                const selectboxLabels = document.querySelectorAll('.stSelectbox label');
                selectboxLabels.forEach(el => {
                    el.style.color = '#212529';
                });
                
                const markdownElements = document.querySelectorAll('.stMarkdown');
                markdownElements.forEach(el => {
                    el.style.color = '#212529';
                });
                
                const allLabels = document.querySelectorAll('label');
                allLabels.forEach(el => {
                    el.style.color = '#212529';
                });
            }, 100);
        }
        
        // Trigger a custom event for theme change
        window.dispatchEvent(new CustomEvent('themeChanged', { 
            detail: { theme: isDark ? 'dark' : 'light' } 
        }));
        
        // Force re-render of dynamic elements
        requestAnimationFrame(() => {
            document.body.style.visibility = 'hidden';
            document.body.offsetHeight; // Trigger reflow
            document.body.style.visibility = 'visible';
        });
    }
    
    function startThemeMonitoring() {
        // Initial detection
        detectAndApplyTheme();
        
        // Listen for system theme changes
        if (window.matchMedia) {
            const darkModeQuery = window.matchMedia('(prefers-color-scheme: dark)');
            darkModeQuery.addListener(detectAndApplyTheme);
        }
        
        // Monitor DOM changes for Streamlit's dynamic updates
        const observer = new MutationObserver(function(mutations) {
            let shouldRecheck = false;
            mutations.forEach(function(mutation) {
                if (mutation.type === 'attributes' && 
                    (mutation.attributeName === 'style' || mutation.attributeName === 'class')) {
                    shouldRecheck = true;
                } else if (mutation.type === 'childList') {
                    // Check if new Streamlit elements were added
                    for (let addedNode of mutation.addedNodes) {
                        if (addedNode.nodeType === 1 && // Element node
                            (addedNode.querySelector && addedNode.querySelector('[data-testid]'))) {
                            shouldRecheck = true;
                            break;
                        }
                    }
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
            attributeFilter: ['style', 'class', 'data-theme']
        });
        
        // Also check on window focus and visibility change
        window.addEventListener('focus', detectAndApplyTheme);
        window.addEventListener('visibilitychange', detectAndApplyTheme);
        
        // Periodic check as fallback (reduced frequency)
        themeCheckInterval = setInterval(detectAndApplyTheme, 2000);
        
        console.log('🚀 Enhanced theme monitoring started');
    }
    
    // Add theme toggle functionality (optional)
    function createThemeToggle() {
        const toggleButton = document.createElement('button');
        toggleButton.innerHTML = '🌓';
        toggleButton.title = 'Toggle Theme';
        toggleButton.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 9999;
            background: var(--bg-secondary);
            border: 2px solid var(--border-color);
            border-radius: 50%;
            width: 50px;
            height: 50px;
            font-size: 20px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: var(--shadow-md);
        `;
        
        toggleButton.addEventListener('click', function() {
            const isDark = currentTheme === 'dark';
            applyTheme(!isDark);
            currentTheme = isDark ? 'light' : 'dark';
        });
        
        toggleButton.addEventListener('mouseenter', function() {
            this.style.transform = 'scale(1.1) rotate(180deg)';
        });
        
        toggleButton.addEventListener('mouseleave', function() {
            this.style.transform = 'scale(1) rotate(0deg)';
        });
        
        document.body.appendChild(toggleButton);
    }
    
    // Initialize everything when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() {
            startThemeMonitoring();
            // Uncomment the next line to add theme toggle button
            // createThemeToggle();
        });
    } else {
        startThemeMonitoring();
        // Uncomment the next line to add theme toggle button
        // createThemeToggle();
    }
    
    // Cleanup on page unload
    window.addEventListener('beforeunload', function() {
        if (themeCheckInterval) {
            clearInterval(themeCheckInterval);
        }
    });
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
st.markdown('<h2 style="color: #1f77b4;">📊 Step 1: Model Evaluation Dashboard</h2>', unsafe_allow_html=True)
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

# Add train2.csv if it exists
train2_file = "data/train2.csv"
if os.path.exists(train2_file):
    data_options.append("Extended Training Data (train2.csv)")

for exp_file in sorted(experiment_files):
    filename = os.path.basename(exp_file)
    data_options.append(f"Experiment Data ({filename})")

prediction_selected_option = st.selectbox("Choose dataset for prediction:", data_options, key="prediction_data")

if prediction_selected_option:
    # Determine which file to load
    if "train.csv" in prediction_selected_option:
        prediction_selected_file = train_file
        prediction_data_type = "training"
    elif "train2.csv" in prediction_selected_option:
        prediction_selected_file = train2_file
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

# Completion message
st.markdown("---")
st.markdown('<h2 style="color: #28a745;">🎉 Pipeline Complete!</h2>', unsafe_allow_html=True)
st.success("You have completed all steps of the Machine Learning Pipeline!")
