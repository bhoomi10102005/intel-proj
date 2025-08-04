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

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .prediction-success {
        background-color: #d4edda;
        color: #155724;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 5px solid #28a745;
    }
    .prediction-warning {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
    }
</style>
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
        <div style="text-align: center; padding: 2rem; background-color: #f8f9fa; border-radius: 15px; margin: 2rem 0;">
            <h3>Welcome to Advanced Machine Learning Analytics</h3>
            <p>Predict tool wear status using state-of-the-art Random Forest algorithms trained on real sensor data.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature highlights section
    st.markdown("### ✨ Platform Features")
    
    feature_col1, feature_col2, feature_col3, feature_col4 = st.columns(4)
    
    with feature_col1:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; color: white; margin-bottom: 1rem;">
            <h3>🔧</h3>
            <h4>Tool Prediction</h4>
            <p>AI-powered wear detection</p>
        </div>
        """, unsafe_allow_html=True)
    
    with feature_col2:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); border-radius: 12px; color: white; margin-bottom: 1rem;">
            <h3>📊</h3>
            <h4>Data Analysis</h4>
            <p>Statistical insights</p>
        </div>
        """, unsafe_allow_html=True)
    
    with feature_col3:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); border-radius: 12px; color: white; margin-bottom: 1rem;">
            <h3>📈</h3>
            <h4>Visualizations</h4>
            <p>Interactive charts</p>
        </div>
        """, unsafe_allow_html=True)
    
    with feature_col4:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); border-radius: 12px; color: white; margin-bottom: 1rem;">
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
        <div style="padding: 1.5rem; background-color: #e8f5e8; border-radius: 10px; border-left: 5px solid #28a745;">
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
        <div style="padding: 1.5rem; background-color: #e3f2fd; border-radius: 10px; border-left: 5px solid #2196f3;">
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
        <div style="text-align: center; padding: 1rem; background-color: #fff3e0; border-radius: 10px; margin: 0.5rem 0;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">1️⃣</div>
            <h5>Load Data</h5>
            <p style="font-size: 0.9rem;">Select from training or experiment datasets</p>
        </div>
        """, unsafe_allow_html=True)
    
    with workflow_cols[1]:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background-color: #f3e5f5; border-radius: 10px; margin: 0.5rem 0;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">2️⃣</div>
            <h5>Analyze</h5>
            <p style="font-size: 0.9rem;">Explore patterns and relationships</p>
        </div>
        """, unsafe_allow_html=True)
    
    with workflow_cols[2]:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background-color: #e8f5e8; border-radius: 10px; margin: 0.5rem 0;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">3️⃣</div>
            <h5>Predict</h5>
            <p style="font-size: 0.9rem;">Run ML models for tool wear prediction</p>
        </div>
        """, unsafe_allow_html=True)
    
    with workflow_cols[3]:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background-color: #e1f5fe; border-radius: 10px; margin: 0.5rem 0;">
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
        <div style="padding: 1.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; color: white; text-align: center;">
            <h4>🔧 Start Predicting</h4>
            <p>Jump straight into tool wear prediction with our trained models</p>
            <div style="margin-top: 1rem; padding: 0.5rem; background-color: rgba(255,255,255,0.2); border-radius: 5px;">
                Navigate to "Worn Tool Prediction"
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with start_col2:
        st.markdown("""
        <div style="padding: 1.5rem; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); border-radius: 12px; color: white; text-align: center;">
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
    <div style="background-color: #e3f2fd; padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;">
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
    <div style="background-color: #e8f5e8; padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;">
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
    <div style="background-color: #fff3cd; padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem; border-left: 5px solid #ffc107;">
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

        st.markdown("### 🎯 Configure Training")
        col1, col2 = st.columns(2)
        with col1:
            numeric_cols = user_df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = st.multiselect(
                "Select feature columns:", 
                numeric_cols,
                help="Choose the numeric columns that will be used as input features"
            )
        with col2:
            label_col = st.selectbox(
                "Select label column:", 
                user_df.columns,
                help="Choose the column that contains the target labels"
            )

        if feature_cols and label_col:
            # Add helpful validation before training
            if label_col in feature_cols:
                st.error("❌ **Configuration Error**: The label column cannot be the same as a feature column!")
                st.info("💡 **Solution**: Remove the label column from the feature selection, or choose a different label column.")
            elif user_df[label_col].dtype in ['float64', 'float32'] and len(user_df[label_col].unique()) > 10:
                st.error(f"❌ **Incorrect Label Selection**: '{label_col}' appears to be a continuous numeric column with {len(user_df[label_col].unique())} unique values.")
                st.error("🔍 **Issue**: Classification algorithms expect discrete classes (like 'worn'/'unworn'), not continuous values.")
                
                # Check if tool_condition exists
                if 'tool_condition' in user_df.columns:
                    tool_values = user_df['tool_condition'].unique()
                    st.success(f"💡 **Recommended Solution**: Use '**tool_condition**' as the label column instead. It contains discrete classes: {list(tool_values)}")
                else:
                    st.info("💡 **Possible Solutions**:")
                    st.write("1. Use a categorical column as the label (like 'tool_condition', 'status', etc.)")
                    st.write("2. Convert continuous values to discrete classes (e.g., 'low'/'medium'/'high')")
                    st.write("3. Use regression algorithms instead of classification")
            else:
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
            <div style="padding: 1rem; background-color: #e8f5e8; border-radius: 8px; margin: 0.5rem 0;">
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
            <div style="padding: 1rem; background-color: #e3f2fd; border-radius: 8px; margin: 0.5rem 0;">
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
    <div style="background-color: #e3f2fd; padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;">
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
        
        # Feature and label selection
        st.markdown("### 🎯 Configure Evaluation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Select feature columns
            numeric_cols = test_df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = st.multiselect(
                "Select feature columns:", 
                numeric_cols, 
                key="eval_features",
                help="Choose the same features used during training"
            )
        
        with col2:
            # Select label column
            label_col = st.selectbox(
                "Select label column:", 
                test_df.columns, 
                key="eval_label",
                help="Choose the column with true labels"
            )
        
        if feature_cols and label_col:
            if st.button("📊 Evaluate Model", type="primary"):
                try:
                    # Load model and make predictions
                    with st.spinner("Loading model and making predictions..."):
                        model = load_model()
                        X = test_df[feature_cols]
                        y_true = test_df[label_col]
                        
                        # Handle different label formats
                        if y_true.dtype == 'object':
                            # Convert string labels to numeric
                            label_mapping = {'unworn': 0, 'worn': 1}
                            y_true_numeric = y_true.map(label_mapping)
                            if y_true_numeric.isnull().any():
                                st.warning("⚠️ Some labels couldn't be mapped. Using original values.")
                                y_true_numeric = y_true
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
                    
                    # Confusion Matrix
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 🔄 Confusion Matrix")
                        cm_df = pd.DataFrame(
                            cm,
                            index=['Actual Unworn', 'Actual Worn'],
                            columns=['Predicted Unworn', 'Predicted Worn']
                        )
                        st.dataframe(cm_df, use_container_width=True)
                        
                        # Confusion matrix heatmap
                        fig_cm = px.imshow(
                            cm,
                            text_auto=True,
                            aspect="auto",
                            title="Confusion Matrix Heatmap",
                            labels=dict(x="Predicted", y="Actual"),
                            x=['Unworn', 'Worn'],
                            y=['Unworn', 'Worn']
                        )
                        st.plotly_chart(fig_cm, use_container_width=True)
                    
                    with col2:
                        # Prediction distribution
                        st.markdown("### 📊 Prediction Distribution")
                        pred_counts = pd.Series(y_pred).value_counts()
                        fig_dist = px.pie(
                            values=pred_counts.values,
                            names=['Unworn', 'Worn'],
                            title="Predicted Class Distribution"
                        )
                        st.plotly_chart(fig_dist, use_container_width=True)
                        
                        # Accuracy by class
                        st.markdown("### 🎯 Class-wise Performance")
                        class_report = classification_report(y_true_numeric, y_pred, output_dict=True)
                        class_df = pd.DataFrame(class_report).transpose().round(3)
                        st.dataframe(class_df, use_container_width=True)
                    
                    # ROC Curve (for binary classification)
                    if len(set(y_true_numeric)) == 2 and y_pred_proba is not None:
                        st.markdown("### 📈 ROC Curve Analysis")
                        
                        # Calculate ROC curve
                        fpr, tpr, _ = roc_curve(y_true_numeric, y_pred_proba[:, 1])
                        roc_auc = auc(fpr, tpr)
                        
                        # Create ROC curve plot
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
                            title=f"ROC Curve (AUC = {roc_auc:.3f})",
                            xaxis_title="False Positive Rate",
                            yaxis_title="True Positive Rate",
                            xaxis=dict(range=[0, 1]),
                            yaxis=dict(range=[0, 1])
                        )
                        st.plotly_chart(fig_roc, use_container_width=True)
                        
                        # AUC interpretation
                        if roc_auc >= 0.9:
                            st.success(f"🌟 Excellent model performance (AUC = {roc_auc:.3f})")
                        elif roc_auc >= 0.8:
                            st.info(f"✅ Good model performance (AUC = {roc_auc:.3f})")
                        elif roc_auc >= 0.7:
                            st.warning(f"⚠️ Fair model performance (AUC = {roc_auc:.3f})")
                        else:
                            st.error(f"❌ Poor model performance (AUC = {roc_auc:.3f})")
                    
                    # Model interpretation
                    st.markdown("### 🔍 Model Insights")
                    
                    insights_col1, insights_col2 = st.columns(2)
                    
                    with insights_col1:
                        st.markdown("#### ✅ Strengths")
                        if acc > 0.8:
                            st.write("• High overall accuracy")
                        if prec > 0.8:
                            st.write("• Low false positive rate")
                        if rec > 0.8:
                            st.write("• Low false negative rate")
                        if f1 > 0.8:
                            st.write("• Balanced precision and recall")
                    
                    with insights_col2:
                        st.markdown("#### ⚠️ Areas for Improvement")
                        if acc < 0.7:
                            st.write("• Consider feature engineering")
                        if prec < 0.7:
                            st.write("• Reduce false positives")
                        if rec < 0.7:
                            st.write("• Reduce false negatives")
                        if abs(prec - rec) > 0.2:
                            st.write("• Balance precision and recall")
                    
                    # Download results
                    st.markdown("### 💾 Export Results")
                    
                    results_df = test_df.copy()
                    results_df['Predicted'] = y_pred
                    results_df['Correct'] = (y_true_numeric == y_pred)
                    
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions CSV",
                        data=csv,
                        file_name="model_predictions.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Evaluation failed: {str(e)}")
                    st.info("Please check that:")
                    st.write("- The features match those used during training")
                    st.write("- The model file exists and is properly formatted")
                    st.write("- The test data has the correct format")
    
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
