"""
Sensor Data Visualizer Module
Provides comprehensive visualization functions for sensor data analysis
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import streamlit as st

class SensorDataVisualizer:
    """
    A class to handle all sensor data visualizations
    """
    
    def __init__(self):
        self.color_palette = {
            'worn': '#dc3545',
            'unworn': '#28a745',
            'primary': '#1f77b4',
            'secondary': '#ff7f0e'
        }
    
    def create_distribution_plots(self, df, feature_col, condition_col='tool_condition'):
        """
        Create distribution plots for a feature across tool conditions
        """
        if condition_col not in df.columns:
            return None
            
        # Box plot
        box_fig = px.box(
            df, 
            x=condition_col, 
            y=feature_col,
            title=f"{feature_col.title()} Distribution by Tool Condition",
            color=condition_col,
            color_discrete_map={'worn': self.color_palette['worn'], 
                              'unworn': self.color_palette['unworn']}
        )
        box_fig.update_layout(
            xaxis_title="Tool Condition",
            yaxis_title=feature_col.title(),
            showlegend=False
        )
        
        # Histogram
        hist_fig = px.histogram(
            df, 
            x=feature_col, 
            color=condition_col,
            title=f"{feature_col.title()} Distribution Histogram",
            marginal="rug",
            color_discrete_map={'worn': self.color_palette['worn'], 
                              'unworn': self.color_palette['unworn']}
        )
        hist_fig.update_layout(
            xaxis_title=feature_col.title(),
            yaxis_title="Count"
        )
        
        return box_fig, hist_fig
    
    def create_scatter_plot(self, df, x_col, y_col, condition_col='tool_condition'):
        """
        Create scatter plot for two features colored by tool condition
        """
        if condition_col not in df.columns:
            condition_col = None
            
        fig = px.scatter(
            df, 
            x=x_col, 
            y=y_col,
            color=condition_col if condition_col else None,
            title=f"{x_col.title()} vs {y_col.title()}",
            color_discrete_map={'worn': self.color_palette['worn'], 
                              'unworn': self.color_palette['unworn']} if condition_col else None,
            hover_data=[condition_col] if condition_col else None
        )
        
        fig.update_layout(
            xaxis_title=x_col.title(),
            yaxis_title=y_col.title()
        )
        
        return fig
    
    def create_line_plot(self, df, x_col, y_col, condition_col='tool_condition'):
        """
        Create line plot for time series or sequential data
        """
        fig = go.Figure()
        
        if condition_col in df.columns:
            for condition in df[condition_col].unique():
                condition_data = df[df[condition_col] == condition]
                fig.add_trace(go.Scatter(
                    x=condition_data[x_col],
                    y=condition_data[y_col],
                    mode='lines+markers',
                    name=f"{condition.title()}",
                    line=dict(color=self.color_palette.get(condition, self.color_palette['primary']))
                ))
        else:
            fig.add_trace(go.Scatter(
                x=df[x_col],
                y=df[y_col],
                mode='lines+markers',
                name=y_col.title(),
                line=dict(color=self.color_palette['primary'])
            ))
        
        fig.update_layout(
            title=f"{y_col.title()} Over {x_col.title()}",
            xaxis_title=x_col.title(),
            yaxis_title=y_col.title()
        )
        
        return fig
    
    def create_bar_plot(self, df, x_col, y_col=None, condition_col='tool_condition'):
        """
        Create bar plot for categorical data
        """
        if y_col is None:
            # Count plot
            if condition_col in df.columns:
                grouped_data = df.groupby([x_col, condition_col]).size().reset_index(name='count')
                fig = px.bar(
                    grouped_data,
                    x=x_col,
                    y='count',
                    color=condition_col,
                    title=f"Count of {x_col.title()} by Tool Condition",
                    color_discrete_map={'worn': self.color_palette['worn'], 
                                      'unworn': self.color_palette['unworn']}
                )
            else:
                value_counts = df[x_col].value_counts()
                fig = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    title=f"Count of {x_col.title()}"
                )
        else:
            # Regular bar plot
            if condition_col in df.columns:
                fig = px.bar(
                    df,
                    x=x_col,
                    y=y_col,
                    color=condition_col,
                    title=f"{y_col.title()} by {x_col.title()}",
                    color_discrete_map={'worn': self.color_palette['worn'], 
                                      'unworn': self.color_palette['unworn']}
                )
            else:
                fig = px.bar(
                    df,
                    x=x_col,
                    y=y_col,
                    title=f"{y_col.title()} by {x_col.title()}"
                )
        
        fig.update_layout(
            xaxis_title=x_col.title(),
            yaxis_title="Count" if y_col is None else y_col.title()
        )
        
        return fig
    
    def create_correlation_heatmap(self, df, features=None):
        """
        Create correlation heatmap for numeric features
        """
        if features is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_cols = [col for col in features if col in df.columns and df[col].dtype in ['int64', 'float64']]
        
        if len(numeric_cols) < 2:
            return None
            
        correlation_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            title="Feature Correlation Heatmap",
            color_continuous_scale="RdBu_r"
        )
        
        fig.update_layout(
            width=800,
            height=600
        )
        
        return fig
    
    def create_multi_feature_comparison(self, df, features, condition_col='tool_condition'):
        """
        Create subplots comparing multiple features
        """
        if condition_col not in df.columns:
            return None
            
        n_features = len(features)
        cols = 2
        rows = (n_features + 1) // 2
        
        fig = make_subplots(
            rows=rows, 
            cols=cols,
            subplot_titles=[f"{feature.title()} Distribution" for feature in features]
        )
        
        for i, feature in enumerate(features):
            row = (i // cols) + 1
            col = (i % cols) + 1
            
            for condition in df[condition_col].unique():
                condition_data = df[df[condition_col] == condition][feature]
                fig.add_trace(
                    go.Box(
                        y=condition_data,
                        name=f"{condition.title()}",
                        marker_color=self.color_palette.get(condition, self.color_palette['primary']),
                        showlegend=(i == 0)  # Only show legend for first subplot
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title="Multi-Feature Comparison by Tool Condition",
            height=300 * rows
        )
        
        return fig
    
    def create_pattern_analysis_chart(self, df, condition_col='tool_condition'):
        """
        Create comprehensive pattern analysis for worn vs unworn tools
        """
        if condition_col not in df.columns:
            return None
            
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            return None
            
        # Calculate means for each condition
        means_data = df.groupby(condition_col)[numeric_cols].mean().reset_index()
        
        # Create radar chart
        fig = go.Figure()
        
        for condition in df[condition_col].unique():
            condition_means = means_data[means_data[condition_col] == condition]
            
            # Normalize values for radar chart (0-1 scale)
            values = []
            for col in numeric_cols[:6]:  # Limit to 6 features for readability
                col_data = df[col]
                normalized_val = (condition_means[col].iloc[0] - col_data.min()) / (col_data.max() - col_data.min())
                values.append(normalized_val)
            
            fig.add_trace(go.Scatterpolar(
                r=values + [values[0]],  # Close the polygon
                theta=numeric_cols[:6] + [numeric_cols[0]],
                fill='toself',
                name=f"{condition.title()} Tools",
                line_color=self.color_palette.get(condition, self.color_palette['primary'])
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            title="Pattern Analysis: Worn vs Unworn Tools",
            showlegend=True
        )
        
        return fig
    
    def get_feature_statistics(self, df, feature_col, condition_col='tool_condition'):
        """
        Get statistical summary for a feature by condition
        """
        if condition_col not in df.columns:
            stats = df[feature_col].describe()
            return pd.DataFrame(stats).round(3)
        
        stats = df.groupby(condition_col)[feature_col].describe().round(3)
        return stats
    
    def identify_outliers(self, df, feature_col, condition_col='tool_condition'):
        """
        Identify outliers in the data using IQR method
        """
        outliers_info = {}
        
        if condition_col in df.columns:
            for condition in df[condition_col].unique():
                condition_data = df[df[condition_col] == condition][feature_col]
                Q1 = condition_data.quantile(0.25)
                Q3 = condition_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = condition_data[(condition_data < lower_bound) | (condition_data > upper_bound)]
                outliers_info[condition] = {
                    'count': len(outliers),
                    'percentage': (len(outliers) / len(condition_data)) * 100,
                    'values': outliers.tolist()
                }
        else:
            Q1 = df[feature_col].quantile(0.25)
            Q3 = df[feature_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[feature_col][(df[feature_col] < lower_bound) | (df[feature_col] > upper_bound)]
            outliers_info['all'] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(df)) * 100,
                'values': outliers.tolist()
            }
        
        return outliers_info
