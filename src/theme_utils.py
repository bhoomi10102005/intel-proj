"""
Theme utilities for automatic dark/light mode detection and styling.
This module provides utilities for managing themes across the application.
"""

import streamlit as st

def inject_theme_css():
    """
    Inject comprehensive theme-aware CSS for dark/light mode support.
    This function provides advanced styling for all Streamlit components.
    """
    
    theme_css = """
    <style>
        /* Theme indicator for user feedback */
        .theme-notification {
            position: fixed;
            top: 20px;
            right: 20px;
            background: linear-gradient(135deg, #4caf50 0%, #81c784 100%);
            color: white;
            padding: 12px 16px;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 500;
            z-index: 10000;
            opacity: 0;
            transform: translateX(100px);
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            pointer-events: none;
        }
        
        .theme-notification.show {
            opacity: 1;
            transform: translateX(0);
        }
        
        /* Enhanced input field styling */
        .custom-input-container {
            position: relative;
            margin-bottom: 1rem;
        }
        
        .custom-input-label {
            position: absolute;
            top: -8px;
            left: 12px;
            background: var(--bg-primary);
            padding: 0 8px;
            font-size: 12px;
            font-weight: 500;
            color: var(--text-secondary);
            transition: var(--transition-normal);
        }
        
        /* Custom card component */
        .custom-card {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: var(--border-radius-lg);
            padding: 1.5rem;
            margin: 1rem 0;
            transition: var(--transition-normal);
            position: relative;
            overflow: hidden;
        }
        
        .custom-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--primary-color), var(--info-color));
        }
        
        .custom-card:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-lg);
            border-color: var(--primary-color);
        }
        
        /* Status badges */
        .status-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .status-badge.success {
            background: rgba(76, 175, 80, 0.2);
            color: var(--success-color);
            border: 1px solid rgba(76, 175, 80, 0.3);
        }
        
        .status-badge.warning {
            background: rgba(255, 193, 7, 0.2);
            color: var(--warning-color);
            border: 1px solid rgba(255, 193, 7, 0.3);
        }
        
        .status-badge.error {
            background: rgba(244, 67, 54, 0.2);
            color: var(--danger-color);
            border: 1px solid rgba(244, 67, 54, 0.3);
        }
        
        .status-badge.info {
            background: rgba(33, 150, 243, 0.2);
            color: var(--info-color);
            border: 1px solid rgba(33, 150, 243, 0.3);
        }
        
        /* Progress indicators */
        .progress-container {
            background: var(--bg-tertiary);
            border-radius: 10px;
            overflow: hidden;
            height: 8px;
            margin: 1rem 0;
        }
        
        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, var(--primary-color), var(--info-color));
            border-radius: 10px;
            transition: width 0.3s ease;
            position: relative;
        }
        
        .progress-bar::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(45deg, transparent 30%, rgba(255, 255, 255, 0.3) 50%, transparent 70%);
            animation: progress-shine 2s infinite;
        }
        
        @keyframes progress-shine {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
        
        /* Tooltip styling */
        .custom-tooltip {
            position: relative;
            cursor: help;
        }
        
        .custom-tooltip::after {
            content: attr(data-tooltip);
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            background: var(--bg-tertiary);
            color: var(--text-primary);
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 12px;
            white-space: nowrap;
            opacity: 0;
            visibility: hidden;
            transition: all 0.3s ease;
            z-index: 1000;
            border: 1px solid var(--border-color);
            box-shadow: var(--shadow-md);
        }
        
        .custom-tooltip:hover::after {
            opacity: 1;
            visibility: visible;
            transform: translateX(-50%) translateY(-5px);
        }
        
        /* Loading skeleton */
        .skeleton {
            background: linear-gradient(90deg, var(--bg-secondary) 0%, var(--bg-tertiary) 50%, var(--bg-secondary) 100%);
            background-size: 200% 100%;
            animation: skeleton-loading 1.5s infinite;
            border-radius: var(--border-radius-sm);
        }
        
        @keyframes skeleton-loading {
            0% { background-position: 200% 0; }
            100% { background-position: -200% 0; }
        }
        
        /* Chart container styling */
        .chart-container {
            background: var(--bg-primary);
            border: 1px solid var(--border-color);
            border-radius: var(--border-radius-lg);
            padding: 1rem;
            margin: 1rem 0;
            transition: var(--transition-normal);
        }
        
        .chart-container:hover {
            box-shadow: var(--shadow-md);
            border-color: var(--primary-color);
        }
        
        /* Data table enhancements */
        .enhanced-table {
            background: var(--bg-primary);
            border: 1px solid var(--border-color);
            border-radius: var(--border-radius-lg);
            overflow: hidden;
            margin: 1rem 0;
        }
        
        .enhanced-table thead {
            background: var(--bg-secondary);
        }
        
        .enhanced-table th, .enhanced-table td {
            padding: 12px 16px;
            border-bottom: 1px solid var(--border-color);
            color: var(--text-primary);
        }
        
        .enhanced-table tr:hover {
            background: var(--hover-bg);
        }
        
        /* Advanced button variants */
        .btn-primary {
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--info-color) 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: var(--border-radius-md);
            font-weight: 500;
            cursor: pointer;
            transition: var(--transition-normal);
            position: relative;
            overflow: hidden;
        }
        
        .btn-secondary {
            background: var(--bg-secondary);
            color: var(--text-primary);
            border: 2px solid var(--border-color);
            padding: 12px 24px;
            border-radius: var(--border-radius-md);
            font-weight: 500;
            cursor: pointer;
            transition: var(--transition-normal);
        }
        
        .btn-secondary:hover {
            border-color: var(--primary-color);
            background: var(--hover-bg);
            transform: translateY(-1px);
        }
        
        /* Form styling */
        .form-group {
            margin-bottom: 1.5rem;
        }
        
        .form-label {
            display: block;
            margin-bottom: 0.5rem;
            color: var(--text-primary);
            font-weight: 500;
            font-size: 14px;
        }
        
        .form-input {
            width: 100%;
            padding: 12px 16px;
            border: 2px solid var(--input-border);
            border-radius: var(--border-radius-md);
            background: var(--input-bg);
            color: var(--text-primary);
            font-family: inherit;
            transition: var(--transition-normal);
        }
        
        .form-input:focus {
            border-color: var(--input-focus);
            box-shadow: 0 0 0 3px rgba(13, 110, 253, 0.25);
            outline: none;
        }
        
        /* Dark theme specific adjustments */
        @media (prefers-color-scheme: dark) {
            .theme-notification {
                background: linear-gradient(135deg, #66bb6a 0%, #81c784 100%);
            }
            
            .status-badge.success {
                background: rgba(102, 187, 106, 0.2);
                border-color: rgba(102, 187, 106, 0.3);
            }
        }
    </style>
    """
    
    return theme_css

def create_custom_component(component_type, content, **kwargs):
    """
    Create a custom styled component with theme support.
    
    Args:
        component_type (str): Type of component ('card', 'badge', 'progress', etc.)
        content (str): Content to display
        **kwargs: Additional styling options
    
    Returns:
        str: HTML string for the component
    """
    
    if component_type == 'card':
        title = kwargs.get('title', '')
        return f"""
        <div class="custom-card">
            {f'<h4 style="margin-top: 0; color: var(--text-primary);">{title}</h4>' if title else ''}
            <div style="color: var(--text-primary);">{content}</div>
        </div>
        """
    
    elif component_type == 'badge':
        status = kwargs.get('status', 'info')
        return f'<span class="status-badge {status}">{content}</span>'
    
    elif component_type == 'progress':
        percentage = kwargs.get('percentage', 0)
        return f"""
        <div class="progress-container">
            <div class="progress-bar" style="width: {percentage}%;"></div>
        </div>
        """
    
    elif component_type == 'tooltip':
        tooltip_text = kwargs.get('tooltip', '')
        return f'<span class="custom-tooltip" data-tooltip="{tooltip_text}">{content}</span>'
    
    return content

def show_theme_notification(theme_name):
    """
    Show a notification when theme changes.
    
    Args:
        theme_name (str): Name of the current theme
    """
    
    notification_script = f"""
    <script>
        (function() {{
            // Remove existing notification
            const existing = document.querySelector('.theme-notification');
            if (existing) existing.remove();
            
            // Create new notification
            const notification = document.createElement('div');
            notification.className = 'theme-notification';
            notification.innerHTML = '🎨 {theme_name.title()} theme active';
            document.body.appendChild(notification);
            
            // Show notification
            setTimeout(() => notification.classList.add('show'), 100);
            
            // Hide after 3 seconds
            setTimeout(() => {{
                notification.classList.remove('show');
                setTimeout(() => notification.remove(), 300);
            }}, 3000);
        }})();
    </script>
    """
    
    st.markdown(notification_script, unsafe_allow_html=True)

def apply_custom_theme():
    """
    Apply the complete custom theme to the Streamlit app.
    This is the main function to call for theme setup.
    """
    
    # Inject the main theme CSS
    st.markdown(inject_theme_css(), unsafe_allow_html=True)
    
    # Add theme detection script
    theme_script = """
    <script>
        // Initialize theme on page load
        document.addEventListener('DOMContentLoaded', function() {
            // Detect initial theme
            const isDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
            
            // Show initial notification
            setTimeout(() => {
                const notification = document.createElement('div');
                notification.className = 'theme-notification show';
                notification.innerHTML = `🎨 ${isDark ? 'Dark' : 'Light'} theme detected`;
                document.body.appendChild(notification);
                
                setTimeout(() => {
                    notification.classList.remove('show');
                    setTimeout(() => notification.remove(), 300);
                }, 2000);
            }, 1000);
        });
    </script>
    """
    
    st.markdown(theme_script, unsafe_allow_html=True)

def create_enhanced_metrics(metrics_data):
    """
    Create enhanced metric displays with theme support.
    
    Args:
        metrics_data (list): List of dictionaries with metric data
                           Each dict should have: label, value, delta (optional)
    
    Returns:
        str: HTML string for enhanced metrics
    """
    
    metrics_html = '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 1rem 0;">'
    
    for metric in metrics_data:
        label = metric.get('label', '')
        value = metric.get('value', '')
        delta = metric.get('delta', '')
        trend = metric.get('trend', 'neutral')  # positive, negative, neutral
        
        trend_colors = {
            'positive': 'var(--success-color)',
            'negative': 'var(--danger-color)',
            'neutral': 'var(--text-secondary)'
        }
        
        trend_icons = {
            'positive': '↗️',
            'negative': '↘️',
            'neutral': '➡️'
        }
        
        metrics_html += f"""
        <div class="metric-card">
            <div style="font-size: 14px; color: var(--text-secondary); margin-bottom: 0.5rem;">{label}</div>
            <div style="font-size: 2rem; font-weight: bold; color: var(--text-primary); margin-bottom: 0.5rem;">{value}</div>
            {f'<div style="font-size: 12px; color: {trend_colors[trend]};">{trend_icons[trend]} {delta}</div>' if delta else ''}
        </div>
        """
    
    metrics_html += '</div>'
    return metrics_html

# Export the main functions
__all__ = [
    'inject_theme_css',
    'create_custom_component', 
    'show_theme_notification',
    'apply_custom_theme',
    'create_enhanced_metrics'
]
