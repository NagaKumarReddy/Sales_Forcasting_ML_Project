"""
Visualization Module for Sales Forecasting Project
Handles all plotting and visualization tasks
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SalesVisualizer:
    def __init__(self):
        """Initialize the SalesVisualizer"""
        self.fig_size = (12, 8)
        
    def plot_sales_trend(self, data, title="Monthly Sales Trend", save_path=None):
        """
        Plot monthly sales trend
        
        Args:
            data (pd.DataFrame): Monthly sales data with 'Date' and 'Sales' columns
            title (str): Plot title
            save_path (str): Path to save the plot
        """
        plt.figure(figsize=self.fig_size)
        
        plt.plot(data['Date'], data['Sales'], marker='o', linewidth=2, markersize=6)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Sales ($)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Add trend line
        z = np.polyfit(range(len(data)), data['Sales'], 1)
        p = np.poly1d(z)
        plt.plot(data['Date'], p(range(len(data)), "r--", alpha=0.8, label='Trend Line')
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_sales_distribution(self, data, title="Sales Distribution", save_path=None):
        """
        Plot sales distribution
        
        Args:
            data (pd.DataFrame): Sales data
            title (str): Plot title
            save_path (str): Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(data['Sales'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Sales Distribution', fontsize=14)
        plt.xlabel('Sales ($)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        
        plt.subplot(1, 2, 2)
        plt.boxplot(data['Sales'])
        plt.title('Sales Box Plot', fontsize=14)
        plt.ylabel('Sales ($)', fontsize=12)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_forecast(self, historical_data, forecast_data, title="Sales Forecast", save_path=None):
        """
        Plot historical data with forecast
        
        Args:
            historical_data (pd.DataFrame): Historical sales data
            forecast_data (pd.DataFrame): Forecast data
            title (str): Plot title
            save_path (str): Path to save the plot
        """
        plt.figure(figsize=self.fig_size)
        
        # Plot historical data
        plt.plot(historical_data['Date'], historical_data['Sales'], 
                marker='o', linewidth=2, markersize=6, label='Historical Sales')
        
        # Plot forecast
        plt.plot(forecast_data['Date'], forecast_data['Forecast'], 
                marker='s', linewidth=2, markersize=6, color='red', label='Forecast')
        
        # Add confidence intervals if available
        if 'Lower_Bound' in forecast_data.columns and 'Upper_Bound' in forecast_data.columns:
            plt.fill_between(forecast_data['Date'], 
                           forecast_data['Lower_Bound'], 
                           forecast_data['Upper_Bound'], 
                           alpha=0.3, color='red', label='Confidence Interval')
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Sales ($)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_interactive_plot(self, historical_data, forecast_data=None):
        """
        Create interactive plot using Plotly
        
        Args:
            historical_data (pd.DataFrame): Historical sales data
            forecast_data (pd.DataFrame): Forecast data (optional)
        
        Returns:
            plotly.graph_objects.Figure: Interactive plot
        """
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=historical_data['Date'],
            y=historical_data['Sales'],
            mode='lines+markers',
            name='Historical Sales',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))
        
        # Add forecast if available
        if forecast_data is not None:
            fig.add_trace(go.Scatter(
                x=forecast_data['Date'],
                y=forecast_data['Forecast'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color='red', width=2, dash='dash'),
                marker=dict(size=6, symbol='square')
            ))
            
            # Add confidence intervals if available
            if 'Lower_Bound' in forecast_data.columns and 'Upper_Bound' in forecast_data.columns:
                fig.add_trace(go.Scatter(
                    x=forecast_data['Date'].tolist() + forecast_data['Date'].tolist()[::-1],
                    y=forecast_data['Upper_Bound'].tolist() + forecast_data['Lower_Bound'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(255,0,0,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Confidence Interval'
                ))
        
        fig.update_layout(
            title='Sales Forecast - Interactive View',
            xaxis_title='Date',
            yaxis_title='Sales ($)',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def plot_seasonal_decomposition(self, data, period=12, title="Seasonal Decomposition", save_path=None):
        """
        Plot seasonal decomposition of time series
        
        Args:
            data (pd.Series): Time series data
            period (int): Period for seasonal decomposition
            title (str): Plot title
            save_path (str): Path to save the plot
        """
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        # Perform seasonal decomposition
        decomposition = seasonal_decompose(data, period=period, extrapolate_trend='freq')
        
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        
        # Original data
        axes[0].plot(data.index, data.values)
        axes[0].set_title('Original Time Series')
        axes[0].grid(True, alpha=0.3)
        
        # Trend
        axes[1].plot(data.index, decomposition.trend)
        axes[1].set_title('Trend')
        axes[1].grid(True, alpha=0.3)
        
        # Seasonal
        axes[2].plot(data.index, decomposition.seasonal)
        axes[2].set_title('Seasonal')
        axes[2].grid(True, alpha=0.3)
        
        # Residual
        axes[3].plot(data.index, decomposition.resid)
        axes[3].set_title('Residual')
        axes[3].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

def main():
    """Test the visualization functionality"""
    from data_loader import DataLoader
    
    # Load and preprocess data
    loader = DataLoader('global_superstore_2016.xlsx')
    data = loader.load_data()
    monthly_data = loader.preprocess_data()
    
    if monthly_data is not None:
        # Initialize visualizer
        visualizer = SalesVisualizer()
        
        # Plot sales trend
        visualizer.plot_sales_trend(monthly_data)
        
        # Plot sales distribution
        visualizer.plot_sales_distribution(monthly_data)
        
        # Create interactive plot
        fig = visualizer.create_interactive_plot(monthly_data)
        fig.show()

if __name__ == "__main__":
    main() 