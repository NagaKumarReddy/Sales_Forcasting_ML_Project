"""
Main Application for Sales Forecasting Project
Complete pipeline for sales forecasting using time series analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from data_loader import DataLoader
from forecasting_models import ARIMAModel, ProphetModel, ForecastingModelSelector
from visualization import SalesVisualizer

class SalesForecastingApp:
    def __init__(self, file_path='global_superstore_2016.xlsx'):
        """
        Initialize the sales forecasting application
        
        Args:
            file_path (str): Path to the Excel file
        """
        self.file_path = file_path
        self.data_loader = None
        self.monthly_data = None
        self.model_selector = None
        self.visualizer = SalesVisualizer()
        
    def run_complete_pipeline(self, forecast_months=12):
        """
        Run the complete sales forecasting pipeline
        
        Args:
            forecast_months (int): Number of months to forecast
        """
        print("=" * 60)
        print("SALES FORECASTING SYSTEM - COMPLETE PIPELINE")
        print("=" * 60)
        
        # Step 1: Load and preprocess data
        print("\n📊 STEP 1: DATA LOADING AND PREPROCESSING")
        print("-" * 40)
        
        self.data_loader = DataLoader(self.file_path)
        data = self.data_loader.load_data()
        
        if data is None:
            print("Error: Could not load data!")
            return None
        
        # Get data info
        info = self.data_loader.get_data_info()
        print(f"Dataset Shape: {info['shape']}")
        print(f"Date Range: {info['date_range']['min']} to {info['date_range']['max']}")
        print(f"Total Sales: ${info['total_sales']:,.2f}")
        
        # Preprocess data
        self.monthly_data = self.data_loader.preprocess_data()
        if self.monthly_data is None:
            print("Error: Could not preprocess data!")
            return None
        
        print(f"Monthly data shape: {self.monthly_data.shape}")
        print("✅ Data preprocessing completed!")
        
        # Step 2: Data visualization
        print("\n📈 STEP 2: DATA VISUALIZATION")
        print("-" * 40)
        
        # Plot sales trend
        print("Generating sales trend plot...")
        self.visualizer.plot_sales_trend(self.monthly_data, 
                                       title="Global Superstore Monthly Sales Trend")
        
        # Plot sales distribution
        print("Generating sales distribution plot...")
        self.visualizer.plot_sales_distribution(self.monthly_data)
        
        # Seasonal decomposition
        print("Generating seasonal decomposition...")
        self.visualizer.plot_seasonal_decomposition(self.monthly_data['Sales'])
        
        print("✅ Data visualization completed!")
        
        # Step 3: Model training
        print("\n🤖 STEP 3: MODEL TRAINING")
        print("-" * 40)
        
        # Split data
        train_size = int(len(self.monthly_data) * 0.8)
        train_data = self.monthly_data.iloc[:train_size]
        test_data = self.monthly_data.iloc[train_size:]
        
        print(f"Training data: {len(train_data)} months")
        print(f"Test data: {len(test_data)} months")
        
        # Initialize model selector
        self.model_selector = ForecastingModelSelector()
        
        # Add ARIMA model
        print("\nTraining ARIMA model...")
        arima_model = ARIMAModel(order=(1, 1, 1))
        self.model_selector.add_model('ARIMA', arima_model)
        
        # Add Prophet model if available
        try:
            prophet_model = ProphetModel()
            self.model_selector.add_model('Prophet', prophet_model)
            print("Prophet model added successfully!")
        except ImportError:
            print("Prophet not available. Using ARIMA only.")
        
        # Fit all models
        self.model_selector.fit_all_models(train_data)
        
        # Evaluate models
        print("\nEvaluating models...")
        results = self.model_selector.evaluate_all_models(test_data)
        
        print("\nModel Performance:")
        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.2f}")
        
        print("✅ Model training completed!")
        
        # Step 4: Generate forecasts
        print(f"\n🔮 STEP 4: FORECAST GENERATION ({forecast_months} MONTHS)")
        print("-" * 40)
        
        # Generate forecasts
        forecasts = self.model_selector.forecast_all_models(steps=forecast_months)
        
        # Display and plot results
        for model_name, forecast in forecasts.items():
            print(f"\n{model_name} Forecast:")
            print("-" * 30)
            
            # Calculate future dates
            last_date = self.monthly_data['Date'].iloc[-1]
            future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                       periods=forecast_months, freq='M')
            
            forecast['Date'] = future_dates
            
            # Display forecast
            for i, row in forecast.iterrows():
                date_str = row['Date'].strftime('%Y-%m')
                forecast_val = row['Forecast']
                print(f"{date_str}: ${forecast_val:,.2f}")
            
            # Plot forecast
            print(f"\nGenerating {model_name} forecast plot...")
            self.visualizer.plot_forecast(self.monthly_data, forecast, 
                                        title=f"{model_name} Sales Forecast")
        
        print("✅ Forecast generation completed!")
        
        # Step 5: Summary
        print("\n📋 STEP 5: SUMMARY")
        print("-" * 40)
        
        print("Pipeline completed successfully!")
        print(f"Dataset: {self.file_path}")
        print(f"Total records: {info['shape'][0]:,}")
        print(f"Monthly periods: {len(self.monthly_data)}")
        print(f"Models trained: {len(self.model_selector.models)}")
        print(f"Forecast period: {forecast_months} months")
        
        return {
            'monthly_data': self.monthly_data,
            'models': self.model_selector.models,
            'forecasts': forecasts,
            'evaluation': results
        }
    
    def get_forecast_summary(self, months=12):
        """
        Get a summary of forecasts for a specific number of months
        
        Args:
            months (int): Number of months to forecast
            
        Returns:
            dict: Forecast summary
        """
        if self.model_selector is None:
            print("Error: Models not trained! Please run the pipeline first.")
            return None
        
        forecasts = self.model_selector.forecast_all_models(steps=months)
        
        summary = {}
        for model_name, forecast in forecasts.items():
            # Calculate future dates
            last_date = self.monthly_data['Date'].iloc[-1]
            future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                       periods=months, freq='M')
            
            forecast['Date'] = future_dates
            
            summary[model_name] = {
                'forecast': forecast,
                'total_forecast': forecast['Forecast'].sum(),
                'avg_forecast': forecast['Forecast'].mean(),
                'min_forecast': forecast['Forecast'].min(),
                'max_forecast': forecast['Forecast'].max()
            }
        
        return summary

def main():
    """Main function to run the sales forecasting application"""
    # Initialize the application
    app = SalesForecastingApp()
    
    # Run the complete pipeline
    results = app.run_complete_pipeline(forecast_months=12)
    
    if results:
        print("\n🎉 Sales forecasting pipeline completed successfully!")
        print("You can now use the CLI or Streamlit interface for interactive forecasting.")
    else:
        print("\n❌ Pipeline failed. Please check the error messages above.")

if __name__ == "__main__":
    main() 