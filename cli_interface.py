"""
Command Line Interface for Sales Forecasting Project
Provides an interactive CLI for sales forecasting
"""

import argparse
import sys
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data_loader import DataLoader
from forecasting_models import ARIMAModel, ProphetModel, ForecastingModelSelector
from visualization import SalesVisualizer

class SalesForecastingCLI:
    def __init__(self):
        """Initialize the CLI interface"""
        self.data_loader = None
        self.monthly_data = None
        self.model_selector = None
        self.visualizer = SalesVisualizer()
        
    def load_data(self, file_path):
        """
        Load and preprocess data
        
        Args:
            file_path (str): Path to the Excel file
        """
        print("=" * 50)
        print("SALES FORECASTING SYSTEM")
        print("=" * 50)
        
        # Initialize data loader
        self.data_loader = DataLoader(file_path)
        
        # Load data
        data = self.data_loader.load_data()
        if data is None:
            print("Error: Could not load data!")
            return False
        
        # Get data info
        info = self.data_loader.get_data_info()
        print(f"\nDataset Information:")
        print(f"Shape: {info['shape']}")
        print(f"Date Range: {info['date_range']['min']} to {info['date_range']['max']}")
        print(f"Total Sales: ${info['total_sales']:,.2f}")
        
        # Preprocess data
        self.monthly_data = self.data_loader.preprocess_data()
        if self.monthly_data is None:
            print("Error: Could not preprocess data!")
            return False
        
        print(f"\nMonthly data shape: {self.monthly_data.shape}")
        print("Data loaded successfully!")
        return True
    
    def visualize_data(self):
        """Visualize the sales data"""
        if self.monthly_data is None:
            print("Error: No data loaded!")
            return
        
        print("\n" + "=" * 30)
        print("DATA VISUALIZATION")
        print("=" * 30)
        
        # Plot sales trend
        print("Generating sales trend plot...")
        self.visualizer.plot_sales_trend(self.monthly_data)
        
        # Plot sales distribution
        print("Generating sales distribution plot...")
        self.visualizer.plot_sales_distribution(self.monthly_data)
        
        # Seasonal decomposition
        print("Generating seasonal decomposition...")
        self.visualizer.plot_seasonal_decomposition(self.monthly_data['Sales'])
    
    def train_models(self):
        """Train forecasting models"""
        if self.monthly_data is None:
            print("Error: No data loaded!")
            return
        
        print("\n" + "=" * 30)
        print("MODEL TRAINING")
        print("=" * 30)
        
        # Split data
        train_size = int(len(self.monthly_data) * 0.8)
        train_data = self.monthly_data.iloc[:train_size]
        test_data = self.monthly_data.iloc[train_size:]
        
        print(f"Training data: {len(train_data)} months")
        print(f"Test data: {len(test_data)} months")
        
        # Initialize model selector
        self.model_selector = ForecastingModelSelector()
        
        # Add ARIMA model
        print("\nAdding ARIMA model...")
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
    
    def generate_forecast(self, months=12):
        """
        Generate sales forecast
        
        Args:
            months (int): Number of months to forecast
        """
        if self.model_selector is None:
            print("Error: Models not trained! Please train models first.")
            return
        
        print(f"\n" + "=" * 30)
        print(f"FORECASTING ({months} MONTHS)")
        print("=" * 30)
        
        # Generate forecasts
        forecasts = self.model_selector.forecast_all_models(steps=months)
        
        # Display results
        for model_name, forecast in forecasts.items():
            print(f"\n{model_name} Forecast:")
            print("-" * 20)
            
            # Calculate future dates
            last_date = self.monthly_data['Date'].iloc[-1]
            future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                       periods=months, freq='M')
            
            forecast['Date'] = future_dates
            
            # Display forecast
            for i, row in forecast.iterrows():
                date_str = row['Date'].strftime('%Y-%m')
                forecast_val = row['Forecast']
                lower_val = row.get('Lower_Bound', 'N/A')
                upper_val = row.get('Upper_Bound', 'N/A')
                
                print(f"{date_str}: ${forecast_val:,.2f}")
                if lower_val != 'N/A':
                    print(f"  Range: ${lower_val:,.2f} - ${upper_val:,.2f}")
            
            # Plot forecast
            print(f"\nGenerating {model_name} forecast plot...")
            self.visualizer.plot_forecast(self.monthly_data, forecast, 
                                        title=f"{model_name} Sales Forecast")
    
    def interactive_mode(self):
        """Run interactive mode"""
        print("\n" + "=" * 50)
        print("INTERACTIVE MODE")
        print("=" * 50)
        
        while True:
            print("\nOptions:")
            print("1. Visualize data")
            print("2. Train models")
            print("3. Generate forecast")
            print("4. Exit")
            
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == '1':
                self.visualize_data()
            
            elif choice == '2':
                self.train_models()
            
            elif choice == '3':
                if self.model_selector is None:
                    print("Please train models first (option 2)")
                    continue
                
                try:
                    months = int(input("Enter number of months to forecast: "))
                    self.generate_forecast(months)
                except ValueError:
                    print("Invalid input! Please enter a number.")
            
            elif choice == '4':
                print("Goodbye!")
                break
            
            else:
                print("Invalid choice! Please enter 1-4.")

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description='Sales Forecasting CLI')
    parser.add_argument('--file', default='global_superstore_2016.xlsx', 
                       help='Path to the Excel file')
    parser.add_argument('--months', type=int, default=12, 
                       help='Number of months to forecast')
    parser.add_argument('--interactive', action='store_true', 
                       help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Initialize CLI
    cli = SalesForecastingCLI()
    
    # Load data
    if not cli.load_data(args.file):
        sys.exit(1)
    
    if args.interactive:
        cli.interactive_mode()
    else:
        # Run full pipeline
        cli.visualize_data()
        cli.train_models()
        cli.generate_forecast(args.months)

if __name__ == "__main__":
    main() 