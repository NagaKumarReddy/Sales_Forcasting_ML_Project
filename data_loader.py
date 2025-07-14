"""
Data Loader Module for Sales Forecasting Project
Handles loading and preprocessing of the Global Superstore dataset
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    def __init__(self, file_path):
        """
        Initialize DataLoader with file path
        
        Args:
            file_path (str): Path to the Excel file
        """
        self.file_path = file_path
        self.data = None
        
    def load_data(self, sheet_name=0):
        """
        Load data from Excel file
        
        Args:
            sheet_name: Sheet name or index to load (default: 0)
        
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            print(f"Loading data from {self.file_path}...")
            self.data = pd.read_excel(self.file_path, sheet_name=sheet_name)
            print(f"Data loaded successfully! Shape: {self.data.shape}")
            return self.data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def preprocess_data(self):
        """
        Preprocess the data for time series analysis
        
        Returns:
            pd.DataFrame: Preprocessed data with monthly sales aggregation
        """
        if self.data is None:
            print("No data loaded. Please load data first.")
            return None
        
        print("Preprocessing data...")
        
        # Convert Order Date to datetime
        self.data['Order Date'] = pd.to_datetime(self.data['Order Date'])
        
        # Create a copy for processing
        df = self.data.copy()
        
        # Aggregate sales by month
        df['Year-Month'] = df['Order Date'].dt.to_period('M')
        monthly_sales = df.groupby('Year-Month')['Sales'].sum().reset_index()
        monthly_sales['Year-Month'] = monthly_sales['Year-Month'].astype(str)
        monthly_sales['Date'] = pd.to_datetime(monthly_sales['Year-Month'])
        
        # Sort by date
        monthly_sales = monthly_sales.sort_values('Date').reset_index(drop=True)
        
        print(f"Preprocessing complete! Monthly data shape: {monthly_sales.shape}")
        return monthly_sales
    
    def get_data_info(self):
        """
        Get basic information about the loaded data
        
        Returns:
            dict: Data information
        """
        if self.data is None:
            return None
        
        info = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'date_range': {
                'min': self.data['Order Date'].min(),
                'max': self.data['Order Date'].max()
            },
            'total_sales': self.data['Sales'].sum(),
            'missing_values': self.data.isnull().sum().to_dict()
        }
        
        return info

def main():
    """Test the DataLoader functionality"""
    # Initialize loader
    loader = DataLoader('global_superstore_2016.xlsx')
    
    # Load data
    data = loader.load_data()
    
    if data is not None:
        # Get data info
        info = loader.get_data_info()
        print("\nData Information:")
        print(f"Shape: {info['shape']}")
        print(f"Date Range: {info['date_range']['min']} to {info['date_range']['max']}")
        print(f"Total Sales: ${info['total_sales']:,.2f}")
        
        # Preprocess data
        monthly_data = loader.preprocess_data()
        
        if monthly_data is not None:
            print("\nFirst 5 rows of monthly data:")
            print(monthly_data.head())
            
            return monthly_data
    
    return None

if __name__ == "__main__":
    main() 