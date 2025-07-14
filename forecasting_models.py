"""
Forecasting Models Module for Sales Forecasting Project
Implements ARIMA and Prophet models for time series forecasting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ARIMA imports
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

# Prophet imports
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    print("Prophet not available. Install with: pip install prophet")
    PROPHET_AVAILABLE = False

# Evaluation metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

class ARIMAModel:
    def __init__(self, order=(1, 1, 1)):
        """
        Initialize ARIMA model
        
        Args:
            order (tuple): (p, d, q) parameters for ARIMA
        """
        self.order = order
        self.model = None
        self.fitted_model = None
        
    def check_stationarity(self, data):
        """
        Check if time series is stationary using Augmented Dickey-Fuller test
        
        Args:
            data (pd.Series): Time series data
            
        Returns:
            dict: Test results
        """
        result = adfuller(data)
        
        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05
        }
    
    def find_best_order(self, data, max_p=3, max_d=2, max_q=3):
        """
        Find best ARIMA order using AIC
        
        Args:
            data (pd.Series): Time series data
            max_p (int): Maximum p value
            max_d (int): Maximum d value
            max_q (int): Maximum q value
            
        Returns:
            tuple: Best (p, d, q) order
        """
        best_aic = np.inf
        best_order = None
        
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(data, order=(p, d, q))
                        fitted_model = model.fit()
                        aic = fitted_model.aic
                        
                        if aic < best_aic:
                            best_aic = aic
                            best_order = (p, d, q)
                    except:
                        continue
        
        print(f"Best ARIMA order: {best_order} (AIC: {best_aic:.2f})")
        return best_order
    
    def fit(self, data):
        """
        Fit ARIMA model to data
        
        Args:
            data (pd.Series): Time series data
        """
        print(f"Fitting ARIMA{self.order} model...")
        
        # Check stationarity
        stationarity = self.check_stationarity(data)
        print(f"Series is {'stationary' if stationarity['is_stationary'] else 'non-stationary'}")
        
        # Fit model
        self.model = ARIMA(data, order=self.order)
        self.fitted_model = self.model.fit()
        
        print("ARIMA model fitted successfully!")
        print(f"AIC: {self.fitted_model.aic:.2f}")
        print(f"BIC: {self.fitted_model.bic:.2f}")
    
    def forecast(self, steps=12):
        """
        Generate forecast
        
        Args:
            steps (int): Number of steps to forecast
            
        Returns:
            pd.DataFrame: Forecast results
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before forecasting")
        
        # Generate forecast
        forecast_result = self.fitted_model.forecast(steps=steps)
        conf_int = self.fitted_model.get_forecast(steps=steps).conf_int()
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'Forecast': forecast_result,
            'Lower_Bound': conf_int.iloc[:, 0],
            'Upper_Bound': conf_int.iloc[:, 1]
        })
        
        return forecast_df
    
    def evaluate(self, test_data):
        """
        Evaluate model performance
        
        Args:
            test_data (pd.Series): Test data
            
        Returns:
            dict: Evaluation metrics
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before evaluation")
        
        # Generate predictions for test period
        predictions = self.fitted_model.forecast(steps=len(test_data))
        
        # Calculate metrics
        mae = mean_absolute_error(test_data, predictions)
        mse = mean_squared_error(test_data, predictions)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(test_data, predictions) * 100
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape
        }

class ProphetModel:
    def __init__(self, yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False):
        """
        Initialize Prophet model
        
        Args:
            yearly_seasonality (bool): Include yearly seasonality
            weekly_seasonality (bool): Include weekly seasonality
            daily_seasonality (bool): Include daily seasonality
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is not available. Please install it first.")
        
        self.model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality
        )
        self.fitted_model = None
        
    def prepare_data(self, data):
        """
        Prepare data for Prophet (requires 'ds' and 'y' columns)
        
        Args:
            data (pd.DataFrame): Data with 'Date' and 'Sales' columns
            
        Returns:
            pd.DataFrame: Prophet-formatted data
        """
        prophet_data = data.copy()
        prophet_data.columns = ['ds', 'y']
        return prophet_data
    
    def fit(self, data):
        """
        Fit Prophet model to data
        
        Args:
            data (pd.DataFrame): Data with 'Date' and 'Sales' columns
        """
        print("Fitting Prophet model...")
        
        # Prepare data for Prophet
        prophet_data = self.prepare_data(data)
        
        # Fit model
        self.fitted_model = self.model.fit(prophet_data)
        
        print("Prophet model fitted successfully!")
    
    def forecast(self, periods=12):
        """
        Generate forecast
        
        Args:
            periods (int): Number of periods to forecast
            
        Returns:
            pd.DataFrame: Forecast results
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before forecasting")
        
        # Create future dataframe
        future = self.fitted_model.make_future_dataframe(periods=periods, freq='M')
        
        # Generate forecast
        forecast = self.fitted_model.predict(future)
        
        # Extract forecast results
        forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
        forecast_df.columns = ['Date', 'Forecast', 'Lower_Bound', 'Upper_Bound']
        
        return forecast_df
    
    def evaluate(self, test_data):
        """
        Evaluate model performance
        
        Args:
            test_data (pd.DataFrame): Test data with 'Date' and 'Sales' columns
            
        Returns:
            dict: Evaluation metrics
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before evaluation")
        
        # Prepare test data
        test_prophet = self.prepare_data(test_data)
        
        # Generate predictions for test period
        future = self.fitted_model.make_future_dataframe(periods=len(test_data), freq='M')
        forecast = self.fitted_model.predict(future)
        
        # Get predictions for test period
        predictions = forecast[['ds', 'yhat']].tail(len(test_data))
        predictions.columns = ['Date', 'Prediction']
        
        # Align predictions with test data
        merged = pd.merge(test_data, predictions, on='Date')
        
        # Calculate metrics
        mae = mean_absolute_error(merged['Sales'], merged['Prediction'])
        mse = mean_squared_error(merged['Sales'], merged['Prediction'])
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(merged['Sales'], merged['Prediction']) * 100
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape
        }

class ForecastingModelSelector:
    def __init__(self):
        """Initialize model selector"""
        self.models = {}
        self.results = {}
        
    def add_model(self, name, model):
        """
        Add a model to the selector
        
        Args:
            name (str): Model name
            model: Model instance
        """
        self.models[name] = model
    
    def fit_all_models(self, data):
        """
        Fit all models to data
        
        Args:
            data (pd.DataFrame): Training data
        """
        for name, model in self.models.items():
            print(f"\nFitting {name}...")
            try:
                model.fit(data)
                print(f"{name} fitted successfully!")
            except Exception as e:
                print(f"Error fitting {name}: {e}")
    
    def forecast_all_models(self, steps=12):
        """
        Generate forecasts for all models
        
        Args:
            steps (int): Number of steps to forecast
            
        Returns:
            dict: Forecasts for all models
        """
        forecasts = {}
        
        for name, model in self.models.items():
            try:
                forecast = model.forecast(steps=steps)
                forecasts[name] = forecast
                print(f"{name} forecast generated successfully!")
            except Exception as e:
                print(f"Error generating forecast for {name}: {e}")
        
        return forecasts
    
    def evaluate_all_models(self, test_data):
        """
        Evaluate all models
        
        Args:
            test_data (pd.DataFrame): Test data
            
        Returns:
            dict: Evaluation results for all models
        """
        results = {}
        
        for name, model in self.models.items():
            try:
                metrics = model.evaluate(test_data)
                results[name] = metrics
                print(f"{name} evaluation completed!")
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
        
        return results

def main():
    """Test the forecasting models"""
    from data_loader import DataLoader
    
    # Load and preprocess data
    loader = DataLoader('global_superstore_2016.xlsx')
    data = loader.load_data()
    monthly_data = loader.preprocess_data()
    
    if monthly_data is not None:
        # Split data into train and test
        train_size = int(len(monthly_data) * 0.8)
        train_data = monthly_data.iloc[:train_size]
        test_data = monthly_data.iloc[train_size:]
        
        print(f"Train data: {len(train_data)} months")
        print(f"Test data: {len(test_data)} months")
        
        # Initialize model selector
        selector = ForecastingModelSelector()
        
        # Add ARIMA model
        arima_model = ARIMAModel(order=(1, 1, 1))
        selector.add_model('ARIMA', arima_model)
        
        # Add Prophet model if available
        if PROPHET_AVAILABLE:
            prophet_model = ProphetModel()
            selector.add_model('Prophet', prophet_model)
        
        # Fit all models
        selector.fit_all_models(train_data)
        
        # Generate forecasts
        forecasts = selector.forecast_all_models(steps=12)
        
        # Evaluate models
        results = selector.evaluate_all_models(test_data)
        
        print("\nEvaluation Results:")
        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.2f}")

if __name__ == "__main__":
    main() 