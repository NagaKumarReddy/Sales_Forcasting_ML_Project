# Sales Forecasting ML Project

A comprehensive sales forecasting system using time series analysis with ARIMA and Prophet models. This project provides multiple interfaces for sales forecasting including a complete pipeline, CLI interface, and Streamlit web application.

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)
- [Troubleshooting](#troubleshooting)

## 🚀 Features

- **Data Loading & Preprocessing**: Automatic Excel file loading and monthly sales aggregation
- **Time Series Analysis**: ARIMA and Prophet models for forecasting
- **Data Visualization**: Interactive plots with Plotly and static plots with Matplotlib
- **Multiple Interfaces**: 
  - Complete pipeline script
  - Command-line interface (CLI)
  - Streamlit web application
- **Model Evaluation**: Comprehensive metrics (MAE, MSE, RMSE, MAPE)
- **Forecast Generation**: Predict sales for customizable time periods

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Download the Project

```bash
# If using git
git clone <repository-url>
cd sales-forecasting-project

# Or simply download and extract the project files
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
python -c "import pandas, numpy, matplotlib, streamlit; print('All dependencies installed successfully!')"
```

## 🎯 Quick Start

### Option 1: Run Complete Pipeline

```bash
python main.py
```

This will:
- Load the Global Superstore dataset
- Preprocess and visualize the data
- Train ARIMA and Prophet models
- Generate 12-month forecasts
- Display results and plots

### Option 2: Use Command Line Interface

```bash
# Interactive mode
python cli_interface.py --interactive

# Direct forecasting
python cli_interface.py --months 12
```

### Option 3: Launch Streamlit Web App

```bash
streamlit run streamlit_app.py
```

Then open your browser to `http://localhost:8501`

## 📖 Usage

### 1. Complete Pipeline (`main.py`)

The main application provides a complete end-to-end pipeline:

```python
from main import SalesForecastingApp

# Initialize the application
app = SalesForecastingApp('global_superstore_2016.xlsx')

# Run complete pipeline
results = app.run_complete_pipeline(forecast_months=12)

# Get forecast summary
summary = app.get_forecast_summary(months=6)
```

### 2. Data Loading (`data_loader.py`)

```python
from data_loader import DataLoader

# Load data
loader = DataLoader('global_superstore_2016.xlsx')
data = loader.load_data()

# Preprocess data
monthly_data = loader.preprocess_data()

# Get data info
info = loader.get_data_info()
```

### 3. Forecasting Models (`forecasting_models.py`)

```python
from forecasting_models import ARIMAModel, ProphetModel, ForecastingModelSelector

# Initialize models
arima_model = ARIMAModel(order=(1, 1, 1))
prophet_model = ProphetModel()

# Train models
arima_model.fit(monthly_data['Sales'])
prophet_model.fit(monthly_data[['Date', 'Sales']])

# Generate forecasts
arima_forecast = arima_model.forecast(steps=12)
prophet_forecast = prophet_model.forecast(periods=12)
```

### 4. Visualization (`visualization.py`)

```python
from visualization import SalesVisualizer

visualizer = SalesVisualizer()

# Plot sales trend
visualizer.plot_sales_trend(monthly_data)

# Create interactive plot
fig = visualizer.create_interactive_plot(monthly_data, forecast_data)
fig.show()
```

### 5. CLI Interface (`cli_interface.py`)

```bash
# Interactive mode
python cli_interface.py --interactive

# Direct forecasting with custom parameters
python cli_interface.py --file data.xlsx --months 18
```

### 6. Streamlit Web App (`streamlit_app.py`)

```bash
streamlit run streamlit_app.py
```

The web app provides:
- File upload interface
- Interactive data visualization
- Model training and evaluation
- Forecast generation with customizable parameters

## 📁 Project Structure

```
sales-forecasting-project/
├── global_superstore_2016.xlsx    # Dataset
├── requirements.txt                # Dependencies
├── README.md                      # This file
├── main.py                        # Complete pipeline
├── data_loader.py                 # Data loading and preprocessing
├── forecasting_models.py          # ARIMA and Prophet models
├── visualization.py               # Plotting and visualization
├── cli_interface.py              # Command-line interface
└── streamlit_app.py              # Streamlit web application
```

## 🔧 API Documentation

### DataLoader Class

```python
class DataLoader:
    def __init__(self, file_path: str)
    def load_data(self, sheet_name=0) -> pd.DataFrame
    def preprocess_data(self) -> pd.DataFrame
    def get_data_info(self) -> dict
```

### ARIMAModel Class

```python
class ARIMAModel:
    def __init__(self, order=(1, 1, 1))
    def fit(self, data: pd.Series)
    def forecast(self, steps=12) -> pd.DataFrame
    def evaluate(self, test_data: pd.Series) -> dict
    def check_stationarity(self, data: pd.Series) -> dict
```

### ProphetModel Class

```python
class ProphetModel:
    def __init__(self, yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    def fit(self, data: pd.DataFrame)
    def forecast(self, periods=12) -> pd.DataFrame
    def evaluate(self, test_data: pd.DataFrame) -> dict
```

### SalesVisualizer Class

```python
class SalesVisualizer:
    def plot_sales_trend(self, data: pd.DataFrame, title="Monthly Sales Trend", save_path=None)
    def plot_sales_distribution(self, data: pd.DataFrame, title="Sales Distribution", save_path=None)
    def plot_forecast(self, historical_data: pd.DataFrame, forecast_data: pd.DataFrame, title="Sales Forecast", save_path=None)
    def create_interactive_plot(self, historical_data: pd.DataFrame, forecast_data=None) -> plotly.graph_objects.Figure
    def plot_seasonal_decomposition(self, data: pd.Series, period=12, title="Seasonal Decomposition", save_path=None)
```

## 🛠️ Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Solution: Install missing dependencies
   pip install -r requirements.txt
   ```

2. **Prophet Installation Issues**
   ```bash
   # On Windows, you might need:
   conda install -c conda-forge prophet
   
   # Or try:
   pip install prophet --no-deps
   pip install cmdstanpy
   ```

3. **Memory Issues with Large Datasets**
   ```python
   # Use chunking for large files
   data = pd.read_excel('file.xlsx', chunksize=10000)
   ```

4. **Plot Display Issues**
   ```python
   # For Jupyter notebooks
   %matplotlib inline
   
   # For headless environments
   import matplotlib
   matplotlib.use('Agg')
   ```

### Performance Tips

1. **For Large Datasets**: Use data sampling or aggregation
2. **For Faster Training**: Reduce ARIMA parameter search space
3. **For Better Visualizations**: Use Plotly for interactive plots

### Model Selection Guide

- **ARIMA**: Good for stationary time series with clear trends
- **Prophet**: Excellent for seasonal data with holidays and trend changes
- **Combined**: Use both models and compare results

## 📊 Example Output

### Data Information
```
Dataset Shape: (51,294 rows × 21 cols)
Date Range: 2011-01-01 to 2014-12-31
Total Sales: $2,297,200.86
Monthly data shape: (48, 3)
```

### Model Performance
```
ARIMA:
  MAE: 12,345.67
  MSE: 234,567,890.12
  RMSE: 15,345.67
  MAPE: 8.45%

Prophet:
  MAE: 11,234.56
  MSE: 198,765,432.10
  RMSE: 14,098.76
  MAPE: 7.23%
```

### Forecast Example
```
ARIMA Forecast:
2025-01: $45,234.56
2025-02: $46,789.12
2025-03: $47,123.45
...
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Global Superstore dataset for providing sample data
- Facebook Prophet team for the excellent forecasting library
- Statsmodels team for ARIMA implementation
- Streamlit team for the web framework

---

**Happy Forecasting! 📈** 