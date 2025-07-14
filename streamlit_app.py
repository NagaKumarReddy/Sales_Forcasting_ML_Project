"""
Streamlit Web Application for Sales Forecasting Project
Provides an interactive web interface for sales forecasting
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from data_loader import DataLoader
from forecasting_models import ARIMAModel, ProphetModel, ForecastingModelSelector
from visualization import SalesVisualizer

# Page configuration
st.set_page_config(
    page_title="Sales Forecasting System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StreamlitApp:
    def __init__(self):
        """Initialize the Streamlit app"""
        self.data_loader = None
        self.monthly_data = None
        self.model_selector = None
        self.visualizer = SalesVisualizer()
        
        # Initialize session state
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'models_trained' not in st.session_state:
            st.session_state.models_trained = False
        if 'forecasts' not in st.session_state:
            st.session_state.forecasts = {}
    
    def load_data(self):
        """Load and preprocess data"""
        st.header("📊 Data Loading")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload your Excel file", 
            type=['xlsx', 'xls'],
            help="Upload the Global Superstore dataset"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with open("temp_data.xlsx", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load data
            self.data_loader = DataLoader("temp_data.xlsx")
            data = self.data_loader.load_data()
            
            if data is not None:
                # Get data info
                info = self.data_loader.get_data_info()
                
                # Display data info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Dataset Shape", f"{info['shape'][0]:,} rows × {info['shape'][1]} cols")
                with col2:
                    st.metric("Date Range", f"{info['date_range']['min'].strftime('%Y-%m')} to {info['date_range']['max'].strftime('%Y-%m')}")
                with col3:
                    st.metric("Total Sales", f"${info['total_sales']:,.2f}")
                
                # Preprocess data
                self.monthly_data = self.data_loader.preprocess_data()
                
                if self.monthly_data is not None:
                    st.session_state.data_loaded = True
                    st.success("Data loaded successfully!")
                    
                    # Show sample data
                    st.subheader("Sample Monthly Data")
                    st.dataframe(self.monthly_data.head(10))
                    
                    return True
        
        return False
    
    def visualize_data(self):
        """Visualize the sales data"""
        if not st.session_state.data_loaded:
            st.warning("Please load data first!")
            return
        
        st.header("📈 Data Visualization")
        
        # Sales trend plot
        st.subheader("Monthly Sales Trend")
        fig_trend = self.visualizer.create_interactive_plot(self.monthly_data)
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Sales distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sales Distribution")
            fig_hist = px.histogram(
                self.monthly_data, 
                x='Sales', 
                nbins=20,
                title="Sales Distribution"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            st.subheader("Sales Box Plot")
            fig_box = px.box(
                self.monthly_data, 
                y='Sales',
                title="Sales Box Plot"
            )
            st.plotly_chart(fig_box, use_container_width=True)
        
        # Seasonal decomposition
        st.subheader("Seasonal Decomposition")
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        decomposition = seasonal_decompose(
            self.monthly_data['Sales'], 
            period=12, 
            extrapolate_trend='freq'
        )
        
        # Create decomposition plot
        fig_decomp = go.Figure()
        
        # Original
        fig_decomp.add_trace(go.Scatter(
            x=self.monthly_data['Date'],
            y=self.monthly_data['Sales'],
            name='Original',
            mode='lines'
        ))
        
        # Trend
        fig_decomp.add_trace(go.Scatter(
            x=self.monthly_data['Date'],
            y=decomposition.trend,
            name='Trend',
            mode='lines'
        ))
        
        # Seasonal
        fig_decomp.add_trace(go.Scatter(
            x=self.monthly_data['Date'],
            y=decomposition.seasonal,
            name='Seasonal',
            mode='lines'
        ))
        
        # Residual
        fig_decomp.add_trace(go.Scatter(
            x=self.monthly_data['Date'],
            y=decomposition.resid,
            name='Residual',
            mode='lines'
        ))
        
        fig_decomp.update_layout(
            title="Seasonal Decomposition",
            xaxis_title="Date",
            yaxis_title="Sales",
            height=600
        )
        
        st.plotly_chart(fig_decomp, use_container_width=True)
    
    def train_models(self):
        """Train forecasting models"""
        if not st.session_state.data_loaded:
            st.warning("Please load data first!")
            return
        
        st.header("🤖 Model Training")
        
        # Model selection
        st.subheader("Select Models")
        use_arima = st.checkbox("ARIMA Model", value=True)
        use_prophet = st.checkbox("Prophet Model", value=True)
        
        if st.button("Train Models", type="primary"):
            with st.spinner("Training models..."):
                # Split data
                train_size = int(len(self.monthly_data) * 0.8)
                train_data = self.monthly_data.iloc[:train_size]
                test_data = self.monthly_data.iloc[train_size:]
                
                st.info(f"Training data: {len(train_data)} months, Test data: {len(test_data)} months")
                
                # Initialize model selector
                self.model_selector = ForecastingModelSelector()
                
                # Add selected models
                if use_arima:
                    arima_model = ARIMAModel(order=(1, 1, 1))
                    self.model_selector.add_model('ARIMA', arima_model)
                
                if use_prophet:
                    try:
                        prophet_model = ProphetModel()
                        self.model_selector.add_model('Prophet', prophet_model)
                    except ImportError:
                        st.error("Prophet not available. Please install it with: pip install prophet")
                        return
                
                # Fit models
                self.model_selector.fit_all_models(train_data)
                
                # Evaluate models
                results = self.model_selector.evaluate_all_models(test_data)
                
                # Display results
                st.subheader("Model Performance")
                
                for model_name, metrics in results.items():
                    with st.expander(f"{model_name} Results"):
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("MAE", f"${metrics['MAE']:,.2f}")
                        with col2:
                            st.metric("MSE", f"${metrics['MSE']:,.2f}")
                        with col3:
                            st.metric("RMSE", f"${metrics['RMSE']:,.2f}")
                        with col4:
                            st.metric("MAPE", f"{metrics['MAPE']:.2f}%")
                
                st.session_state.models_trained = True
                st.success("Models trained successfully!")
    
    def generate_forecast(self):
        """Generate sales forecast"""
        if not st.session_state.models_trained:
            st.warning("Please train models first!")
            return
        
        st.header("🔮 Sales Forecast")
        
        # Forecast parameters
        col1, col2 = st.columns(2)
        with col1:
            months = st.slider("Number of months to forecast", 1, 24, 12)
        with col2:
            selected_models = st.multiselect(
                "Select models for forecasting",
                list(self.model_selector.models.keys()),
                default=list(self.model_selector.models.keys())
            )
        
        if st.button("Generate Forecast", type="primary"):
            with st.spinner("Generating forecasts..."):
                # Generate forecasts
                forecasts = {}
                for model_name in selected_models:
                    if model_name in self.model_selector.models:
                        forecast = self.model_selector.models[model_name].forecast(steps=months)
                        forecasts[model_name] = forecast
                
                st.session_state.forecasts = forecasts
                
                # Display results
                st.subheader("Forecast Results")
                
                for model_name, forecast in forecasts.items():
                    with st.expander(f"{model_name} Forecast"):
                        # Calculate future dates
                        last_date = self.monthly_data['Date'].iloc[-1]
                        future_dates = pd.date_range(
                            start=last_date + pd.DateOffset(months=1), 
                            periods=months, 
                            freq='M'
                        )
                        forecast['Date'] = future_dates
                        
                        # Display forecast table
                        forecast_display = forecast.copy()
                        forecast_display['Date'] = forecast_display['Date'].dt.strftime('%Y-%m')
                        forecast_display['Forecast'] = forecast_display['Forecast'].apply(lambda x: f"${x:,.2f}")
                        
                        if 'Lower_Bound' in forecast_display.columns and 'Upper_Bound' in forecast_display.columns:
                            forecast_display['Range'] = forecast_display.apply(
                                lambda row: f"${row['Lower_Bound']:,.2f} - ${row['Upper_Bound']:,.2f}", 
                                axis=1
                            )
                            forecast_display = forecast_display[['Date', 'Forecast', 'Range']]
                        else:
                            forecast_display = forecast_display[['Date', 'Forecast']]
                        
                        st.dataframe(forecast_display, use_container_width=True)
                        
                        # Plot forecast
                        fig = self.visualizer.create_interactive_plot(self.monthly_data, forecast)
                        fig.update_layout(title=f"{model_name} Sales Forecast")
                        st.plotly_chart(fig, use_container_width=True)
    
    def run(self):
        """Run the Streamlit app"""
        st.title("📈 Sales Forecasting System")
        st.markdown("---")
        
        # Sidebar navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox(
            "Choose a page",
            ["Data Loading", "Data Visualization", "Model Training", "Forecast Generation"]
        )
        
        # Page routing
        if page == "Data Loading":
            self.load_data()
        elif page == "Data Visualization":
            self.visualize_data()
        elif page == "Model Training":
            self.train_models()
        elif page == "Forecast Generation":
            self.generate_forecast()
        
        # Sidebar info
        st.sidebar.markdown("---")
        st.sidebar.subheader("System Status")
        
        if st.session_state.data_loaded:
            st.sidebar.success("✅ Data Loaded")
        else:
            st.sidebar.error("❌ Data Not Loaded")
        
        if st.session_state.models_trained:
            st.sidebar.success("✅ Models Trained")
        else:
            st.sidebar.error("❌ Models Not Trained")
        
        # Instructions
        st.sidebar.markdown("---")
        st.sidebar.subheader("Instructions")
        st.sidebar.markdown("""
        1. **Data Loading**: Upload your Excel file
        2. **Data Visualization**: Explore sales trends
        3. **Model Training**: Train forecasting models
        4. **Forecast Generation**: Generate predictions
        """)

def main():
    """Main function to run the Streamlit app"""
    app = StreamlitApp()
    app.run()

if __name__ == "__main__":
    main() 