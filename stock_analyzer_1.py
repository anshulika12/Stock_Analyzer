# -*- coding: utf-8 -*-
"""
Enhanced Stock Analysis App - Streamlit Community Cloud Version
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import yfinance as yf
from datetime import datetime, timedelta
import time
import warnings
import requests

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Try importing technical analysis libraries with error handling
try:
    from ta.volatility import BollingerBands
    from ta.momentum import RSIIndicator
    from ta.trend import MACD
    TA_AVAILABLE = True
except ImportError:
    st.error("Technical analysis library 'ta' is not available. Please install it using: pip install ta")
    TA_AVAILABLE = False

# Set page configuration
st.set_page_config(
    page_title="Enhanced Stock Analysis App",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_data
def load_data(ticker, start, end):
    """Load stock data from Yahoo Finance with retry logic"""
    max_retries = 3
    base_delay = 2
    
    for attempt in range(max_retries):
        try:
            # Validate ticker first
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if not info or len(info) < 5:  # More robust check
                st.error(f"Invalid ticker {ticker}. Please check the ticker symbol.")
                return pd.DataFrame()
            
            # Download data
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
            
            if df.empty:
                st.error(f"No data found for ticker {ticker}. Please check the ticker symbol or date range.")
                return df
            
            # Clean up the data
            df.index = pd.to_datetime(df.index)
            
            # Handle multi-level columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            
            # Ensure we have the required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {missing_columns}")
                return pd.DataFrame()
            
            time.sleep(1)  # Rate limiting
            return df
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                delay = base_delay * (2 ** attempt)
                st.warning(f"Rate limit reached for {ticker}. Retrying in {delay} seconds...")
                time.sleep(delay)
                continue
            else:
                st.error(f"HTTP Error fetching data for {ticker}: {str(e)}")
                return pd.DataFrame()
                
        except Exception as e:
            if "JSONDecodeError" in str(e) or "RemoteDataError" in str(e):
                st.warning(f"Attempt {attempt + 1}: Failed to fetch data for {ticker}. Retrying...")
                time.sleep(base_delay * (2 ** attempt))
                continue
            else:
                st.error(f"Error fetching data for {ticker}: {str(e)}")
                return pd.DataFrame()
    
    st.error(f"Failed to fetch data for {ticker} after {max_retries} attempts.")
    return pd.DataFrame()

@st.cache_data
def load_company_list(country):
    """Load company list based on selected country"""
    try:
        if country == 'India':
            # For India, we'll use a predefined list since CSV might not be available
            indian_companies = {
                'RELIANCE.NS': 'Reliance Industries Limited',
                'TCS.NS': 'Tata Consultancy Services Limited',
                'HDFCBANK.NS': 'HDFC Bank Limited',
                'BHARTIARTL.NS': 'Bharti Airtel Limited',
                'ICICIBANK.NS': 'ICICI Bank Limited',
                'SBIN.NS': 'State Bank of India',
                'INFY.NS': 'Infosys Limited',
                'ITC.NS': 'ITC Limited',
                'HINDUNILVR.NS': 'Hindustan Unilever Limited',
                'LT.NS': 'Larsen & Toubro Limited',
                'HCLTECH.NS': 'HCL Technologies Limited',
                'MARUTI.NS': 'Maruti Suzuki India Limited',
                'BAJFINANCE.NS': 'Bajaj Finance Limited',
                'ASIANPAINT.NS': 'Asian Paints Limited',
                'WIPRO.NS': 'Wipro Limited'
            }
            
            df = pd.DataFrame(list(indian_companies.items()), columns=['Symbol', 'Company Name'])
            return df.set_index('Symbol')
            
        else:  # America
            # Load S&P 500 companies from Wikipedia
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            tables = pd.read_html(url)
            df = tables[0]
            
            # Clean up the data
            if 'Symbol' in df.columns and 'Security' in df.columns:
                return df[['Symbol', 'Security']].set_index('Symbol')
            else:
                # Fallback with major US companies
                us_companies = {
                    'AAPL': 'Apple Inc.',
                    'MSFT': 'Microsoft Corporation',
                    'GOOGL': 'Alphabet Inc.',
                    'AMZN': 'Amazon.com Inc.',
                    'TSLA': 'Tesla Inc.',
                    'META': 'Meta Platforms Inc.',
                    'NVDA': 'NVIDIA Corporation',
                    'JPM': 'JPMorgan Chase & Co.',
                    'JNJ': 'Johnson & Johnson',
                    'V': 'Visa Inc.',
                    'PG': 'Procter & Gamble Company',
                    'UNH': 'UnitedHealth Group Incorporated',
                    'HD': 'Home Depot Inc.',
                    'MA': 'Mastercard Incorporated',
                    'BAC': 'Bank of America Corporation'
                }
                
                df = pd.DataFrame(list(us_companies.items()), columns=['Symbol', 'Security'])
                return df.set_index('Symbol')
                
    except Exception as e:
        st.error(f"Error loading company list: {str(e)}")
        # Return fallback list
        fallback_companies = {
            'AAPL': 'Apple Inc.',
            'MSFT': 'Microsoft Corporation',
            'GOOGL': 'Alphabet Inc.',
            'TSLA': 'Tesla Inc.',
            'AMZN': 'Amazon.com Inc.'
        }
        df = pd.DataFrame(list(fallback_companies.items()), columns=['Symbol', 'Company Name'])
        return df.set_index('Symbol')

def get_stock_info(ticker):
    """Get stock information with error handling"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        time.sleep(0.5)  # Rate limiting
        return info if info else {}
    except Exception as e:
        st.warning(f"Could not fetch detailed info for {ticker}: {str(e)}")
        return {}

def calculate_technical_indicators(df):
    """Calculate technical indicators with proper error handling"""
    if not TA_AVAILABLE:
        st.warning("Technical analysis features are not available. Install 'ta' library for full functionality.")
        return df
    
    try:
        # Ensure we have proper 1D data
        close_values = df['Close'].values
        if close_values.ndim > 1:
            close_values = close_values.flatten()
        
        close_series = pd.Series(close_values, index=df.index, name='Close')
        
        # MACD
        macd = MACD(close=close_series)
        df['MACD'] = macd.macd()
        df['Signal Line'] = macd.macd_signal()
        df['MACD Hist'] = macd.macd_diff()
        
        # RSI
        indicator_rsi = RSIIndicator(close=close_series)
        df['RSI'] = indicator_rsi.rsi()
        
        # Bollinger Bands
        indicator_bb = BollingerBands(close=close_series)
        df['bb_upper'] = indicator_bb.bollinger_hband()
        df['bb_lower'] = indicator_bb.bollinger_lband()
        df['bb_middle'] = indicator_bb.bollinger_mavg()
        
        return df
        
    except Exception as e:
        st.error(f"Error calculating technical indicators: {str(e)}")
        return df

def perform_swot_analysis(df):
    """Perform SWOT analysis based on technical indicators"""
    if not TA_AVAILABLE or 'MACD' not in df.columns:
        return [], [], [], []
    
    strengths = []
    weaknesses = []
    opportunities = []
    threats = []
    
    try:
        # Check if we have enough data
        if len(df) < 20:
            return ["Insufficient data for analysis"], [], [], []
        
        # MACD Analysis
        if 'MACD' in df.columns and 'Signal Line' in df.columns:
            if df['MACD'].iloc[-1] > df['Signal Line'].iloc[-1]:
                strengths.append('Positive MACD crossover indicates upward momentum.')
        
        # RSI Analysis
        if 'RSI' in df.columns:
            current_rsi = df['RSI'].iloc[-1]
            if pd.notna(current_rsi):
                if 30 < current_rsi < 70:
                    strengths.append('RSI is in a neutral zone, indicating stable momentum.')
                elif current_rsi > 70:
                    weaknesses.append('RSI indicates overbought conditions.')
                elif current_rsi < 30:
                    threats.append('RSI indicates oversold conditions.')
        
        # Price vs Moving Average
        if 'bb_middle' in df.columns:
            if df['Close'].iloc[-1] > df['bb_middle'].iloc[-1]:
                strengths.append('Price is above the moving average, indicating potential uptrend.')
        
        # Volume Analysis
        if 'Volume' in df.columns and len(df) >= 20:
            recent_volume = df['Volume'].iloc[-1]
            avg_volume = df['Volume'].rolling(window=20).mean().iloc[-1]
            if pd.notna(recent_volume) and pd.notna(avg_volume):
                if recent_volume < avg_volume:
                    weaknesses.append('Trading volume is below average, indicating low market interest.')
        
        # Bollinger Bands Analysis
        if 'bb_lower' in df.columns and 'bb_upper' in df.columns:
            current_price = df['Close'].iloc[-1]
            if current_price < df['bb_lower'].iloc[-1]:
                opportunities.append('Price is below the lower Bollinger Band, potential oversold opportunity.')
                threats.append('Price is below the lower Bollinger Band, indicating strong downward volatility.')
        
    except Exception as e:
        st.warning(f"Error in SWOT analysis: {str(e)}")
    
    return strengths, weaknesses, opportunities, threats

def main():
    """Main application function"""
    st.title('📈 Enhanced Stock Analysis App')
    st.markdown("---")
    
    # Sidebar for user inputs
    with st.sidebar:
        st.header('🔧 Configuration')
        
        # Country selection
        country = st.selectbox(
            'Select Market',
            ['America', 'India'],
            help="Choose the stock market to analyze"
        )
        
        # Load company list
        with st.spinner('Loading company list...'):
            company_df = load_company_list(country)
        
        if company_df.empty:
            st.error("Failed to load company list. Please try again.")
            return
        
        # Company selection
        st.subheader('📊 Select Company')
        if country == 'India':
            if 'Company Name' in company_df.columns:
                company = st.selectbox('Company', company_df['Company Name'].tolist())
                ticker = company_df[company_df['Company Name'] == company].index[0]
            else:
                # Fallback for different column structure
                company = st.selectbox('Company', company_df.index.tolist())
                ticker = company
        else:
            if 'Security' in company_df.columns:
                company = st.selectbox('Company', company_df['Security'].tolist())
                ticker = company_df[company_df['Security'] == company].index[0]
            else:
                # Fallback for different column structure
                company = st.selectbox('Company', company_df.index.tolist())
                ticker = company
        
        # Date selection
        st.subheader('📅 Date Range')
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                'Start Date',
                datetime.today() - timedelta(days=365*2),
                max_value=datetime.today()
            )
        with col2:
            end_date = st.date_input(
                'End Date',
                min_value=start_date,
                max_value=datetime.today(),
                value=datetime.today()
            )
        
        if start_date >= end_date:
            st.error('End date must be after start date.')
            return
    
    # Main content area
    with st.spinner(f'Loading data for {ticker}...'):
        df = load_data(ticker, start_date, end_date)
    
    if df.empty:
        st.error("No data available. Please try a different ticker or date range.")
        return
    
    # Get company information
    info = get_stock_info(ticker)
    currency = info.get('currency', 'USD')
    
    # Display current price and basic info
    current_price = df['Close'].iloc[-1]
    prev_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
    
    # Price display
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label=f"Current Price ({currency})",
            value=f"{current_price:.2f}",
            delta=f"{price_change:.2f} ({price_change_pct:.2f}%)"
        )
    with col2:
        if len(df) >= 52:
            high_52w = df['High'].tail(252).max()
            low_52w = df['Low'].tail(252).min()
            st.metric("52W High", f"{high_52w:.2f}")
            st.metric("52W Low", f"{low_52w:.2f}")
    with col3:
        avg_volume = df['Volume'].tail(20).mean()
        st.metric("Avg Volume (20D)", f"{avg_volume:,.0f}")
    
    # Main chart and company info
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader(f'📈 Price Chart - {ticker}')
        
        # Candlestick chart
        fig_candle = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        )])
        
        fig_candle.update_layout(
            title=f"{company} Stock Price",
            yaxis_title=f'Price ({currency})',
            xaxis_title='Date',
            xaxis_rangeslider_visible=False,
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig_candle, use_container_width=True)
    
    with col2:
        st.subheader('ℹ️ Company Info')
        
        info_items = {
            'Name': info.get('longName', 'N/A'),
            'Sector': info.get('sector', 'N/A'),
            'Industry': info.get('industry', 'N/A'),
            'Country': info.get('country', 'N/A'),
            'Employees': info.get('fullTimeEmployees', 'N/A'),
            'Market Cap': info.get('marketCap', 'N/A')
        }
        
        for key, value in info_items.items():
            if value != 'N/A' and value is not None:
                if key == 'Market Cap' and isinstance(value, (int, float)):
                    if value >= 1e12:
                        value = f"${value/1e12:.2f}T"
                    elif value >= 1e9:
                        value = f"${value/1e9:.2f}B"
                    elif value >= 1e6:
                        value = f"${value/1e6:.2f}M"
                elif key == 'Employees' and isinstance(value, (int, float)):
                    value = f"{value:,}"
            
            st.text(f"{key}: {value}")
    
    # Technical Analysis Section
    if TA_AVAILABLE:
        st.markdown("---")
        st.header("🔍 Technical Analysis")
        
        # Calculate indicators
        df = calculate_technical_indicators(df)
        
        if 'MACD' in df.columns:
            # MACD Section
            st.subheader('📊 MACD (Moving Average Convergence Divergence)')
            
            with st.expander("📖 Understanding MACD"):
                st.markdown("""
                **MACD Components:**
                - **MACD Line:** Difference between 12-period and 26-period EMAs
                - **Signal Line:** 9-period EMA of the MACD Line
                - **Histogram:** Difference between MACD Line and Signal Line
                - **Bullish Signal:** MACD crosses above Signal Line (🔺)
                - **Bearish Signal:** MACD crosses below Signal Line (🔻)
                """)
            
            # MACD trend analysis
            if pd.notna(df['MACD'].iloc[-1]) and pd.notna(df['Signal Line'].iloc[-1]):
                if df['MACD'].iloc[-1] > df['Signal Line'].iloc[-1]:
                    st.success("🔺 **Bullish Trend Detected**")
                else:
                    st.error("🔻 **Bearish Trend Detected**")
            
            # MACD Chart
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD Line', line=dict(color='blue')))
            fig_macd.add_trace(go.Scatter(x=df.index, y=df['Signal Line'], name='Signal Line', line=dict(color='red')))
            fig_macd.add_trace(go.Bar(x=df.index, y=df['MACD Hist'], name='Histogram', opacity=0.7))
            
            fig_macd.update_layout(
                title="MACD Indicator",
                yaxis_title='MACD',
                xaxis_title='Date',
                height=400
            )
            st.plotly_chart(fig_macd, use_container_width=True)
        
        if 'RSI' in df.columns:
            # RSI Section
            st.subheader('⚡ RSI (Relative Strength Index)')
            
            with st.expander("📖 Understanding RSI"):
                st.markdown("""
                **RSI Interpretation:**
                - **Range:** 0 to 100
                - **Overbought:** RSI > 70 (potential sell signal)
                - **Oversold:** RSI < 30 (potential buy signal)
                - **Neutral:** 30 < RSI < 70 (balanced momentum)
                """)
            
            # RSI Chart
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')))
            fig_rsi.add_hline(y=70, line_dash='dash', line_color='red', annotation_text="Overbought")
            fig_rsi.add_hline(y=30, line_dash='dash', line_color='green', annotation_text="Oversold")
            
            fig_rsi.update_layout(
                title="RSI Indicator",
                yaxis_title='RSI',
                xaxis_title='Date',
                yaxis=dict(range=[0, 100]),
                height=400
            )
            st.plotly_chart(fig_rsi, use_container_width=True)
        
        if 'bb_upper' in df.columns:
            # Bollinger Bands Section
            st.subheader('📏 Bollinger Bands')
            
            with st.expander("📖 Understanding Bollinger Bands"):
                st.markdown("""
                **Bollinger Bands Components:**
                - **Upper Band:** Moving average + (2 × standard deviation)
                - **Middle Band:** 20-period moving average
                - **Lower Band:** Moving average - (2 × standard deviation)
                - **Price touching upper band:** Potential overbought condition
                - **Price touching lower band:** Potential oversold condition
                """)
            
            # Bollinger Bands Chart
            fig_bb = go.Figure()
            fig_bb.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price', line=dict(color='black')))
            fig_bb.add_trace(go.Scatter(x=df.index, y=df['bb_upper'], name='Upper Band', line=dict(color='red', dash='dash')))
            fig_bb.add_trace(go.Scatter(x=df.index, y=df['bb_middle'], name='Middle Band', line=dict(color='blue')))
            fig_bb.add_trace(go.Scatter(x=df.index, y=df['bb_lower'], name='Lower Band', line=dict(color='green', dash='dash'), fill='tonexty', fillcolor='rgba(0,100,80,0.1)'))
            
            fig_bb.update_layout(
                title="Bollinger Bands",
                yaxis_title=f'Price ({currency})',
                xaxis_title='Date',
                height=400
            )
            st.plotly_chart(fig_bb, use_container_width=True)
    
    # Volume Chart
    st.subheader('📊 Volume Analysis')
    fig_volume = go.Figure()
    fig_volume.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='lightblue'))
    fig_volume.update_layout(
        title="Trading Volume",
        yaxis_title='Volume',
        xaxis_title='Date',
        height=300
    )
    st.plotly_chart(fig_volume, use_container_width=True)
    
    # SWOT Analysis
    st.markdown("---")
    st.header("🎯 SWOT Analysis")
    
    strengths, weaknesses, opportunities, threats = perform_swot_analysis(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💪 Strengths")
        if strengths:
            for s in strengths:
                st.success(f"✅ {s}")
        else:
            st.info("No specific strengths identified from current data.")
        
        st.subheader("🎯 Opportunities")
        if opportunities:
            for o in opportunities:
                st.info(f"🔍 {o}")
        else:
            st.info("No specific opportunities identified from current data.")
    
    with col2:
        st.subheader("⚠️ Weaknesses")
        if weaknesses:
            for w in weaknesses:
                st.warning(f"⚡ {w}")
        else:
            st.info("No specific weaknesses identified from current data.")
        
        st.subheader("🚨 Threats")
        if threats:
            for t in threats:
                st.error(f"🔴 {t}")
        else:
            st.info("No specific threats identified from current data.")
    
    # Raw Data Section
    st.markdown("---")
    st.header("📋 Raw Data")
    
    # Display summary statistics
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Summary Statistics")
        st.dataframe(df[['Open', 'High', 'Low', 'Close', 'Volume']].describe())
    
    with col2:
        st.subheader("📈 Recent Data")
        st.dataframe(df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(10))
    
    # Download option
    csv = df.to_csv()
    st.download_button(
        label="📥 Download Data as CSV",
        data=csv,
        file_name=f"{ticker}_stock_data.csv",
        mime="text/csv"
    )

if __name__ == '__main__':
    main()
