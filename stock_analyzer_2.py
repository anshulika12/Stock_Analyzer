# -*- coding: utf-8 -*-
"""
Enhanced Stock Analysis App - Streamlit Community Cloud Version
Fixed company list loading, removed lxml dependency, and added search functionality
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
import io

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
    """Load company list based on selected country from GitHub CSV files"""
    try:
        if country == 'India':
            # Load NSE 500 companies from GitHub
            url = 'https://raw.githubusercontent.com/anshulika12/Stock_Analyzer/main/nse500.csv'
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                
                # Read CSV content
                df = pd.read_csv(io.StringIO(response.text))
                
                # Clean up the data - handle different possible column names
                if 'Symbol' in df.columns and 'Company Name' in df.columns:
                    # Add .NS suffix for NSE stocks if not present
                    df['Symbol'] = df['Symbol'].apply(lambda x: x if x.endswith('.NS') else f"{x}.NS")
                    return df[['Symbol', 'Company Name']].set_index('Symbol')
                elif 'symbol' in df.columns and 'company_name' in df.columns:
                    df['symbol'] = df['symbol'].apply(lambda x: x if x.endswith('.NS') else f"{x}.NS")
                    return df[['symbol', 'company_name']].rename(columns={'symbol': 'Symbol', 'company_name': 'Company Name'}).set_index('Symbol')
                else:
                    st.warning("CSV structure different than expected. Using fallback data.")
                    raise Exception("Column structure mismatch")
                    
            except Exception as e:
                st.warning(f"Could not load NSE 500 list from GitHub: {str(e)}. Using fallback data.")
                # Fallback with major Indian companies
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
                    'WIPRO.NS': 'Wipro Limited',
                    'KOTAKBANK.NS': 'Kotak Mahindra Bank Limited',
                    'AXISBANK.NS': 'Axis Bank Limited',
                    'TITAN.NS': 'Titan Company Limited',
                    'NESTLEIND.NS': 'Nestle India Limited',
                    'ULTRACEMCO.NS': 'UltraTech Cement Limited'
                }
                
                df = pd.DataFrame(list(indian_companies.items()), columns=['Symbol', 'Company Name'])
                return df.set_index('Symbol')
            
        else:  # America
            # Load S&P 500 companies from GitHub
            url = 'https://raw.githubusercontent.com/anshulika12/Stock_Analyzer/main/SP500_list.csv'
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                
                # Read CSV content
                df = pd.read_csv(io.StringIO(response.text))
                
                # Clean up the data - handle different possible column names
                if 'Symbol' in df.columns and 'Security' in df.columns:
                    return df[['Symbol', 'Security']].set_index('Symbol')
                elif 'symbol' in df.columns and 'security' in df.columns:
                    return df[['symbol', 'security']].rename(columns={'symbol': 'Symbol', 'security': 'Security'}).set_index('Symbol')
                elif 'Symbol' in df.columns and 'Name' in df.columns:
                    return df[['Symbol', 'Name']].rename(columns={'Name': 'Security'}).set_index('Symbol')
                elif 'Ticker' in df.columns and 'Company' in df.columns:
                    return df[['Ticker', 'Company']].rename(columns={'Ticker': 'Symbol', 'Company': 'Security'}).set_index('Symbol')
                else:
                    # Try to use first two columns if structure is different
                    if len(df.columns) >= 2:
                        df_clean = df.iloc[:, [0, 1]].copy()
                        df_clean.columns = ['Symbol', 'Security']
                        return df_clean.set_index('Symbol')
                    else:
                        raise Exception("Insufficient columns in CSV")
                        
            except Exception as e:
                st.warning(f"Could not load S&P 500 list from GitHub: {str(e)}. Using fallback data.")
                # Fallback with major US companies
                us_companies = {
                    'AAPL': 'Apple Inc.',
                    'MSFT': 'Microsoft Corporation',
                    'GOOGL': 'Alphabet Inc. Class A',
                    'GOOG': 'Alphabet Inc. Class C',
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
                    'BAC': 'Bank of America Corporation',
                    'XOM': 'Exxon Mobil Corporation',
                    'DIS': 'Walt Disney Company',
                    'ADBE': 'Adobe Inc.',
                    'CRM': 'Salesforce Inc.',
                    'NFLX': 'Netflix Inc.',
                    'KO': 'Coca-Cola Company',
                    'PEP': 'PepsiCo Inc.',
                    'TMO': 'Thermo Fisher Scientific Inc.',
                    'COST': 'Costco Wholesale Corporation',
                    'ABBV': 'AbbVie Inc.',
                    'ACN': 'Accenture plc',
                    'MRK': 'Merck & Co. Inc.',
                    'LLY': 'Eli Lilly and Company',
                    'WMT': 'Walmart Inc.'
                }
                
                df = pd.DataFrame(list(us_companies.items()), columns=['Symbol', 'Security'])
                return df.set_index('Symbol')
                
    except Exception as e:
        st.error(f"Error loading company list: {str(e)}")
        # Return minimal fallback list
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

def perform_swot_analysis(df):
    """Perform SWOT analysis based on technical indicators"""
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
            if pd.notna(df['MACD'].iloc[-1]) and pd.notna(df['Signal Line'].iloc[-1]):
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
                strengths.append('Price is above the moving average, indicating a potential uptrend.')

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

        # MACD Histogram Analysis
        if 'MACD Hist' in df.columns:
            if pd.notna(df['MACD Hist'].iloc[-1]) and len(df) > 1:
                if df['MACD Hist'].iloc[-1] > 0 and df['MACD Hist'].diff().iloc[-1] > 0:
                    opportunities.append('Increasing MACD histogram suggests strengthening momentum.')

        # Additional threat analysis
        if 'MACD' in df.columns and 'Signal Line' in df.columns:
            if pd.notna(df['MACD'].iloc[-1]) and pd.notna(df['Signal Line'].iloc[-1]):
                if df['MACD'].iloc[-1] < df['Signal Line'].iloc[-1]:
                    threats.append('Negative MACD crossover indicates downward momentum.')

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
        
        # Display number of companies loaded
        st.success(f"✅ Loaded {len(company_df)} companies")
        
        # Company selection
        st.subheader('📊 Select Company')
        
        # Add search functionality
        search_term = st.text_input("🔍 Search companies:", placeholder="Type to search...")
        
        # Filter companies based on search
        if search_term:
            if country == 'India':
                if 'Company Name' in company_df.columns:
                    filtered_df = company_df[
                        company_df['Company Name'].str.contains(search_term, case=False, na=False) |
                        company_df.index.str.contains(search_term, case=False, na=False)
                    ]
                else:
                    filtered_df = company_df[
                        company_df.index.str.contains(search_term, case=False, na=False)
                    ]
            else:
                if 'Security' in company_df.columns:
                    filtered_df = company_df[
                        company_df['Security'].str.contains(search_term, case=False, na=False) |
                        company_df.index.str.contains(search_term, case=False, na=False)
                    ]
                else:
                    filtered_df = company_df[
                        company_df.index.str.contains(search_term, case=False, na=False)
                    ]
        else:
            filtered_df = company_df
        
        if filtered_df.empty:
            st.warning("No companies found matching your search.")
            filtered_df = company_df
        
        # Company selection dropdown
        if country == 'India':
            if 'Company Name' in filtered_df.columns:
                company_options = filtered_df['Company Name'].tolist()
                company = st.selectbox('Company', company_options)
                ticker = filtered_df[filtered_df['Company Name'] == company].index[0]
            else:
                company_options = filtered_df.index.tolist()
                company = st.selectbox('Company', company_options)
                ticker = company
        else:
            if 'Security' in filtered_df.columns:
                company_options = filtered_df['Security'].tolist()
                company = st.selectbox('Company', company_options)
                ticker = filtered_df[filtered_df['Security'] == company].index[0]
            else:
                company_options = filtered_df.index.tolist()
                company = st.selectbox('Company', company_options)
                ticker = company
        
        # Display selected ticker
        st.info(f"Selected ticker: **{ticker}**")
        
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
        st.subheader(f'📈 Candlestick Chart for {ticker}')
        
        # Candlestick chart
        fig_candle = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Candlesticks'
        )])
        
        fig_candle.update_layout(
            title=f"{company} Stock Price",
            yaxis_title=f'Price ({currency})',
            xaxis_title='Date',
            xaxis_rangeslider_visible=False,
            height=500,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        
        st.plotly_chart(fig_candle, use_container_width=True)
    
    with col2:
        st.subheader(f'ℹ️ Company Information for {ticker}')
        
        info_items = {
            'Long Name': info.get('longName', 'N/A'),
            'Sector': info.get('sector', 'N/A'),
            'Industry': info.get('industry', 'N/A'),
            'Full Time Employees': info.get('fullTimeEmployees', 'N/A'),
            'City': info.get('city', 'N/A'),
            'State': info.get('state', 'N/A'),
            'Country': info.get('country', 'N/A'),
            'Website': info.get('website', 'N/A'),
            'Market Cap': info.get('marketCap', 'N/A'),
            'Currency': info.get('currency', 'N/A')
        }
        
        # Format market cap
        if info_items['Market Cap'] != 'N/A' and info_items['Market Cap'] is not None:
            market_cap = info_items['Market Cap']
            if isinstance(market_cap, (int, float)):
                if market_cap >= 1e12:
                    info_items['Market Cap'] = f"${market_cap/1e12:.2f}T"
                elif market_cap >= 1e9:
                    info_items['Market Cap'] = f"${market_cap/1e9:.2f}B"
                elif market_cap >= 1e6:
                    info_items['Market Cap'] = f"${market_cap/1e6:.2f}M"
        
        # Format employees
        if info_items['Full Time Employees'] != 'N/A' and info_items['Full Time Employees'] is not None:
            employees = info_items['Full Time Employees']
            if isinstance(employees, (int, float)):
                info_items['Full Time Employees'] = f"{employees:,}"
        
        info_df = pd.DataFrame.from_dict(info_items, orient='index', columns=['Value'])
        info_df['Value'] = info_df['Value'].astype(str)
        st.dataframe(info_df)
    
    # Technical Analysis Section
    if TA_AVAILABLE:
        st.markdown("---")
        st.header("🔍 Technical Analysis")
        
        # Calculate indicators with proper data handling
        try:
            # Fix: Convert to pandas Series explicitly and ensure proper data handling
            close_values = df['Close'].values
            if close_values.ndim > 1:
                close_values = close_values.flatten()
            close_series = pd.Series(close_values, index=df.index, name='Close')
            
            # MACD Analysis
            st.subheader('📊 Moving Average Convergence Divergence (MACD)')
            
            if st.button("Understand Me", key='macd'):
                st.info("""
                **Understanding MACD:**

                - **MACD Line:** Difference between the 12-period and 26-period EMAs.
                - **Signal Line:** 9-period EMA of the MACD Line.
                - **Histogram:** Visual representation of the difference between the MACD Line and Signal Line.
                - **Bullish Crossover (Green Triangle):** MACD Line crosses above the Signal Line, indicating potential upward momentum.
                - **Bearish Crossover (Red Triangle):** MACD Line crosses below the Signal Line, indicating potential downward momentum.
                """)
            
            macd = MACD(close=close_series)
            df['MACD'] = macd.macd()
            df['Signal Line'] = macd.macd_signal()
            df['MACD Hist'] = macd.macd_diff()

            # Calculate crossovers for triangle markers
            df['Crossover'] = np.where(df['MACD'] > df['Signal Line'], 1, 0)
            df['Signal'] = df['Crossover'].diff()

            bullish_cross = df[df['Signal'] == 1]
            bearish_cross = df[df['Signal'] == -1]

            # Determine current trend
            recent_data = df.tail(60)
            recent_macd = recent_data['MACD']
            recent_signal = recent_data['Signal Line']

            if len(recent_macd) > 0 and len(recent_signal) > 0:
                if recent_macd.iloc[-1] > recent_signal.iloc[-1]:
                    trend = 'Bullish Trend Detected'
                    trend_color = 'green'
                else:
                    trend = 'Bearish Trend Detected'
                    trend_color = 'red'
                
                st.markdown(f"<h3 style='color: {trend_color};'>{trend}</h3>", unsafe_allow_html=True)
            
            # MACD Chart with triangles
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(
                x=df.index, y=df['MACD'],
                mode='lines', name='MACD Line'
            ))
            fig_macd.add_trace(go.Scatter(
                x=df.index, y=df['Signal Line'],
                mode='lines', name='Signal Line'
            ))
            fig_macd.add_trace(go.Bar(
                x=df.index, y=df['MACD Hist'],
                name='Histogram'
            ))
            # Add triangle markers for crossovers
            fig_macd.add_trace(go.Scatter(
                x=bullish_cross.index, y=bullish_cross['MACD'],
                mode='markers', name='Bullish Crossover',
                marker=dict(color='green', size=8, symbol='triangle-up')
            ))
            fig_macd.add_trace(go.Scatter(
                x=bearish_cross.index, y=bearish_cross['MACD'],
                mode='markers', name='Bearish Crossover',
                marker=dict(color='red', size=8, symbol='triangle-down')
            ))
            fig_macd.update_layout(
                yaxis_title='MACD',
                xaxis_title='Date',
                xaxis_rangeslider_visible=False,
                height=400
            )
            st.plotly_chart(fig_macd, use_container_width=True)
            
            # RSI Analysis
            st.subheader('⚡ Relative Strength Index (RSI)')
            
            if st.button("Understand Me", key='rsi'):
                st.info("""
                **Understanding RSI:**

                - **RSI (Relative Strength Index):** Measures the speed and change of price movements.
                - **Overbought (>70):** May indicate a price reversal downward.
                - **Oversold (<30):** May indicate a price reversal upward.
                """)
            
            indicator_rsi = RSIIndicator(close=close_series)
            df['RSI'] = indicator_rsi.rsi()

            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(
                x=df.index, y=df['RSI'],
                mode='lines', name='RSI'
            ))
            fig_rsi.add_hline(y=70, line_dash='dash', line_color='red')
            fig_rsi.add_hline(y=30, line_dash='dash', line_color='green')
            fig_rsi.update_layout(
                yaxis_title='RSI',
                xaxis_title='Date',
                xaxis_rangeslider_visible=False,
                yaxis=dict(range=[0, 100]),
                height=400
            )
            st.plotly_chart(fig_rsi, use_container_width=True)
            
            # Bollinger Bands Analysis
            st.subheader('📏 Bollinger Bands')
            
            if st.button("Understand Me", key='bb'):
                st.info("""
                **Understanding Bollinger Bands:**

                - **Bollinger Bands:** Consist of a middle band (moving average) and upper/lower bands (standard deviations).
                - **Upper Band:** Price crossing above may indicate overbought conditions.
                - **Lower Band:** Price crossing below may indicate oversold conditions.
                - **Overbought (Red Triangle):** Potential price reversal downward.
                - **Oversold (Green Triangle):** Potential price reversal upward.
                """)
            
            indicator_bb = BollingerBands(close=close_series)
            df['bb_upper'] = indicator_bb.bollinger_hband()
            df['bb_lower'] = indicator_bb.bollinger_lband()
            df['bb_middle'] = indicator_bb.bollinger_mavg()

            # Calculate overbought/oversold positions for triangle markers
            df['Position'] = None
            df['Position'] = np.where(df['Close'] < df['bb_lower'], 'Oversold', df['Position'])
            df['Position'] = np.where(df['Close'] > df['bb_upper'], 'Overbought', df['Position'])

            oversold_points = df[df['Position'] == 'Oversold']
            overbought_points = df[df['Position'] == 'Overbought']

            # Bollinger Bands Chart with triangles
            fig_bb = go.Figure()
            fig_bb.add_trace(go.Scatter(
                x=df.index, y=df['Close'],
                mode='lines', name='Close Price'
            ))
            fig_bb.add_trace(go.Scatter(
                x=df.index, y=df['bb_upper'],
                line=dict(color='rgba(173,216,230,0.2)'), name='Upper Band'
            ))
            fig_bb.add_trace(go.Scatter(
                x=df.index, y=df['bb_middle'],
                line=dict(color='rgba(0,0,0,0.5)'), name='Middle Band'
            ))
            fig_bb.add_trace(go.Scatter(
                x=df.index, y=df['bb_lower'],
                line=dict(color='rgba(173,216,230,0.2)'), name='Lower Band',
                fill='tonexty', fillcolor='rgba(173,216,230,0.1)'
            ))
            # Add triangle markers for overbought/oversold
            fig_bb.add_trace(go.Scatter(
                x=oversold_points.index, y=oversold_points['Close'],
                mode='markers', name='Oversold',
                marker=dict(color='green', size=8, symbol='triangle-up')
            ))
            fig_bb.add_trace(go.Scatter(
                x=overbought_points.index, y=overbought_points['Close'],
                mode='markers', name='Overbought',
                marker=dict(color='red', size=8, symbol='triangle-down')
            ))
            fig_bb.update_layout(
                yaxis_title=f'Price ({currency})',
                xaxis_title='Date',
                xaxis_rangeslider_visible=False,
                height=400
            )
            st.plotly_chart(fig_bb, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error calculating technical indicators: {str(e)}")
    
    # Volume Chart
    st.subheader('📊 Volume Chart')
    fig_volume = go.Figure()
    fig_volume.add_trace(go.Bar(
        x=df.index, y=df['Volume'],
        name='Volume'
    ))
    fig_volume.update_layout(
        yaxis_title='Volume',
        xaxis_title='Date',
        xaxis_rangeslider_visible=False,
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
                st.markdown(f"- {s}")
        else:
            st.info("No specific strengths identified from current data.")
        
        st.subheader("🎯 Opportunities")
        if opportunities:
            for o in opportunities:
                st.markdown(f"- {o}")
        else:
            st.info("No specific opportunities identified from current data.")
    
    with col2:
        st.subheader("⚠️ Weaknesses")
        if weaknesses:
            for w in weaknesses:
                st.markdown(f"- {w}")
        else:
            st.info("No specific weaknesses identified from current data.")
        
        st.subheader("🚨 Threats")
        if threats:
            for t in threats:
                st.markdown(f"- {t}")
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