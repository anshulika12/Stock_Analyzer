# -*- coding: utf-8 -*-
"""
Enhanced Stock Analysis App with Country Selection and Full Features
"""
import subprocess
import sys
import pkg_resources
import time
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)  # Temporary suppression of RuntimeWarning

def install_modules():
    REQUIRED_PACKAGES = [
        "streamlit>=1.39.0",
        "pandas>=2.2.3",
        "numpy>=1.26.4",
        "matplotlib>=3.8.4",
        "Pillow>=10.3.0",
        "plotly>=5.22.0",
        "ta>=0.11.0",
        "yfinance>=0.2.62",
        "scipy>=1.13.1",
        "python-dateutil>=2.9.0",
        "seaborn>=0.13.2",
        "scikit-learn>=1.2.2",
        "FundamentalAnalysis>=0.3.1"
    ]
    
    for package in REQUIRED_PACKAGES:
        try:
            dist = pkg_resources.get_distribution(package.split(">=")[0])
            print(f"{dist.key} ({dist.version}) is already installed.")
        except pkg_resources.DistributionNotFound:
            print(f"{package} is not installed. Installing now...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except Exception as e:
            print(f"An error occurred while checking/installing {package}: {e}")
    
    if sys.platform != 'win32':
        try:
            subprocess.check_output("dos2unix --version", shell=True)
            print("dos2unix is already installed.")
        except subprocess.CalledProcessError:
            print("dos2unix is not installed. Installing now...")
            subprocess.check_call(["brew", "install", "dos2unix"])
        except FileNotFoundError:
            print("Homebrew is not installed. Please install it to proceed with 'dos2unix' installation.")

install_modules()

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import yfinance as yf
from datetime import datetime, timedelta
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.trend import MACD
import requests

# Set page configuration
st.set_page_config(
    page_title="Enhanced Stock Analysis App",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_data
def load_data(ticker, start, end):
    max_retries = 3
    base_delay = 2
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            if not info or 'symbol' not in info:
                st.error(f"Invalid ticker {ticker}. Please check the ticker symbol.")
                return pd.DataFrame()
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
            if df.empty:
                st.error(f"No data found for ticker {ticker}. Please check the ticker symbol or date range.")
                return df
            df.index = pd.to_datetime(df.index)
            
            # Ensure columns are properly flattened if they have multi-level structure
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            
            time.sleep(1)
            return df
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                delay = base_delay * (2 ** attempt)
                st.warning(f"Rate limit reached for {ticker}. Retrying in {delay} seconds...")
                time.sleep(delay)
                continue
            else:
                st.error(f"Error fetching data for {ticker}: {str(e)}")
                return pd.DataFrame()
        except Exception as e:
            if "JSONDecodeError" in str(e):
                st.warning(f"Attempt {attempt + 1}: Failed to fetch data for {ticker}. Retrying...")
                time.sleep(base_delay * (2 ** attempt))
                continue
            else:
                st.error(f"Error fetching data for {ticker}: {str(e)}")
                return pd.DataFrame()
    st.error(f"Failed to fetch data for {ticker} after {max_retries} attempts due to rate limits.")
    return pd.DataFrame()

@st.cache_data
def load_company_list(country):
    try:
        if country == 'India':
            df = pd.read_csv('nse500.csv')
            if 'Symbol' not in df.columns or 'Company Name' not in df.columns:
                st.error("nse500.csv must contain 'Symbol' and 'Company Name' columns.")
                return pd.DataFrame()
            df['Symbol'] = df['Symbol'].astype(str) + '.NS'
            return df[['Symbol', 'Company Name']].set_index('Symbol')
        else:  # America
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            table = pd.read_html(url)
            df = table[0]
            return df[['Symbol', 'Security']].set_index('Symbol')
    except FileNotFoundError:
        st.error("nse500.csv file not found. Please ensure the file exists in the working directory.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading company list: {str(e)}")
        return pd.DataFrame()

def get_stock_info(ticker):
    max_retries = 3
    base_delay = 2
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            time.sleep(1)
            return info
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                delay = base_delay * (2 ** attempt)
                st.warning(f"Rate limit reached for {ticker} info. Retrying in {delay} seconds...")
                time.sleep(delay)
                continue
            else:
                st.error(f"Error fetching info for {ticker}: {str(e)}")
                return {}
        except Exception as e:
            st.error(f"Error fetching info for {ticker}: {str(e)}")
            return {}
    st.error(f"Failed to fetch info for {ticker} after {max_retries} attempts due to rate limits.")
    return {}

def main():
    st.title('Enhanced Stock Analysis App')

    country = st.sidebar.selectbox('Select Country', ['America', 'India'])

    company_df = load_company_list(country)
    if company_df.empty:
        st.error("Failed to load company list. Please check the data source.")
        return

    st.sidebar.header('Select a Company')
    if country == 'India':
        company = st.sidebar.selectbox('Company', company_df['Company Name'])
        ticker = company_df[company_df['Company Name'] == company].index[0]
    else:
        company = st.sidebar.selectbox('Company', company_df['Security'])
        ticker = company_df[company_df['Security'] == company].index[0]

    start_date = st.sidebar.date_input('Start Date', datetime.today() - timedelta(days=365*2))
    end_date = st.sidebar.date_input('End Date', min_value=start_date, max_value=datetime.today(), value=datetime.today())

    if start_date > end_date:
        st.sidebar.error('Error: End date must fall after start date.')
        return

    df = load_data(ticker, start_date, end_date)
    if df.empty:
        return

    info = get_stock_info(ticker)

    col1, col2 = st.columns([3, 1], gap="small")

    with col1:
        st.subheader(f'Candlestick Chart for {ticker}')
        fig_candle = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Candlesticks'
        )])
        currency = info.get('currency', 'USD')
        fig_candle.update_layout(
            yaxis_title=f'Price ({currency})',
            xaxis_title='Date',
            xaxis_rangeslider_visible=False,
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_candle, use_container_width=True)
    with col2:
        st.subheader(f'Company Information for {ticker}')
        info_items = ['longName', 'sector', 'industry', 'fullTimeEmployees',
                      'city', 'state', 'country', 'website', 'marketCap', 'currency']
        info_dict = {item: str(info.get(item, 'N/A')) for item in info_items}
        info_df = pd.DataFrame.from_dict(info_dict, orient='index', columns=['Value'])
        info_df['Value'] = info_df['Value'].astype(str)
        st.dataframe(info_df)

    st.subheader('Moving Average Convergence Divergence (MACD)')
    if st.button("Understand Me", key='macd'):
        st.info("""
        **Understanding MACD:**

        - **MACD Line:** Difference between the 12-period and 26-period EMAs.
        - **Signal Line:** 9-period EMA of the MACD Line.
        - **Histogram:** Visual representation of the difference between the MACD Line and Signal Line.
        - **Bullish Crossover (Green Triangle):** MACD Line crosses above the Signal Line, indicating potential upward momentum.
        - **Bearish Crossover (Red Triangle):** MACD Line crosses below the Signal Line, indicating potential downward momentum.
        """)
    
    # Fix: Convert to pandas Series explicitly and ensure proper data handling
    # Flatten the array to ensure it's 1D
    close_values = df['Close'].values
    if close_values.ndim > 1:
        close_values = close_values.flatten()
    close_series = pd.Series(close_values, index=df.index, name='Close')
    
    macd = MACD(close=close_series)
    df['MACD'] = macd.macd()
    df['Signal Line'] = macd.macd_signal()
    df['MACD Hist'] = macd.macd_diff()

    df['Crossover'] = np.where(df['MACD'] > df['Signal Line'], 1, 0)
    df['Signal'] = df['Crossover'].diff()

    bullish_cross = df[df['Signal'] == 1]
    bearish_cross = df[df['Signal'] == -1]

    recent_data = df.tail(60)
    recent_macd = recent_data['MACD']
    recent_signal = recent_data['Signal Line']

    if recent_macd.iloc[-1] > recent_signal.iloc[-1]:
        trend = 'Bullish Trend Detected'
        trend_color = 'green'
    else:
        trend = 'Bearish Trend Detected'
        trend_color = 'red'

    st.markdown(f"<h3 style='color: {trend_color};'>{trend}</h3>", unsafe_allow_html=True)

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
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig_macd, use_container_width=True)

    st.subheader('Relative Strength Index (RSI)')
    if st.button("Understand Me", key='rsi'):
        st.info("""
        **Understanding RSI:**

        - **RSI (Relative Strength Index):** Measures the speed and change of price movements.
        - **Overbought (>70):** May indicate a price reversal downward.
        - **Oversold (<30):** May indicate a price reversal upward.
        """)
    rsi_period = 14
    # Fix: Use the same approach for RSI to ensure consistency
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
        yaxis=dict(range=[0, 100])
    )
    st.plotly_chart(fig_rsi, use_container_width=True)

    st.subheader('Bollinger Bands')
    if st.button("Understand Me", key='bb'):
        st.info("""
        **Understanding Bollinger Bands:**

        - **Bollinger Bands:** Consist of a middle band (moving average) and upper/lower bands (standard deviations).
        - **Upper Band:** Price crossing above may indicate overbought conditions.
        - **Lower Band:** Price crossing below may indicate oversold conditions.
        - **Overbought (Red Triangle):** Potential price reversal downward.
        - **Oversold (Green Triangle):** Potential price reversal upward.
        """)
    # Fix: Use the same approach for Bollinger Bands to ensure consistency
    indicator_bb = BollingerBands(close=close_series)
    df['bb_upper'] = indicator_bb.bollinger_hband()
    df['bb_lower'] = indicator_bb.bollinger_lband()
    df['bb_middle'] = indicator_bb.bollinger_mavg()

    df['Position'] = None
    df['Position'] = np.where(df['Close'] < df['bb_lower'], 'Oversold', df['Position'])
    df['Position'] = np.where(df['Close'] > df['bb_upper'], 'Overbought', df['Position'])

    oversold_points = df[df['Position'] == 'Oversold']
    overbought_points = df[df['Position'] == 'Overbought']

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
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig_bb, use_container_width=True)

    st.subheader('Volume Chart')
    fig_volume = go.Figure()
    fig_volume.add_trace(go.Bar(
        x=df.index, y=df['Volume'],
        name='Volume'
    ))
    fig_volume.update_layout(
        yaxis_title='Volume',
        xaxis_title='Date',
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig_volume, use_container_width=True)

    st.subheader('SWOT Analysis')
    strengths, weaknesses, opportunities, threats = perform_swot_analysis(df)
    st.write('**Strengths:**')
    for s in strengths:
        st.markdown(f"- {s}")
    st.write('**Weaknesses:**')
    for w in weaknesses:
        st.markdown(f"- {w}")
    st.write('**Opportunities:**')
    for o in opportunities:
        st.markdown(f"- {o}")
    st.write('**Threats:**')
    for t in threats:
        st.markdown(f"- {t}")

    st.subheader('Raw Data')
    st.write(df.tail())

def perform_swot_analysis(df):
    strengths = []
    weaknesses = []
    opportunities = []
    threats = []

    if df['MACD'].iloc[-1] > df['Signal Line'].iloc[-1]:
        strengths.append('Positive MACD crossover indicates upward momentum.')
    if df['RSI'].iloc[-1] > 30 and df['RSI'].iloc[-1] < 70:
        strengths.append('RSI is in a neutral zone, indicating stable momentum.')
    if df['Close'].iloc[-1] > df['bb_middle'].iloc[-1]:
        strengths.append('Price is above the moving average, indicating a potential uptrend.')

    if df['Volume'].iloc[-1] < df['Volume'].rolling(window=20).mean().iloc[-1]:
        weaknesses.append('Trading volume is below average, indicating low market interest.')
    if df['RSI'].iloc[-1] > 70:
        weaknesses.append('RSI indicates overbought conditions.')

    if df['Close'].iloc[-1] < df['bb_lower'].iloc[-1]:
        opportunities.append('Price is below the lower Bollinger Band, potential oversold opportunity.')
    if df['MACD Hist'].iloc[-1] > 0 and df['MACD Hist'].diff().iloc[-1] > 0:
        opportunities.append('Increasing MACD histogram suggests strengthening momentum.')

    if df['MACD'].iloc[-1] < df['Signal Line'].iloc[-1]:
        threats.append('Negative MACD crossover indicates downward momentum.')
    if df['RSI'].iloc[-1] < 30:
        threats.append('RSI indicates oversold conditions, potential for further decline.')
    if df['Close'].iloc[-1] < df['bb_lower'].iloc[-1]:
        threats.append('Price is below the lower Bollinger Band, indicating strong downward volatility.')

    return strengths, weaknesses, opportunities, threats

if __name__ == '__main__':
    main()
