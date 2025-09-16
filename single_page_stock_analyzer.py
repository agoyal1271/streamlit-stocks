import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List
import time
import warnings
warnings.filterwarnings('ignore')

# Page config
try:
    st.set_page_config(
        page_title="Stock & Options Analyzer",
        page_icon="📊",
        layout="wide"
    )
except:
    pass

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

class StockAnalyzer:
    def __init__(self):
        self.db_name = "stock_data.db"
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                adj_close REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, date)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS options_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                expiry_date TEXT NOT NULL,
                strike REAL NOT NULL,
                option_type TEXT NOT NULL,
                last_price REAL,
                bid REAL,
                ask REAL,
                volume INTEGER,
                open_interest INTEGER,
                implied_volatility REAL,
                data_date TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, expiry_date, strike, option_type, data_date)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def search_and_download_stock_data(self, ticker: str) -> pd.DataFrame:
        try:
            st.info(f"🔄 Searching and downloading data for {ticker}...")
            
            # Verify ticker exists
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if not info or 'symbol' not in info:
                st.error(f"❌ Ticker {ticker} not found")
                return pd.DataFrame()
            
            # Download 1 year of data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            data = stock.history(start=start_date, end=end_date)
            
            if data.empty:
                st.error(f"❌ No data available for {ticker}")
                return pd.DataFrame()
            
            # Save to database
            self.save_stock_data(ticker, data)
            
            st.success(f"✅ Downloaded {len(data)} days of data for {ticker}")
            return data
            
        except Exception as e:
            st.error(f"❌ Error downloading {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def save_stock_data(self, ticker: str, data: pd.DataFrame):
        conn = sqlite3.connect(self.db_name)
        
        for date, row in data.iterrows():
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO stock_data 
                    (ticker, date, open, high, low, close, volume, adj_close)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    ticker,
                    date.strftime('%Y-%m-%d'),
                    row['Open'],
                    row['High'],
                    row['Low'],
                    row['Close'],
                    row['Volume'],
                    row['Close']
                ))
            except:
                continue
        
        conn.commit()
        conn.close()
    
    def get_options_data(self, ticker: str) -> Dict:
        try:
            st.info(f"🔄 Downloading options data for {ticker}...")
            
            stock = yf.Ticker(ticker)
            options_dates = stock.options[:10]  # Next 10 expiry dates
            
            if not options_dates:
                st.error(f"❌ No options data available for {ticker}")
                return {'calls': pd.DataFrame(), 'puts': pd.DataFrame()}
            
            all_calls = []
            all_puts = []
            
            for exp_date in options_dates:
                try:
                    chain = stock.option_chain(exp_date)
                    
                    calls_df = chain.calls.copy()
                    calls_df['expiry'] = exp_date
                    calls_df['days_to_expiry'] = (pd.to_datetime(exp_date) - pd.Timestamp.now()).days
                    all_calls.append(calls_df)
                    
                    puts_df = chain.puts.copy()
                    puts_df['expiry'] = exp_date
                    puts_df['days_to_expiry'] = (pd.to_datetime(exp_date) - pd.Timestamp.now()).days
                    all_puts.append(puts_df)
                    
                except Exception as e:
                    st.warning(f"Could not fetch options for {exp_date}: {str(e)}")
                    continue
            
            calls_data = pd.concat(all_calls) if all_calls else pd.DataFrame()
            puts_data = pd.concat(all_puts) if all_puts else pd.DataFrame()
            
            st.success(f"✅ Downloaded options data for {len(options_dates)} expiry dates")
            
            return {
                'calls': calls_data,
                'puts': puts_data
            }
            
        except Exception as e:
            st.error(f"❌ Error downloading options: {str(e)}")
            return {'calls': pd.DataFrame(), 'puts': pd.DataFrame()}

def calculate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    
    # Moving Averages
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['Close'].ewm(span=12).mean()
    ema_26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    return df

def analyze_options_data(calls_df: pd.DataFrame, puts_df: pd.DataFrame, current_price: float) -> Dict:
    analysis = {
        'put_call_ratio': 0,
        'total_call_oi': 0,
        'total_put_oi': 0,
        'max_pain': 0,
        'support_levels': [],
        'resistance_levels': [],
        'gamma_levels': [],
        'call_wall': 0,
        'put_wall': 0,
        'net_gamma_exposure': 0,
        'dealer_positioning': '',
        'expected_move': 0,
        'volatility_skew': {},
        'unusual_activity': [],
        'flow_analysis': {},
        'insights': []
    }
    
    if calls_df.empty and puts_df.empty:
        return analysis
    
    # Calculate Put-Call Ratio (OI and Volume based)
    if not calls_df.empty and not puts_df.empty:
        total_call_oi = calls_df['openInterest'].sum()
        total_put_oi = puts_df['openInterest'].sum()
        total_call_vol = calls_df['volume'].sum()
        total_put_vol = puts_df['volume'].sum()
        
        analysis['total_call_oi'] = total_call_oi
        analysis['total_put_oi'] = total_put_oi
        
        if total_call_oi > 0:
            analysis['put_call_ratio'] = total_put_oi / total_call_oi
    
    # Enhanced Max Pain Calculation
    if not calls_df.empty and not puts_df.empty:
        all_strikes = sorted(set(calls_df['strike']).union(set(puts_df['strike'])))
        pain_values = []
        
        for strike in all_strikes:
            # ITM calls lose (strike - current_price) per contract
            itm_calls = calls_df[calls_df['strike'] < current_price]
            call_loss = sum(itm_calls['openInterest'] * (current_price - itm_calls['strike']))
            
            # ITM puts lose (current_price - strike) per contract  
            itm_puts = puts_df[puts_df['strike'] > current_price]
            put_loss = sum(itm_puts['openInterest'] * (itm_puts['strike'] - current_price))
            
            total_loss = call_loss + put_loss
            pain_values.append((strike, total_loss))
        
        if pain_values:
            analysis['max_pain'] = max(pain_values, key=lambda x: x[1])[0]
    
    # Enhanced Gamma Levels Analysis with detailed insights
    if not calls_df.empty and not puts_df.empty:
        try:
            gamma_analysis = []
            
            for strike in sorted(set(calls_df['strike']).union(set(puts_df['strike']))):
                call_oi = calls_df[calls_df['strike'] == strike]['openInterest'].sum()
                put_oi = puts_df[puts_df['strike'] == strike]['openInterest'].sum()
                call_vol = calls_df[calls_df['strike'] == strike]['volume'].sum()
                put_vol = puts_df[puts_df['strike'] == strike]['volume'].sum()
                
                # Distance from current price affects gamma
                distance_factor = 1 / (1 + abs(strike - current_price) / current_price)
                
                # Time decay affects gamma (closer to expiry = higher gamma)
                try:
                    if not calls_df[calls_df['strike'] == strike].empty:
                        days_to_expiry = calls_df[calls_df['strike'] == strike]['days_to_expiry'].iloc[0]
                        time_factor = 1 / (1 + days_to_expiry / 30)  # 30-day normalization
                    else:
                        time_factor = 0.5
                except:
                    time_factor = 0.5
                
                # Gamma exposure calculation (simplified)
                call_gamma_exposure = -call_oi * distance_factor * time_factor * 100
                put_gamma_exposure = put_oi * distance_factor * time_factor * 100
                net_gamma_exposure = call_gamma_exposure + put_gamma_exposure
                
                # Gamma acceleration (how much gamma changes as price moves)
                price_distance_pct = (strike - current_price) / current_price * 100
                
                gamma_analysis.append({
                    'strike': strike,
                    'call_oi': call_oi,
                    'put_oi': put_oi,
                    'total_oi': call_oi + put_oi,
                    'call_volume': call_vol,
                    'put_volume': put_vol,
                    'net_gamma_exposure': net_gamma_exposure,
                    'distance_from_current': abs(price_distance_pct),
                    'price_distance_pct': price_distance_pct,
                    'distance_factor': distance_factor,
                    'time_factor': time_factor,
                    'gamma_strength': abs(net_gamma_exposure) * (call_oi + put_oi)
                })
            
            if gamma_analysis:
                gamma_df = pd.DataFrame(gamma_analysis)
                gamma_df = gamma_df.sort_values('gamma_strength', ascending=False)
                
                # Identify key gamma levels
                analysis['gamma_levels'] = gamma_df.head(10)['strike'].tolist()
                
                # Gamma flip level (where net gamma changes sign)
                analysis['gamma_flip_level'] = None
                analysis['gamma_regime'] = 'Neutral'
                
                if not gamma_df.empty:
                    # Find the strike closest to current price
                    gamma_df['distance_to_current'] = gamma_df['strike'].apply(lambda x: abs(x - current_price))
                    closest_idx = gamma_df['distance_to_current'].idxmin()
                    
                    if closest_idx in gamma_df.index:
                        current_net_gamma = gamma_df.loc[closest_idx, 'net_gamma_exposure']
                        
                        if current_net_gamma > 1000:
                            analysis['gamma_regime'] = 'Positive Gamma (Stabilizing)'
                        elif current_net_gamma < -1000:
                            analysis['gamma_regime'] = 'Negative Gamma (Amplifying)'
                
                # Gamma wall analysis
                resistance_strikes = gamma_df[gamma_df['price_distance_pct'] > 0]
                support_strikes = gamma_df[gamma_df['price_distance_pct'] < 0]
                
                analysis['gamma_walls'] = {
                    'resistance': resistance_strikes.head(3)['strike'].tolist() if not resistance_strikes.empty else [],
                    'support': support_strikes.head(3)['strike'].tolist() if not support_strikes.empty else []
                }
                
                # Enhanced gamma insights
                total_gamma_exposure = gamma_df['net_gamma_exposure'].sum()
                gamma_concentration = gamma_df.head(5)['gamma_strength'].sum() / gamma_df['gamma_strength'].sum() if gamma_df['gamma_strength'].sum() > 0 else 0
                
                analysis['gamma_insights'] = []
                
                # Gamma regime insights
                if analysis['gamma_regime'] == 'Positive Gamma (Stabilizing)':
                    analysis['gamma_insights'].append("🛡️ Market makers are net long gamma - Price stabilization expected")
                    analysis['gamma_insights'].append("📉 Volatility suppression likely as dealers hedge by selling into rallies")
                elif analysis['gamma_regime'] == 'Negative Gamma (Amplifying)':
                    analysis['gamma_insights'].append("⚠️ Market makers are net short gamma - Price amplification expected")
                    analysis['gamma_insights'].append("📈 Volatility expansion likely as dealers hedge by buying rallies, selling dips")
                
                # Concentration insights
                if gamma_concentration > 0.6:
                    analysis['gamma_insights'].append(f"🎯 Gamma highly concentrated ({gamma_concentration:.1%}) - Strong pinning effects possible")
                elif gamma_concentration < 0.3:
                    analysis['gamma_insights'].append(f"🌊 Gamma widely distributed ({gamma_concentration:.1%}) - Less pinning, more range")
                
                # Distance-based insights
                nearby_gamma = gamma_df[gamma_df['distance_from_current'] < 5]  # Within 5%
                if not nearby_gamma.empty:
                    max_nearby_idx = nearby_gamma['gamma_strength'].idxmax()
                    if max_nearby_idx in nearby_gamma.index:
                        max_nearby_strike = nearby_gamma.loc[max_nearby_idx, 'strike']
                        analysis['gamma_insights'].append(f"🧲 Strongest nearby gamma at ${max_nearby_strike:.0f} - Potential price magnet")
                
                # Time decay insights
                avg_time_factor = gamma_df['time_factor'].mean()
                if avg_time_factor > 0.7:
                    analysis['gamma_insights'].append("⏰ High time decay sensitivity - Gamma effects intensifying near expiry")
                elif avg_time_factor < 0.3:
                    analysis['gamma_insights'].append("📅 Low time decay impact - Gamma effects more stable")
                
                # Volume vs OI insights for gamma
                high_vol_gamma = gamma_df[gamma_df['call_volume'] + gamma_df['put_volume'] > 
                                         (gamma_df['call_volume'] + gamma_df['put_volume']).quantile(0.8)]
                if not high_vol_gamma.empty:
                    analysis['gamma_insights'].append("🔥 High volume at key gamma levels - Active repositioning detected")
                
                # Store detailed gamma data for display
                analysis['detailed_gamma'] = gamma_df.head(10)
            else:
                analysis['gamma_levels'] = []
                analysis['gamma_insights'] = []
                
        except Exception as e:
            # Fallback to simple gamma calculation if enhanced version fails
            analysis['gamma_levels'] = []
            analysis['gamma_insights'] = [f"Gamma analysis error: {str(e)}"]
    
    # Call and Put Walls (major resistance/support)
    if not calls_df.empty:
        call_oi_by_strike = calls_df.groupby('strike')['openInterest'].sum()
        # Find significant call wall above current price
        above_strikes = call_oi_by_strike[call_oi_by_strike.index > current_price]
        if not above_strikes.empty:
            analysis['call_wall'] = above_strikes.idxmax()
    
    if not puts_df.empty:
        put_oi_by_strike = puts_df.groupby('strike')['openInterest'].sum()
        # Find significant put wall below current price
        below_strikes = put_oi_by_strike[put_oi_by_strike.index < current_price]
        if not below_strikes.empty:
            analysis['put_wall'] = below_strikes.idxmax()
    
    # Net Gamma Exposure (simplified)
    if not calls_df.empty and not puts_df.empty:
        # Calls have positive gamma, puts have negative gamma
        call_gamma_proxy = calls_df['openInterest'].sum()
        put_gamma_proxy = puts_df['openInterest'].sum()
        analysis['net_gamma_exposure'] = call_gamma_proxy - put_gamma_proxy
        
        # Dealer positioning inference
        if analysis['net_gamma_exposure'] > 0:
            analysis['dealer_positioning'] = "Long Gamma (Stabilizing)"
        else:
            analysis['dealer_positioning'] = "Short Gamma (Amplifying)"
    
    # Expected Move Calculation
    if not calls_df.empty and not puts_df.empty:
        # Use closest to ATM straddle for expected move
        atm_strikes = sorted(calls_df['strike'].unique(), key=lambda x: abs(x - current_price))[:3]
        
        expected_moves = []
        for strike in atm_strikes:
            call_price = calls_df[calls_df['strike'] == strike]['lastPrice'].iloc[0] if len(calls_df[calls_df['strike'] == strike]) > 0 else 0
            put_price = puts_df[puts_df['strike'] == strike]['lastPrice'].iloc[0] if len(puts_df[puts_df['strike'] == strike]) > 0 else 0
            straddle_price = call_price + put_price
            expected_moves.append(straddle_price)
        
        if expected_moves:
            analysis['expected_move'] = sum(expected_moves) / len(expected_moves)
    
    # Volatility Skew Analysis
    if not calls_df.empty and not puts_df.empty:
        # OTM put vs OTM call IV comparison
        otm_puts = puts_df[puts_df['strike'] < current_price * 0.95]
        otm_calls = calls_df[calls_df['strike'] > current_price * 1.05]
        
        if not otm_puts.empty and not otm_calls.empty:
            avg_put_iv = otm_puts['impliedVolatility'].mean()
            avg_call_iv = otm_calls['impliedVolatility'].mean()
            
            analysis['volatility_skew'] = {
                'put_iv': avg_put_iv,
                'call_iv': avg_call_iv,
                'skew': avg_put_iv - avg_call_iv
            }
    
    # Unusual Activity Detection
    if not calls_df.empty and not puts_df.empty:
        # High volume relative to open interest
        calls_df['vol_oi_ratio'] = calls_df['volume'] / (calls_df['openInterest'] + 1)
        puts_df['vol_oi_ratio'] = puts_df['volume'] / (puts_df['openInterest'] + 1)
        
        unusual_calls = calls_df[calls_df['vol_oi_ratio'] > 0.5].nlargest(3, 'volume')
        unusual_puts = puts_df[puts_df['vol_oi_ratio'] > 0.5].nlargest(3, 'volume')
        
        for _, row in unusual_calls.iterrows():
            analysis['unusual_activity'].append(f"High Call Volume: ${row['strike']:.0f} strike ({row['volume']:,.0f} vol)")
        
        for _, row in unusual_puts.iterrows():
            analysis['unusual_activity'].append(f"High Put Volume: ${row['strike']:.0f} strike ({row['volume']:,.0f} vol)")
    
    # Options Flow Analysis
    if not calls_df.empty and not puts_df.empty:
        # Categorize flow by premium size
        high_premium_calls = calls_df[calls_df['lastPrice'] > calls_df['lastPrice'].quantile(0.75)]
        high_premium_puts = puts_df[puts_df['lastPrice'] > puts_df['lastPrice'].quantile(0.75)]
        
        analysis['flow_analysis'] = {
            'premium_call_flow': high_premium_calls['volume'].sum(),
            'premium_put_flow': high_premium_puts['volume'].sum(),
            'total_premium_flow': high_premium_calls['volume'].sum() + high_premium_puts['volume'].sum()
        }
    
    # Support and Resistance based on multiple factors
    if not puts_df.empty:
        put_oi_by_strike = puts_df.groupby('strike')['openInterest'].sum()
        put_vol_by_strike = puts_df.groupby('strike')['volume'].sum()
        # Combine OI and volume for stronger support levels
        put_strength = put_oi_by_strike + put_vol_by_strike * 0.3
        analysis['support_levels'] = put_strength.nlargest(5).index.tolist()
    
    if not calls_df.empty:
        call_oi_by_strike = calls_df.groupby('strike')['openInterest'].sum()
        call_vol_by_strike = calls_df.groupby('strike')['volume'].sum()
        # Combine OI and volume for stronger resistance levels
        call_strength = call_oi_by_strike + call_vol_by_strike * 0.3
        analysis['resistance_levels'] = call_strength.nlargest(5).index.tolist()
    
    # Enhanced Insights Generation
    pcr = analysis['put_call_ratio']
    
    # Put-Call Ratio insights
    if pcr > 1.5:
        analysis['insights'].append("🔴 Extremely High Put-Call Ratio (>1.5) - Heavy bearish positioning")
    elif pcr > 1.2:
        analysis['insights'].append("🟠 High Put-Call Ratio (>1.2) - Bearish sentiment dominates")
    elif pcr < 0.6:
        analysis['insights'].append("🟢 Very Low Put-Call Ratio (<0.6) - Strong bullish sentiment")
    elif pcr < 0.8:
        analysis['insights'].append("🟡 Low Put-Call Ratio (<0.8) - Bullish bias present")
    else:
        analysis['insights'].append("⚪ Balanced Put-Call Ratio - Neutral market sentiment")
    
    # Max Pain insights
    max_pain_distance = abs(current_price - analysis['max_pain']) / current_price * 100
    if max_pain_distance < 2:
        analysis['insights'].append(f"⚡ Price very close to Max Pain (${analysis['max_pain']:.2f}) - Potential pinning effect")
    elif current_price > analysis['max_pain']:
        analysis['insights'].append(f"📈 Price above Max Pain (${analysis['max_pain']:.2f}) - Potential downward pressure")
    else:
        analysis['insights'].append(f"📉 Price below Max Pain (${analysis['max_pain']:.2f}) - Potential upward pressure")
    
    # Gamma insights
    if analysis['dealer_positioning'] == "Short Gamma (Amplifying)":
        analysis['insights'].append("⚠️ Dealers likely short gamma - Expect increased volatility")
    else:
        analysis['insights'].append("🛡️ Dealers likely long gamma - Price stabilization expected")
    
    # Wall insights
    if analysis['call_wall'] > 0:
        call_wall_distance = (analysis['call_wall'] - current_price) / current_price * 100
        analysis['insights'].append(f"🧱 Major call wall at ${analysis['call_wall']:.2f} ({call_wall_distance:.1f}% above)")
    
    if analysis['put_wall'] > 0:
        put_wall_distance = (current_price - analysis['put_wall']) / current_price * 100
        analysis['insights'].append(f"🛡️ Major put wall at ${analysis['put_wall']:.2f} ({put_wall_distance:.1f}% below)")
    
    # Volatility skew insights
    if analysis['volatility_skew'] and 'skew' in analysis['volatility_skew']:
        skew = analysis['volatility_skew']['skew']
        if skew > 0.05:
            analysis['insights'].append("📊 Negative volatility skew - Put protection premium")
        elif skew < -0.05:
            analysis['insights'].append("📊 Positive volatility skew - Call speculation premium")
    
    # Flow insights
    if analysis['flow_analysis']:
        premium_ratio = analysis['flow_analysis']['premium_put_flow'] / (analysis['flow_analysis']['premium_call_flow'] + 1)
        if premium_ratio > 1.5:
            analysis['insights'].append("💰 Heavy premium put buying - Defensive positioning")
        elif premium_ratio < 0.67:
            analysis['insights'].append("💰 Heavy premium call buying - Bullish speculation")
    
    return analysis

def create_options_charts(calls_df: pd.DataFrame, puts_df: pd.DataFrame, current_price: float):
    if calls_df.empty or puts_df.empty:
        st.warning("No options data available for charts")
        return
    
    # OI Chart
    fig_oi = go.Figure()
    
    # Aggregate OI by strike
    call_oi = calls_df.groupby('strike')['openInterest'].sum()
    put_oi = puts_df.groupby('strike')['openInterest'].sum()
    
    fig_oi.add_trace(go.Bar(
        x=call_oi.index,
        y=call_oi.values,
        name='Call OI',
        marker_color='green',
        opacity=0.7
    ))
    
    fig_oi.add_trace(go.Bar(
        x=put_oi.index,
        y=-put_oi.values,  # Negative for puts to show below axis
        name='Put OI',
        marker_color='red',
        opacity=0.7
    ))
    
    # Add current price line
    fig_oi.add_vline(x=current_price, line_dash="dash", line_color="blue", 
                     annotation_text=f"Current: ${current_price:.2f}")
    
    fig_oi.update_layout(
        title="Options Open Interest by Strike",
        xaxis_title="Strike Price",
        yaxis_title="Open Interest",
        barmode='overlay',
        height=400
    )
    
    st.plotly_chart(fig_oi, use_container_width=True)
    
    # Volume Chart
    fig_vol = go.Figure()
    
    call_vol = calls_df.groupby('strike')['volume'].sum()
    put_vol = puts_df.groupby('strike')['volume'].sum()
    
    fig_vol.add_trace(go.Bar(
        x=call_vol.index,
        y=call_vol.values,
        name='Call Volume',
        marker_color='lightgreen',
        opacity=0.7
    ))
    
    fig_vol.add_trace(go.Bar(
        x=put_vol.index,
        y=-put_vol.values,
        name='Put Volume',
        marker_color='lightcoral',
        opacity=0.7
    ))
    
    fig_vol.add_vline(x=current_price, line_dash="dash", line_color="blue")
    
    fig_vol.update_layout(
        title="Options Volume by Strike",
        xaxis_title="Strike Price",
        yaxis_title="Volume",
        barmode='overlay',
        height=400
    )
    
    st.plotly_chart(fig_vol, use_container_width=True)

# Initialize analyzer
@st.cache_resource
def get_analyzer():
    return StockAnalyzer()

analyzer = get_analyzer()

# Main App
st.markdown('<h1 class="main-header">📊 Stock & Options Analyzer</h1>', unsafe_allow_html=True)

# Step 1: Stock Search and Download
st.header("🔍 Step 1: Stock Data")

col1, col2 = st.columns([2, 1])

with col1:
    ticker_input = st.text_input(
        "Enter Stock Ticker:",
        placeholder="e.g., AAPL, MSFT, TSLA",
        help="Enter a valid stock ticker symbol"
    ).upper()

with col2:
    st.write("")  # Spacer
    download_btn = st.button("📥 Download Stock Data", use_container_width=True, type="primary")

# Download stock data
if download_btn and ticker_input:
    stock_data = analyzer.search_and_download_stock_data(ticker_input)
    
    if not stock_data.empty:
        st.session_state.ticker = ticker_input
        st.session_state.stock_data = stock_data
        st.session_state.current_price = stock_data['Close'].iloc[-1]
        
        # Calculate technical indicators
        st.session_state.technical_data = calculate_technical_indicators(stock_data)

# Display stock data if available
if 'stock_data' in st.session_state:
    st.success(f"✅ Data loaded for {st.session_state.ticker}")
    
    # Key metrics
    current_price = st.session_state.current_price
    prev_price = st.session_state.stock_data['Close'].iloc[-2]
    change = current_price - prev_price
    change_pct = (change / prev_price) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", f"${current_price:.2f}", f"{change:.2f} ({change_pct:+.2f}%)")
    
    with col2:
        volume = st.session_state.stock_data['Volume'].iloc[-1]
        st.metric("Volume", f"{volume:,.0f}")
    
    with col3:
        high_52w = st.session_state.stock_data['High'].max()
        st.metric("52W High", f"${high_52w:.2f}")
    
    with col4:
        low_52w = st.session_state.stock_data['Low'].min()
        st.metric("52W Low", f"${low_52w:.2f}")
    
    # Technical indicators
    if 'technical_data' in st.session_state:
        tech_data = st.session_state.technical_data
        latest = tech_data.iloc[-1]
        
        st.subheader("📈 Technical Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            rsi = latest['RSI']
            rsi_color = "🟢" if rsi < 30 else "🔴" if rsi > 70 else "🟡"
            st.metric("RSI", f"{rsi_color} {rsi:.1f}")
        
        with col2:
            macd = latest['MACD']
            macd_signal = latest['MACD_Signal']
            macd_color = "🟢" if macd > macd_signal else "🔴"
            st.metric("MACD", f"{macd_color} {macd:.4f}")
        
        with col3:
            ma20 = latest['MA_20']
            ma_color = "🟢" if current_price > ma20 else "🔴"
            st.metric("MA 20", f"{ma_color} ${ma20:.2f}")
        
        with col4:
            bb_upper = latest['BB_Upper']
            bb_lower = latest['BB_Lower']
            if current_price > bb_upper:
                bb_signal = "🔴 Overbought"
            elif current_price < bb_lower:
                bb_signal = "🟢 Oversold"
            else:
                bb_signal = "🟡 Normal"
            st.metric("BB Signal", bb_signal)

# Step 2: Options Analysis
if 'ticker' in st.session_state:
    st.header("📊 Step 2: Options Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Download Options Data", use_container_width=True, type="primary"):
            options_data = analyzer.get_options_data(st.session_state.ticker)
            
            if not options_data['calls'].empty or not options_data['puts'].empty:
                st.session_state.options_data = options_data
                
                # Analyze options
                analysis = analyze_options_data(
                    options_data['calls'], 
                    options_data['puts'], 
                    st.session_state.current_price
                )
                st.session_state.options_analysis = analysis
    
    with col2:
        if st.button("📊 Analyze Settlement History", use_container_width=True, type="secondary"):
            settlement_data = analyzer.analyze_settlement_patterns(st.session_state.ticker)
            st.session_state.settlement_analysis = settlement_data
            
            if settlement_data['total_settled_contracts'] > 0:
                st.success(f"Found {settlement_data['total_settled_contracts']:,.0f} settled contracts")
            else:
                st.info("No historical settlement data available yet")

# Display options analysis
if 'options_analysis' in st.session_state:
    analysis = st.session_state.options_analysis
    
    st.subheader("📊 Advanced Options Analysis")
    
    # Add filtering controls
    st.subheader("🎛️ Filter Options Data")
    
    # Get available strikes and expiry dates
    all_calls = st.session_state.options_data['calls']
    all_puts = st.session_state.options_data['puts']
    
    if not all_calls.empty or not all_puts.empty:
        # Strike price filter
        col1, col2 = st.columns(2)
        
        with col1:
            all_strikes = sorted(set(list(all_calls['strike']) + list(all_puts['strike'])))
            min_strike = min(all_strikes)
            max_strike = max(all_strikes)
            current_price = st.session_state.current_price
            
            # Default range: 20% below to 20% above current price
            default_min = max(min_strike, current_price * 0.8)
            default_max = min(max_strike, current_price * 1.2)
            
            strike_range = st.slider(
                "Strike Price Range",
                min_value=float(min_strike),
                max_value=float(max_strike),
                value=(float(default_min), float(default_max)),
                step=1.0,
                format="$%.0f"
            )
        
        with col2:
            # Date filter
            all_expiries = sorted(set(list(all_calls['expiry']) + list(all_puts['expiry'])))
            
            if len(all_expiries) > 1:
                expiry_range = st.select_slider(
                    "Expiry Date Range",
                    options=all_expiries,
                    value=(all_expiries[0], all_expiries[-1]),
                    format_func=lambda x: f"{x} ({(pd.to_datetime(x) - pd.Timestamp.now()).days}d)"
                )
            else:
                expiry_range = (all_expiries[0], all_expiries[0]) if all_expiries else (None, None)
                st.info(f"Single expiry available: {all_expiries[0] if all_expiries else 'None'}")
        
        # Filter the data based on selections
        filtered_calls = all_calls[
            (all_calls['strike'] >= strike_range[0]) & 
            (all_calls['strike'] <= strike_range[1]) &
            (all_calls['expiry'] >= expiry_range[0]) &
            (all_calls['expiry'] <= expiry_range[1])
        ]
        
        filtered_puts = all_puts[
            (all_puts['strike'] >= strike_range[0]) & 
            (all_puts['strike'] <= strike_range[1]) &
            (all_puts['expiry'] >= expiry_range[0]) &
            (all_puts['expiry'] <= expiry_range[1])
        ]
        
        # Recalculate analysis with filtered data
        if not filtered_calls.empty or not filtered_puts.empty:
            filtered_analysis = analyze_options_data(filtered_calls, filtered_puts, current_price)
            
            # Show filtering results
            st.info(f"Filtered to {len(filtered_calls)} calls and {len(filtered_puts)} puts in selected range")
            
            # Use filtered analysis for display
            analysis = filtered_analysis
            
            # Update session state with filtered data for charts
            st.session_state.filtered_calls = filtered_calls
            st.session_state.filtered_puts = filtered_puts
        else:
            st.warning("No options data in selected range. Showing all data.")
            st.session_state.filtered_calls = all_calls
            st.session_state.filtered_puts = all_puts
    
    # Key metrics - expanded
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        pcr = analysis['put_call_ratio']
        pcr_color = "🔴" if pcr > 1.2 else "🟢" if pcr < 0.8 else "🟡"
        st.metric("Put-Call Ratio", f"{pcr_color} {pcr:.2f}")
    
    with col2:
        st.metric("Total Call OI", f"{analysis['total_call_oi']:,.0f}")
    
    with col3:
        st.metric("Total Put OI", f"{analysis['total_put_oi']:,.0f}")
    
    with col4:
        st.metric("Max Pain", f"${analysis['max_pain']:.2f}")
    
    with col5:
        st.metric("Expected Move", f"${analysis['expected_move']:.2f}")
    
    # Advanced metrics row
    st.subheader("🎯 Advanced Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if analysis['call_wall'] > 0:
            call_wall_dist = (analysis['call_wall'] - st.session_state.current_price) / st.session_state.current_price * 100
            st.metric("Call Wall", f"${analysis['call_wall']:.2f}", f"{call_wall_dist:+.1f}%")
        else:
            st.metric("Call Wall", "None detected")
    
    with col2:
        if analysis['put_wall'] > 0:
            put_wall_dist = (st.session_state.current_price - analysis['put_wall']) / st.session_state.current_price * 100
            st.metric("Put Wall", f"${analysis['put_wall']:.2f}", f"{put_wall_dist:+.1f}%")
        else:
            st.metric("Put Wall", "None detected")
    
    with col3:
        gamma_signal = "🟢 Stabilizing" if "Long Gamma" in analysis['dealer_positioning'] else "🔴 Amplifying"
        st.metric("Dealer Gamma", gamma_signal)
    
    with col4:
        if analysis['volatility_skew'] and 'skew' in analysis['volatility_skew']:
            skew = analysis['volatility_skew']['skew']
            skew_direction = "Put Premium" if skew > 0 else "Call Premium"
            st.metric("Vol Skew", f"{skew:.3f}", skew_direction)
        else:
            st.metric("Vol Skew", "Calculating...")
    
    # Market Insights - Enhanced
    st.subheader("🧠 Professional Market Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📈 Sentiment & Positioning**")
        for insight in analysis['insights'][:len(analysis['insights'])//2]:
            st.write(f"• {insight}")
    
    with col2:
        st.markdown("**⚡ Technical Factors**")
        for insight in analysis['insights'][len(analysis['insights'])//2:]:
            st.write(f"• {insight}")
    
    # Unusual Activity Alert
    if analysis['unusual_activity']:
        st.subheader("🚨 Unusual Options Activity")
        for activity in analysis['unusual_activity']:
            st.warning(f"⚠️ {activity}")
    
    # Settlement Analysis Display
    if 'settlement_analysis' in st.session_state:
        settlement = st.session_state.settlement_analysis
        
        if settlement['total_settled_contracts'] > 0:
            st.subheader("📋 Historical Settlement Analysis")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Settled", f"{settlement['total_settled_contracts']:,.0f}")
            
            with col2:
                st.metric("Settled Calls", f"{settlement['settled_call_contracts']:,.0f}")
            
            with col3:
                st.metric("Settled Puts", f"{settlement['settled_put_contracts']:,.0f}")
            
            with col4:
                if settlement['settled_call_contracts'] > 0:
                    settlement_ratio = settlement['settled_put_contracts'] / settlement['settled_call_contracts']
                    st.metric("Historical P/C", f"{settlement_ratio:.2f}")
                else:
                    st.metric("Historical P/C", "N/A")
            
            if settlement['settlement_insights']:
                st.info("Settlement Insights:")
                for insight in settlement['settlement_insights']:
                    st.write(f"• {insight}")
    
    # Gamma Levels (Enhanced with detailed insights)
    if analysis.get('detailed_gamma') is not None and not analysis['detailed_gamma'].empty:
        st.subheader("⚡ Advanced Gamma Analysis")
        
        # Gamma regime indicator
        col1, col2, col3 = st.columns(3)
        
        with col1:
            regime_color = "🟢" if "Positive" in analysis.get('gamma_regime', '') else "🔴" if "Negative" in analysis.get('gamma_regime', '') else "🟡"
            st.metric("Gamma Regime", f"{regime_color} {analysis.get('gamma_regime', 'Unknown')}")
        
        with col2:
            if analysis.get('gamma_walls', {}).get('resistance'):
                nearest_resistance = min(analysis['gamma_walls']['resistance'], key=lambda x: abs(x - st.session_state.current_price))
                resistance_dist = (nearest_resistance - st.session_state.current_price) / st.session_state.current_price * 100
                st.metric("Gamma Resistance", f"${nearest_resistance:.0f}", f"{resistance_dist:+.1f}%")
            else:
                st.metric("Gamma Resistance", "None nearby")
        
        with col3:
            if analysis.get('gamma_walls', {}).get('support'):
                nearest_support = max(analysis['gamma_walls']['support'], key=lambda x: -abs(x - st.session_state.current_price))
                support_dist = (st.session_state.current_price - nearest_support) / st.session_state.current_price * 100
                st.metric("Gamma Support", f"${nearest_support:.0f}", f"{support_dist:+.1f}%")
            else:
                st.metric("Gamma Support", "None nearby")
        
        # Gamma insights
        if analysis.get('gamma_insights'):
            st.markdown("**🎯 Gamma Insights:**")
            for insight in analysis['gamma_insights']:
                st.write(f"• {insight}")
        
        # Detailed gamma table
        st.markdown("**📊 Top Gamma Levels:**")
        gamma_display = analysis['detailed_gamma'][['strike', 'total_oi', 'net_gamma_exposure', 'distance_from_current', 'gamma_strength']].copy()
        gamma_display['distance_from_current'] = gamma_display['distance_from_current'].apply(lambda x: f"{x:.1f}%")
        gamma_display['net_gamma_exposure'] = gamma_display['net_gamma_exposure'].apply(lambda x: f"{x:,.0f}")
        gamma_display['gamma_strength'] = gamma_display['gamma_strength'].apply(lambda x: f"{x:,.0f}")
        gamma_display.columns = ['Strike', 'Total OI', 'Net Gamma Exp', 'Distance', 'Gamma Strength']
        
        st.dataframe(gamma_display.head(8), use_container_width=True)
    
    elif analysis['gamma_levels']:
        st.subheader("⚡ High Gamma Levels")
        st.info("These strikes have high gamma exposure and may act as magnets for price action:")
        
        gamma_cols = st.columns(len(analysis['gamma_levels'][:5]))
        for i, level in enumerate(analysis['gamma_levels'][:5]):
            with gamma_cols[i]:
                distance = (level - st.session_state.current_price) / st.session_state.current_price * 100
                st.metric(f"Gamma #{i+1}", f"${level:.2f}", f"{distance:+.1f}%")
    
    # Options Flow Analysis
    if analysis['flow_analysis']:
        st.subheader("💰 Options Flow Analysis")
        flow = analysis['flow_analysis']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Premium Call Flow", f"{flow['premium_call_flow']:,.0f}")
        with col2:
            st.metric("Premium Put Flow", f"{flow['premium_put_flow']:,.0f}")
        with col3:
            total_flow = flow['total_premium_flow']
            st.metric("Total Premium Flow", f"{total_flow:,.0f}")
    
    # Enhanced Support and Resistance
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🛡️ Support Levels (Put Strength)")
        st.caption("Based on Open Interest + Volume weighting")
        for i, level in enumerate(analysis['support_levels'][:5], 1):
            distance = ((st.session_state.current_price - level) / st.session_state.current_price) * 100
            strength = "Strong" if i <= 2 else "Moderate" if i <= 3 else "Weak"
            st.write(f"{i}. **${level:.2f}** ({distance:+.1f}%) - {strength}")
    
    with col2:
        st.subheader("⚡ Resistance Levels (Call Strength)")
        st.caption("Based on Open Interest + Volume weighting")
        for i, level in enumerate(analysis['resistance_levels'][:5], 1):
            distance = ((level - st.session_state.current_price) / st.session_state.current_price) * 100
            strength = "Strong" if i <= 2 else "Moderate" if i <= 3 else "Weak"
            st.write(f"{i}. **${level:.2f}** ({distance:+.1f}%) - {strength}")
    
    # Charts with filtered data
    st.subheader("📈 Options Visualization")
    
    # Use filtered data for charts if available
    chart_calls = st.session_state.get('filtered_calls', st.session_state.options_data['calls'])
    chart_puts = st.session_state.get('filtered_puts', st.session_state.options_data['puts'])
    
    create_options_charts(chart_calls, chart_puts, st.session_state.current_price)
    
    # Options tables
    st.subheader("📋 Options Data")
    
    tab1, tab2 = st.tabs(["Call Options", "Put Options"])
    
    with tab1:
        if not st.session_state.options_data['calls'].empty:
            calls_display = st.session_state.options_data['calls'][
                ['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility', 'expiry']
            ].round(4)
            st.dataframe(calls_display, use_container_width=True)
        else:
            st.info("No call options data available")
    
    with tab2:
        if not st.session_state.options_data['puts'].empty:
            puts_display = st.session_state.options_data['puts'][
                ['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility', 'expiry']
            ].round(4)
            st.dataframe(puts_display, use_container_width=True)
        else:
            st.info("No put options data available")

# Footer
st.markdown("---")
st.markdown(
    "📊 **Stock & Options Analyzer** | Built with Streamlit | "
    "⚠️ *For educational purposes only. Not financial advice.*"
)