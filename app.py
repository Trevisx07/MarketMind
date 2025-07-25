import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

#st
st.set_page_config(
    page_title="🚀 Elite AI Stock Market Dashboard",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #00ff88;
        margin: 0.5rem 0;
    }
    .analysis-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 10px;
        border-left: 3px solid #ff6b6b;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid #00ff88;
        margin: 1rem 0;
        animation: glow 2s ease-in-out infinite alternate;
    }
    @keyframes glow {
        from { box-shadow: 0 0 5px #00ff88; }
        to { box-shadow: 0 0 20px #00ff88; }
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2c3e50, #34495e);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

class AdvancedStockAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {
            'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression()
        }
        self.best_model = None
        self.best_model_name = None
        
    def fetch_stock_data(self, symbol, period="1y"):
        """Enhanced stock data fetching with error handling and validation"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            info = stock.info
            
            if data.empty:
                return None, None
                
            
            data = data.dropna()
            if len(data) < 30:  
                return None, None
                
            return data, info
        except Exception as e:
            st.error(f"❌ Error fetching data for {symbol}: {str(e)}")
            return None, None
    
    def calculate_advanced_technical_indicators(self, data):
        """Calculate comprehensive technical indicators with advanced features"""
        df = data.copy()
        
    
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
            df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()
        
 
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        
        for period in [14, 21]:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        
      
        for period, std_dev in [(20, 2), (20, 2.5)]:
            middle = df['Close'].rolling(window=period).mean()
            std = df['Close'].rolling(window=period).std()
            df[f'BB_upper_{period}_{std_dev}'] = middle + (std * std_dev)
            df[f'BB_lower_{period}_{std_dev}'] = middle - (std * std_dev)
            df[f'BB_middle_{period}'] = middle
        
  
        high_14 = df['High'].rolling(window=14).max()
        low_14 = df['Low'].rolling(window=14).min()
        df['Williams_R'] = -100 * ((high_14 - df['Close']) / (high_14 - low_14))
        
     
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        sma_tp = typical_price.rolling(window=20).mean()
        mad = typical_price.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())))
        df['CCI'] = (typical_price - sma_tp) / (0.015 * mad)
        
       
        df['High_Low'] = df['High'] - df['Low']
        df['High_Close'] = np.abs(df['High'] - df['Close'].shift())
        df['Low_Close'] = np.abs(df['Low'] - df['Close'].shift())
        df['True_Range'] = df[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)
        df['ATR'] = df['True_Range'].rolling(window=14).mean()
        df['ATR_percent'] = (df['ATR'] / df['Close']) * 100
        
       
        for period in [14, 21]:
            low_n = df['Low'].rolling(window=period).min()
            high_n = df['High'].rolling(window=period).max()
            df[f'Stoch_K_{period}'] = 100 * ((df['Close'] - low_n) / (high_n - low_n))
            df[f'Stoch_D_{period}'] = df[f'Stoch_K_{period}'].rolling(window=3).mean()
        
   
        df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_SMA_20']
        df['OBV'] = (df['Volume'] * np.sign(df['Close'].diff())).fillna(0).cumsum()
        
        
        df['Price_Change'] = df['Close'] - df['Open']
        df['Price_Change_Pct'] = (df['Close'] - df['Open']) / df['Open'] * 100
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100
        
        
        for period in [1, 5, 10, 20]:
            df[f'Momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1
        
     
        df['Support'] = df['Low'].rolling(window=20).min()
        df['Resistance'] = df['High'].rolling(window=20).max()
        
      
        for period in [10, 20, 30]:
            df[f'Volatility_{period}'] = df['Close'].pct_change().rolling(period).std()
        
        return df
    
    def prepare_advanced_ml_features(self, data):
        """Prepare comprehensive features for machine learning"""
        df = data.copy()
        
        df['Returns'] = df['Close'].pct_change()
        
        for days in [1, 2, 3, 5, 10, 20]:
            df[f'Returns_{days}d'] = df['Close'].pct_change(days)
            df[f'Volume_change_{days}d'] = df['Volume'].pct_change(days)
        
        for lag in [1, 2, 3, 5, 10, 20]:
            df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_lag_{lag}'] = df['Volume'].shift(lag)
            df[f'High_lag_{lag}'] = df['High'].shift(lag)
            df[f'Low_lag_{lag}'] = df['Low'].shift(lag)
            if f'RSI_14' in df.columns:
                df[f'RSI_lag_{lag}'] = df['RSI_14'].shift(lag)
        
        for window in [5, 10, 20, 50]:
            df[f'Close_mean_{window}'] = df['Close'].rolling(window).mean()
            df[f'Close_std_{window}'] = df['Close'].rolling(window).std()
            df[f'Volume_mean_{window}'] = df['Volume'].rolling(window).mean()
            df[f'High_mean_{window}'] = df['High'].rolling(window).mean()
            df[f'Low_mean_{window}'] = df['Low'].rolling(window).mean()
            df[f'Returns_mean_{window}'] = df['Returns'].rolling(window).mean()
            df[f'Returns_std_{window}'] = df['Returns'].rolling(window).std()
        
        for sma in [5, 10, 20, 50]:
            if f'SMA_{sma}' in df.columns:
                df[f'Price_vs_SMA{sma}'] = (df['Close'] - df[f'SMA_{sma}']) / df[f'SMA_{sma}'] * 100
        
  
        if 'RSI_14' in df.columns and 'RSI_21' in df.columns:
            df['RSI_diff'] = df['RSI_14'] - df['RSI_21']
        
        if 'Stoch_K_14' in df.columns and 'Stoch_D_14' in df.columns:
            df['Stoch_diff'] = df['Stoch_K_14'] - df['Stoch_D_14']
        
        
        df['Volume_price_trend'] = df['Volume_ratio'] * np.sign(df['Returns'])
        
  
        df['DayOfWeek'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        
        return df
    
    def train_ensemble_models(self, data):
        """Train multiple models and select the best performer"""
        df = self.prepare_advanced_ml_features(data)
        df = df.dropna()
        
        if len(df) < 100:
            return None
        
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
        feature_cols = [col for col in df.columns if not any(exc in col for exc in exclude_cols)]
        feature_cols = [col for col in feature_cols if not col.startswith('Returns')]
        
        if len(feature_cols) > 50:
            feature_cols = feature_cols[:50] 
        
        X = df[feature_cols].fillna(method='ffill').fillna(method='bfill')
        y = df['Close'].shift(-1)  
        
        X = X[:-1]
        y = y[:-1]
        
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        if len(X) < 50:
            return None
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        model_results = {}
        
        for name, model in self.models.items():
            try:
                model.fit(X_train_scaled, y_train)
                
                train_pred = model.predict(X_train_scaled)
                test_pred = model.predict(X_test_scaled)
                
                train_r2 = r2_score(y_train, train_pred)
                test_r2 = r2_score(y_test, test_pred)
                test_mse = mean_squared_error(y_test, test_pred)
                
                model_results[name] = {
                    'model': model,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'test_mse': test_mse,
                    'predictions': test_pred
                }
            except Exception as e:
                continue
        
        if not model_results:
            return None
        
        best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['test_r2'])
        self.best_model = model_results[best_model_name]['model']
        self.best_model_name = best_model_name
        
        feature_importance = {}
        if hasattr(self.best_model, 'feature_importances_'):
            feature_importance = dict(zip(feature_cols, self.best_model.feature_importances_))
        
        return {
            'model_results': model_results,
            'best_model_name': best_model_name,
            'feature_importance': feature_importance,
            'last_features': X.iloc[-1:],
            'feature_cols': feature_cols,
            'y_test': y_test,
            'predictions': model_results[best_model_name]['predictions']
        }
    
    def predict_multiple_days(self, model_info, days=5):
        """Predict stock price for multiple days ahead"""
        if model_info is None or self.best_model is None:
            return None
        
        predictions = []
        last_features = model_info['last_features'].copy()
        
        for day in range(days):
            last_features_scaled = self.scaler.transform(last_features)
            prediction = self.best_model.predict(last_features_scaled)[0]
            predictions.append(prediction)
            
            last_features = last_features.copy()
        
        return predictions
    
    def generate_advanced_analysis(self, data, info, symbol, model_info=None):
        """Generate comprehensive AI-powered market analysis"""
        latest = data.iloc[-1]
        prev = data.iloc[-2]
        
        analysis = []
        
        price_change = latest['Close'] - prev['Close']
        price_change_pct = (price_change / prev['Close']) * 100
        
        if price_change_pct > 5:
            analysis.append(f"🚀 **EXPLOSIVE BREAKOUT**: {symbol} surged {price_change_pct:.2f}% - exceptional bullish momentum detected!")
        elif price_change_pct > 3:
            analysis.append(f"📈 **STRONG BULLISH**: {symbol} gained {price_change_pct:.2f}% with significant buying pressure")
        elif price_change_pct > 1:
            analysis.append(f"🟢 **POSITIVE MOMENTUM**: {symbol} advanced {price_change_pct:.2f}% showing healthy upward movement")
        elif price_change_pct > 0:
            analysis.append(f"🟡 **MODEST GAINS**: {symbol} edged higher by {price_change_pct:.2f}%")
        elif price_change_pct > -1:
            analysis.append(f"🟡 **SLIGHT DECLINE**: {symbol} dipped {price_change_pct:.2f}% in light selling")
        elif price_change_pct > -3:
            analysis.append(f"🔴 **BEARISH PRESSURE**: {symbol} declined {price_change_pct:.2f}% amid selling pressure")
        else:
            analysis.append(f"🔻 **SHARP SELLOFF**: {symbol} plunged {price_change_pct:.2f}% - significant bearish momentum")
        
        if 'RSI_14' in data.columns:
            rsi_14 = latest['RSI_14']
            if rsi_14 > 80:
                analysis.append(f"🚨 **EXTREMELY OVERBOUGHT**: RSI-14 at {rsi_14:.1f} - high probability of pullback")
            elif rsi_14 > 70:
                analysis.append(f"⚠️ **OVERBOUGHT TERRITORY**: RSI-14 at {rsi_14:.1f} - caution advised")
            elif rsi_14 < 20:
                analysis.append(f"🛒 **SEVERELY OVERSOLD**: RSI-14 at {rsi_14:.1f} - strong bounce candidate")
            elif rsi_14 < 30:
                analysis.append(f"💡 **OVERSOLD BOUNCE SETUP**: RSI-14 at {rsi_14:.1f} - potential reversal zone")
            elif 45 <= rsi_14 <= 55:
                analysis.append(f"⚖️ **BALANCED MOMENTUM**: RSI-14 at {rsi_14:.1f} - neutral territory")
        
        ma_analysis = []
        if all(col in data.columns for col in ['SMA_20', 'SMA_50', 'SMA_200']):
            price = latest['Close']
            sma_20 = latest['SMA_20']
            sma_50 = latest['SMA_50']
            sma_200 = latest['SMA_200']
            
            if price > sma_20 > sma_50 > sma_200:
                ma_analysis.append("🔥 **PERFECT BULL ALIGNMENT** - Price above all major MAs")
            elif price < sma_20 < sma_50 < sma_200:
                ma_analysis.append("❄️ **BEAR MARKET STRUCTURE** - Price below all major MAs")
            elif price > sma_20 and sma_20 > sma_50:
                ma_analysis.append("📈 **SHORT-TERM BULLISH** - Above 20 & 50 day MAs")
            elif price < sma_20 and sma_20 < sma_50:
                ma_analysis.append("📉 **SHORT-TERM BEARISH** - Below 20 & 50 day MAs")
        
        if ma_analysis:
            analysis.extend(ma_analysis)
        
        if 'Volume_ratio' in data.columns:
            vol_ratio = latest['Volume_ratio']
            if vol_ratio > 3:
                analysis.append("🔥 **MASSIVE VOLUME SURGE** - Institutional activity likely")
            elif vol_ratio > 2:
                analysis.append("📊 **HIGH VOLUME CONFIRMATION** - Strong conviction behind move")
            elif vol_ratio > 1.5:
                analysis.append("📊 **Above-average volume** validates price action")
            elif vol_ratio < 0.5:
                analysis.append("📊 **Low volume** suggests lack of conviction")
        
        if all(col in data.columns for col in ['MACD', 'MACD_signal']):
            macd = latest['MACD']
            macd_signal = latest['MACD_signal']
            macd_hist = latest.get('MACD_histogram', 0)
            
            if macd > macd_signal and macd > 0 and macd_hist > 0:
                analysis.append("⚡ **STRONG BULLISH MACD** - All systems go for uptrend")
            elif macd < macd_signal and macd < 0 and macd_hist < 0:
                analysis.append("⚡ **STRONG BEARISH MACD** - Downtrend momentum building")
            elif macd > macd_signal:
                analysis.append("⚡ **MACD BULLISH CROSSOVER** - Momentum turning positive")
            else:
                analysis.append("⚡ **MACD BEARISH CROSSOVER** - Momentum turning negative")
        
        if 'ATR_percent' in data.columns:
            atr_pct = latest['ATR_percent']
            if atr_pct > 5:
                analysis.append(f"🌊 **HIGH VOLATILITY** - ATR at {atr_pct:.1f}% suggests elevated risk/reward")
            elif atr_pct < 2:
                analysis.append(f"😴 **LOW VOLATILITY** - ATR at {atr_pct:.1f}% suggests consolidation")
        
        if model_info and 'model_results' in model_info:
            best_model = model_info['best_model_name']
            test_r2 = model_info['model_results'][best_model]['test_r2']
            
            if test_r2 > 0.8:
                analysis.append(f"🤖 **HIGH-CONFIDENCE AI PREDICTION** - {best_model} model shows {test_r2:.1%} accuracy")
            elif test_r2 > 0.6:
                analysis.append(f"🤖 **MODERATE-CONFIDENCE AI PREDICTION** - {best_model} model shows {test_r2:.1%} accuracy")
            else:
                analysis.append(f"🤖 **LOW-CONFIDENCE AI PREDICTION** - Market conditions challenging for ML models")
        
        recent_high = data['High'].rolling(20).max().iloc[-1]
        recent_low = data['Low'].rolling(20).min().iloc[-1]
        current_price = latest['Close']
        
        position_in_range = (current_price - recent_low) / (recent_high - recent_low)
        
        if position_in_range > 0.8:
            analysis.append("🎯 **NEAR 20-DAY HIGHS** - Strong relative strength")
        elif position_in_range < 0.2:
            analysis.append("🎯 **NEAR 20-DAY LOWS** - Potential support test")
        else:
            analysis.append(f"🎯 **MID-RANGE TRADING** - {position_in_range:.0%} of 20-day range")
        
        return analysis

def create_professional_chart(data, symbol):
    """Create professional-grade trading chart"""
    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=(
            f'{symbol} - Professional Trading Analysis',
            'Volume Profile',
            'MACD Momentum',
            'RSI & Stochastic',
            'Support/Resistance'
        ),
        row_heights=[0.4, 0.15, 0.15, 0.15, 0.15]
    )
    
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='OHLC',
            increasing_line_color='#00C851',
            decreasing_line_color='#FF4444',
            increasing_fillcolor='#00C851',
            decreasing_fillcolor='#FF4444'
        ),
        row=1, col=1
    )
    
    ma_colors = ['#FFD700', '#FF6347', '#4169E1', '#32CD32']
    ma_configs = [
        ('SMA_20', 'SMA 20', ma_colors[0], 2),
        ('SMA_50', 'SMA 50', ma_colors[1], 2),
        ('SMA_100', 'SMA 100', ma_colors[2], 1.5),
        ('SMA_200', 'SMA 200', ma_colors[3], 3)
    ]
    
    for ma_col, ma_name, color, width in ma_configs:
        if ma_col in data.columns and not data[ma_col].isna().all():
            fig.add_trace(
                go.Scatter(
                    x=data.index, y=data[ma_col],
                    line=dict(color=color, width=width),
                    name=ma_name,
                    opacity=0.8
                ),
                row=1, col=1
            )
    
    if all(col in data.columns for col in ['BB_upper_20_2', 'BB_lower_20_2']):
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data['BB_upper_20_2'],
                line=dict(color='rgba(128,128,128,0.3)', width=1),
                name='BB Upper', showlegend=False
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data['BB_lower_20_2'],
                line=dict(color='rgba(128,128,128,0.3)', width=1),
                name='BB Lower', showlegend=False,
                fill='tonexty', fillcolor='rgba(128,128,128,0.1)'
            ),
            row=1, col=1
        )
    
    volume_colors = []
    for i in range(len(data)):
        if data['Close'].iloc[i] >= data['Open'].iloc[i]:
            volume_colors.append('#00C851')
        else:
            volume_colors.append('#FF4444')
    
    fig.add_trace(
        go.Bar(
            x=data.index, y=data['Volume'],
            marker_color=volume_colors,
            name='Volume', opacity=0.7
        ),
        row=2, col=1
    )
    
    if 'Volume_SMA_20' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data['Volume_SMA_20'],
                line=dict(color='yellow', width=2),
                name='Vol SMA'
            ),
            row=2, col=1
        )
    
    if all(col in data.columns for col in ['MACD', 'MACD_signal', 'MACD_histogram']):
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data['MACD'],
                line=dict(color='#00BFFF', width=2),
                name='MACD'
            ),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data['MACD_signal'],
                line=dict(color='#FF6347', width=2),
                name='Signal'
            ),
            row=3, col=1
        )
        
        histogram_colors = ['#00C851' if val >= 0 else '#FF4444' for val in data['MACD_histogram']]
        fig.add_trace(
            go.Bar(
                x=data.index, y=data['MACD_histogram'],
                marker_color=histogram_colors,
                name='Histogram', opacity=0.6
            ),
            row=3, col=1
        )
    
    if 'RSI_14' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data['RSI_14'],
                line=dict(color='#9932CC', width=2),
                name='RSI'
            ),
            row=4, col=1
        )
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.7, row=4, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.7, row=4, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.5, row=4, col=1)
    
    if 'Stoch_K_14' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data['Stoch_K_14'],
                line=dict(color='#FFD700', width=1.5),
                name='Stoch %K'
            ),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data['Stoch_D_14'],
                line=dict(color='#FF8C00', width=1.5),
                name='Stoch %D'
            ),
            row=4, col=1
        )
    
    if all(col in data.columns for col in ['Support', 'Resistance']):
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data['Support'],
                line=dict(color='green', width=2, dash='dot'),
                name='Support'
            ),
            row=5, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data['Resistance'],
                line=dict(color='red', width=2, dash='dot'),
                name='Resistance'
            ),
            row=5, col=1
        )
    
    fig.update_layout(
        title=dict(
            text=f'{symbol} - Elite Trading Dashboard',
            font=dict(size=24, color='white'),
            x=0.5
        ),
        xaxis_rangeslider_visible=False,
        height=1200,
        showlegend=True,
        template='plotly_dark',
        font=dict(size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    for i in range(1, 5):
        fig.update_xaxes(showticklabels=False, row=i, col=1)
    
    return fig

def create_prediction_visualization(data, predictions, symbol):
    """Create advanced prediction visualization"""
    if not predictions:
        return None
    
    recent_data = data.tail(30)
    
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=len(predictions), freq='D')
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=recent_data.index,
            y=recent_data['Close'],
            mode='lines',
            name='Historical Price',
            line=dict(color='#00BFFF', width=3)
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=predictions,
            mode='lines+markers',
            name='AI Predictions',
            line=dict(color='#FF6B6B', width=3, dash='dash'),
            marker=dict(size=8, symbol='diamond')
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=[recent_data.index[-1], future_dates[0]],
            y=[recent_data['Close'].iloc[-1], predictions[0]],
            mode='lines',
            name='Transition',
            line=dict(color='#FFD700', width=2),
            showlegend=False
        )
    )
    
    fig.update_layout(
        title=f'{symbol} - AI Price Predictions (Next {len(predictions)} Days)',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        template='plotly_dark',
        height=400
    )
    
    return fig

def create_model_comparison_chart(model_results):
    """Create model performance comparison"""
    if not model_results:
        return None
    
    models = list(model_results.keys())
    train_scores = [model_results[model]['train_r2'] for model in models]
    test_scores = [model_results[model]['test_r2'] for model in models]
    
    fig = go.Figure(data=[
        go.Bar(name='Training R²', x=models, y=train_scores, marker_color='#00C851'),
        go.Bar(name='Testing R²', x=models, y=test_scores, marker_color='#FF6B6B')
    ])
    
    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Models',
        yaxis_title='R² Score',
        barmode='group',
        template='plotly_dark',
        height=400
    )
    
    return fig

def create_enhanced_metrics(data, info, symbol):
    """Create enhanced performance metrics with more details"""
    data['Daily_Returns'] = data['Close'].pct_change()
    data['Cumulative_Returns'] = (1 + data['Daily_Returns']).cumprod() - 1
    
    total_return = data['Cumulative_Returns'].iloc[-1] * 100
    volatility = data['Daily_Returns'].std() * np.sqrt(252) * 100
    sharpe_ratio = (data['Daily_Returns'].mean() * 252) / (data['Daily_Returns'].std() * np.sqrt(252))
    
    rolling_max = data['Close'].expanding().max()
    drawdown = (data['Close'] / rolling_max) - 1
    max_drawdown = drawdown.min() * 100
    
    # Win rate
    win_rate = (data['Daily_Returns'] > 0).mean() * 100
    
    gains = data['Daily_Returns'][data['Daily_Returns'] > 0].mean() * 100
    losses = data['Daily_Returns'][data['Daily_Returns'] < 0].mean() * 100
    
    beta = "N/A"
    if info and 'beta' in info:
        beta = f"{info['beta']:.2f}"
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📈 Total Return</h4>
            <h2 style="color: {'#00C851' if total_return > 0 else '#FF4444'}">{total_return:+.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📊 Volatility</h4>
            <h2>{volatility:.1f}%</h2>
            <small>Annualized</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        color = '#00C851' if sharpe_ratio > 1 else '#FFD700' if sharpe_ratio > 0.5 else '#FF4444'
        st.markdown(f"""
        <div class="metric-card">
            <h4>⚡ Sharpe Ratio</h4>
            <h2 style="color: {color}">{sharpe_ratio:.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📉 Max Drawdown</h4>
            <h2 style="color: #FF4444">{max_drawdown:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🎯 Win Rate</h4>
            <h2 style="color: {'#00C851' if win_rate > 50 else '#FF4444'}">{win_rate:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col6:
        st.markdown(f"""
        <div class="metric-card">
            <h4>⚖️ Beta</h4>
            <h2>{beta}</h2>
            <small>vs Market</small>
        </div>
        """, unsafe_allow_html=True)
    
    col7, col8, col9 = st.columns(3)
    
    with col7:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📈 Avg Gain</h4>
            <h2 style="color: #00C851">{gains:.2f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col8:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📉 Avg Loss</h4>
            <h2 style="color: #FF4444">{losses:.2f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col9:
        profit_factor = abs(gains / losses) if losses != 0 else float('inf')
        st.markdown(f"""
        <div class="metric-card">
            <h4>💰 Profit Factor</h4>
            <h2 style="color: {'#00C851' if profit_factor > 1.5 else '#FFD700' if profit_factor > 1 else '#FF4444'}">{profit_factor:.2f}</h2>
        </div>
        """, unsafe_allow_html=True)

def main():

    st.markdown("""
    <div class="main-header">
        <h1>🚀 ELITE AI STOCK MARKET DASHBOARD</h1>
        <p style="font-size: 18px; margin-top: 10px;">Professional-Grade Technical Analysis with Advanced Machine Learning</p>
        <p style="font-size: 14px; opacity: 0.8;">Real-time data • Multi-model predictions • Comprehensive analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("## 🎛️ TRADING CONTROL PANEL")
        st.markdown("---")
        
        st.markdown("### 📊 Asset Selection")
        
      
        stock_categories = {
            '🔥 Mega Cap Tech': {
                'Apple': 'AAPL', 'Microsoft': 'MSFT', 'Google': 'GOOGL', 
                'Amazon': 'AMZN', 'Meta': 'META', 'NVIDIA': 'NVDA'
            },
            '⚡ High Growth': {
                'Tesla': 'TSLA', 'Netflix': 'NFLX', 'AMD': 'AMD', 
                'Palantir': 'PLTR', 'CrowdStrike': 'CRWD'
            },
            '🏦 Financial': {
                'JPMorgan': 'JPM', 'Bank of America': 'BAC', 
                'Goldman Sachs': 'GS', 'Berkshire': 'BRK-B'
            },
            '🛍️ Consumer': {
                'Coca-Cola': 'KO', 'Nike': 'NKE', 'Walmart': 'WMT', 
                'Disney': 'DIS', 'McDonald\'s': 'MCD'
            }
        }
        
        selected_category = st.selectbox("Category:", list(stock_categories.keys()))
        stock_choice = st.selectbox("Stock:", list(stock_categories[selected_category].keys()) + ['Custom'])
        
        if stock_choice == 'Custom':
            symbol = st.text_input("Enter Symbol:", value="AAPL", max_chars=10).upper()
        else:
            symbol = stock_categories[selected_category][stock_choice]
        
        st.markdown("---")
        
       
        st.markdown("### 📅 Analysis Timeframe")
        period_options = {
            '1 Month': '1mo',
            '3 Months': '3mo', 
            '6 Months': '6mo',
            '1 Year': '1y',
            '2 Years': '2y',
            '5 Years': '5y'
        }
        period_choice = st.selectbox("Period:", list(period_options.keys()), index=3)
        period = period_options[period_choice]
        
        st.markdown("---")
   
        st.markdown("### 🔧 Analysis Modules")
        show_prediction = st.checkbox("🤖 AI Predictions", value=True)
        show_technical = st.checkbox("📈 Technical Charts", value=True)
        show_performance = st.checkbox("📊 Performance Metrics", value=True)
        show_analysis = st.checkbox("🧠 Market Analysis", value=True)
        show_comparison = st.checkbox("⚖️ Model Comparison", value=False)
        
        st.markdown("---")
        
       
        with st.expander("⚙️ Advanced Settings"):
            prediction_days = st.slider("Prediction Days:", 1, 10, 5)
            confidence_threshold = st.slider("Min Model Confidence:", 0.0, 1.0, 0.6)
        
        st.markdown("---")
        
        if st.button("🚀 ANALYZE", type="primary", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    

    analyzer = AdvancedStockAnalyzer()
    
   
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("🔍 Fetching market data...")
    progress_bar.progress(20)
    
    data, info = analyzer.fetch_stock_data(symbol, period)
    
    if data is None or data.empty:
        st.error(f"❌ Unable to fetch data for {symbol}. Please verify the symbol.")
        st.info("💡 **Suggestions:**\n- Check symbol spelling\n- Try popular symbols: AAPL, MSFT, GOOGL\n- Ensure market is open for real-time data")
        return
    
    progress_bar.progress(40)
    status_text.text("⚙️ Calculating technical indicators...")
    
   
    data = analyzer.calculate_advanced_technical_indicators(data)
    
    progress_bar.progress(60)
    status_text.text("🤖 Training AI models...")
    

    model_info = None
    if show_prediction:
        model_info = analyzer.train_ensemble_models(data)
    
    progress_bar.progress(80)
    status_text.text("📊 Generating analysis...")
    
   
    analysis = analyzer.generate_advanced_analysis(data, info, symbol, model_info)
    
    progress_bar.progress(100)
    status_text.text("✅ Analysis complete!")
    
   
    progress_bar.empty()
    status_text.empty()
    
    st.markdown("---")
    
    create_enhanced_metrics(data, info, symbol)
    
    st.markdown("---")
    

    if show_prediction and model_info:
        st.markdown("## 🔮 AI PRICE PREDICTIONS")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
        
            predictions = analyzer.predict_multiple_days(model_info, days=prediction_days)
            
            if predictions:
                current_price = data['Close'].iloc[-1]
      
                pred_cols = st.columns(min(len(predictions), 5))
                for i, pred in enumerate(predictions[:5]):
                    with pred_cols[i]:
                        change = ((pred - current_price) / current_price) * 100
                        color = '#00C851' if change > 0 else '#FF4444'
                        st.markdown(f"""
                        <div class="prediction-card">
                            <h4>Day {i+1}</h4>
                            <h2 style="color: {color}">${pred:.2f}</h2>
                            <p style="color: {color}">{change:+.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                
             
                pred_chart = create_prediction_visualization(data, predictions, symbol)
                if pred_chart:
                    st.plotly_chart(pred_chart, use_container_width=True)
        
        with col2:
            if model_info and 'model_results' in model_info:
                best_model = model_info['best_model_name']
                results = model_info['model_results'][best_model]
                
                st.markdown(f"""
                <div class="analysis-card">
                    <h3>🎯 Model Performance</h3>
                    <p><strong>Best Model:</strong> {best_model}</p>
                    <p><strong>Training R²:</strong> {results['train_r2']:.1%}</p>
                    <p><strong>Testing R²:</strong> {results['test_r2']:.1%}</p>
                    <p><strong>Confidence:</strong> {'High' if results['test_r2'] > 0.7 else 'Medium' if results['test_r2'] > 0.5 else 'Low'}</p>
                </div>
                """, unsafe_allow_html=True)
                
    
                if model_info['feature_importance']:
                    importance_df = pd.DataFrame(
                        list(model_info['feature_importance'].items()),
                        columns=['Feature', 'Importance']
                    ).sort_values('Importance', ascending=False).head(8)
                    
                    fig_importance = px.bar(
                        importance_df, 
                        x='Importance', 
                        y='Feature',
                        orientation='h',
                        title="🔍 Key Prediction Factors",
                        template='plotly_dark',
                        color='Importance',
                        color_continuous_scale='viridis'
                    )
                    fig_importance.update_layout(height=300)
                    st.plotly_chart(fig_importance, use_container_width=True)
    

    if show_comparison and model_info and 'model_results' in model_info:
        st.markdown("## ⚖️ MODEL COMPARISON")
        comparison_chart = create_model_comparison_chart(model_info['model_results'])
        if comparison_chart:
            st.plotly_chart(comparison_chart, use_container_width=True)
    
  
    if show_technical:
        st.markdown("## 📈 PROFESSIONAL TECHNICAL ANALYSIS")
        chart = create_professional_chart(data, symbol)
        st.plotly_chart(chart, use_container_width=True)
  
    if show_analysis:
        st.markdown("## 🧠 AI MARKET INTELLIGENCE")
        
    
        analysis_cols = st.columns(2)
        for i, insight in enumerate(analysis):
            with analysis_cols[i % 2]:
            
                if any(word in insight for word in ['🚀', '🔥', '📈']):
                    card_style = "border-left: 4px solid #00C851;"
                elif any(word in insight for word in ['🔻', '🔴', '📉']):
                    card_style = "border-left: 4px solid #FF4444;"
                else:
                    card_style = "border-left: 4px solid #FFD700;"
                
                st.markdown(f"""
                <div class="analysis-card" style="{card_style}">
                    {insight}
                </div>
                """, unsafe_allow_html=True)
    

    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Company Profile", "📊 Market Data", "🔧 Technical Data", "📈 Performance Analytics"])
    
    with tab1:
        if info:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### 🏢 Company Information")
                company_data = {
                    "Name": info.get('longName', 'N/A'),
                    "Sector": info.get('sector', 'N/A'),
                    "Industry": info.get('industry', 'N/A'),
                    "Country": info.get('country', 'N/A'),
                    "Employees": f"{info.get('fullTimeEmployees', 0):,}" if info.get('fullTimeEmployees') else 'N/A',
                    "Website": info.get('website', 'N/A')
                }
                
                for key, value in company_data.items():
                    st.write(f"**{key}:** {value}")
            
            with col2:
                st.markdown("### 💰 Financial Metrics")
                financial_data = {
                    "Market Cap": f"${info.get('marketCap', 0)/1e9:.1f}B" if info.get('marketCap') else 'N/A',
                    "P/E Ratio": f"{info.get('trailingPE', 0):.2f}" if info.get('trailingPE') else 'N/A',
                    "Forward P/E": f"{info.get('forwardPE', 0):.2f}" if info.get('forwardPE') else 'N/A',
                    "PEG Ratio": f"{info.get('pegRatio', 0):.2f}" if info.get('pegRatio') else 'N/A',
                    "Price/Book": f"{info.get('priceToBook', 0):.2f}" if info.get('priceToBook') else 'N/A',
                    "Dividend Yield": f"{info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else 'N/A'
                }
                
                for key, value in financial_data.items():
                    st.write(f"**{key}:** {value}")
            
            with col3:
                st.markdown("### 📊 Trading Metrics")
                trading_data = {
                    "Beta": f"{info.get('beta', 0):.2f}" if info.get('beta') else 'N/A',
                    "52W High": f"${info.get('fiftyTwoWeekHigh', 0):.2f}" if info.get('fiftyTwoWeekHigh') else 'N/A',
                    "52W Low": f"${info.get('fiftyTwoWeekLow', 0):.2f}" if info.get('fiftyTwoWeekLow') else 'N/A',
                    "Avg Volume": f"{info.get('averageVolume', 0):,}" if info.get('averageVolume') else 'N/A',
                    "Float Shares": f"{info.get('impliedSharesOutstanding', 0)/1e6:.1f}M" if info.get('impliedSharesOutstanding') else 'N/A',
                    "Short Ratio": f"{info.get('shortRatio', 0):.2f}" if info.get('shortRatio') else 'N/A'
                }
                
                for key, value in trading_data.items():
                    st.write(f"**{key}:** {value}")
    
    with tab2:
        st.markdown("### 📊 Recent Trading Data")
        display_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if 'RSI_14' in data.columns:
            display_cols.append('RSI_14')
        if 'MACD' in data.columns:
            display_cols.append('MACD')
        
        recent_data = data[display_cols].tail(20).round(3)
        recent_data.index = recent_data.index.strftime('%Y-%m-%d')
        st.dataframe(recent_data, use_container_width=True)
        
      
        col1, col2 = st.columns(2)
        with col1:
            csv = recent_data.to_csv()
            st.download_button(
                "📥 Download Recent Data (CSV)",
                csv,
                f"{symbol}_recent_data.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col2:
            full_csv = data.to_csv()
            st.download_button(
                "📥 Download Full Dataset (CSV)",
                full_csv,
                f"{symbol}_full_data.csv",
                "text/csv",
                use_container_width=True
            )
    
    with tab3:
        st.markdown("### 🔧 Technical Indicators")
        tech_cols = [col for col in data.columns if any(indicator in col for indicator in 
                    ['SMA', 'EMA', 'RSI', 'MACD', 'BB', 'ATR', 'Stoch', 'Williams', 'CCI'])]
        
        if tech_cols:
            tech_data = data[tech_cols].tail(10).round(3)
            tech_data.index = tech_data.index.strftime('%Y-%m-%d')
            st.dataframe(tech_data, use_container_width=True)
        else:
            st.warning("Technical indicators not available")
    
    with tab4:
        if show_performance:
            st.markdown("### 📈 Performance Analytics")
            
            
            data['Cumulative_Returns'] = (1 + data['Close'].pct_change()).cumprod() - 1
            
            fig_perf = go.Figure()
            fig_perf.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['Cumulative_Returns'] * 100,
                    mode='lines',
                    name='Cumulative Returns (%)',
                    line=dict(color='#00C851', width=3),
                    fill='tonexty'
                )
            )
            
            fig_perf.update_layout(
                title=f'{symbol} - Cumulative Performance',
                xaxis_title='Date',
                yaxis_title='Cumulative Return (%)',
                template='plotly_dark',
                height=400
            )
            
            st.plotly_chart(fig_perf, use_container_width=True)
            
            if len(data) > 252:
                monthly_returns = data['Close'].resample('M').last().pct_change() * 100
                monthly_returns.index = monthly_returns.index.strftime('%Y-%m')
                
                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=[monthly_returns.values],
                    x=monthly_returns.index,
                    colorscale='RdYlGn',
                    colorbar=dict(title="Return %")
                ))
                
                fig_heatmap.update_layout(
                    title='Monthly Returns Heatmap',
                    template='plotly_dark',
                    height=300
                )
                
                st.plotly_chart(fig_heatmap, use_container_width=True)
    
   
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-top: 30px;'>
        <h2 style='color: white; margin-bottom: 20px;'>🚀 ELITE AI STOCK DASHBOARD</h2>
        <div style='display: flex; justify-content: center; gap: 40px; flex-wrap: wrap; margin-bottom: 20px;'>
            <div style='text-align: center;'>
                <h3 style='color: #FFD700; margin: 0;'>⚡ Real-Time Analysis</h3>
                <p style='color: white; margin: 5px 0;'>Live market data & indicators</p>
            </div>
            <div style='text-align: center;'>
                <h3 style='color: #FFD700; margin: 0;'>🤖 AI Predictions</h3>
                <p style='color: white; margin: 5px 0;'>Multi-model forecasting</p>
            </div>
            <div style='text-align: center;'>
                <h3 style='color: #FFD700; margin: 0;'>📊 Professional Charts</h3>
                <p style='color: white; margin: 5px 0;'>Advanced technical analysis</p>
            </div>
        </div>
        <div style='border-top: 1px solid rgba(255,255,255,0.3); padding-top: 20px;'>
            <p style='color: white; font-size: 16px; margin: 10px 0;'>
                <strong>Created by Het Patel</strong> | Professional Quantitative Developer
            </p>
            <p style='color: rgba(255,255,255,0.8); font-size: 14px; margin: 0;'>
                ⚠️ <em>For educational and research purposes only. Not financial advice.</em>
            </p>
            <p style='color: rgba(255,255,255,0.6); font-size: 12px; margin: 10px 0;'>
                Powered by yFinance • Plotly • Streamlit • Scikit-learn
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()