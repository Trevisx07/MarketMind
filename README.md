# 🚀 AI-Powered Stock Market Dashboard

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue)](https://docker.com/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Deployed-yellow)](https://huggingface.co/)

> **Professional-grade stock analysis with machine learning predictions and real-time technical indicators**

A comprehensive, AI-powered stock market dashboard that combines advanced technical analysis, machine learning price predictions, and intelligent market insights in a beautiful, interactive interface.

<img width="1877" height="948" alt="image" src="https://github.com/user-attachments/assets/303d1a62-33b4-4593-a60d-e85f1ec707f9" />


## ✨ Features

### 🤖 **Artificial Intelligence**
- **Machine Learning Price Prediction** - Random Forest model with 30+ technical features
- **AI Market Analysis** - Natural language insights based on technical indicators
- **Feature Importance Analysis** - Understand what drives price movements
- **Model Performance Metrics** - Train/test accuracy with confidence levels

### 📈 **Advanced Technical Analysis**
- **Professional Charts** - Multi-panel candlestick charts with technical overlays
- **20+ Technical Indicators** - RSI, MACD, Bollinger Bands, Moving Averages, Stochastic
- **Volume Analysis** - Volume trends and confirmation signals
- **Performance Metrics** - Sharpe ratio, volatility, maximum drawdown

### 🎯 **Real-Time Data**
- **Live Stock Data** - Real-time prices from Yahoo Finance
- **Multiple Timeframes** - 1M to 5Y analysis periods
- **Popular Stock Presets** - Quick access to FAANG+ stocks
- **Custom Symbol Input** - Analyze any publicly traded stock

### 🚀 **Deployment & Containerization**
- **Docker Support** - Containerized deployment for easy setup and scalability
- **Hugging Face Deployment** - Live demo available on Hugging Face Spaces

### 🎨 **Professional Interface**
- **Dark Theme** - Easy on the eyes for extended analysis
- **Responsive Design** - Works perfectly on desktop and mobile
- **Interactive Charts** - Zoom, pan, and explore data
- **Organized Tabs** - Clean separation of different analysis types


## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8 or higher
```

### Installation

#### Option 1: Standard Installation

1. **Clone the repository**
```bash
git clone https://github.com/Trevisx07/MarketMind/
cd ai-stock-dashboard
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run stock_dashboard.py
```

4. **Open your browser**
```
Navigate to http://localhost:8501
```

#### Option 2: Docker Installation

1. **Clone the repository**
```bash
git clone https://github.com/Trevisx07/MarketMind/
cd ai-stock-dashboard
```

2. **Build Docker image**
```bash
docker build -t ai-stock-dashboard .
```

3. **Run Docker container**
```bash
docker run -p 8501:8501 ai-stock-dashboard
```

4. **Open your browser**
```
Navigate to http://localhost:8501
```

#### Option 3: Try Live Demo

🤗 **[Try it on Hugging Face Spaces](https://huggingface.co/spaces/your-username/ai-stock-dashboard)** - No installation required!


## 📦 Dependencies

```
streamlit>=1.28.0
yfinance>=0.2.18
pandas>=1.5.0
numpy>=1.24.0
plotly>=5.15.0
scikit-learn>=1.3.0
```

## 🎮 How to Use

### 1. **Select Your Stock**
- Choose from popular presets (Apple, Tesla, Google, etc.)
- Or enter any stock symbol manually
- Select your preferred analysis timeframe

### 2. **Explore the Analysis**
- **Main Dashboard**: Key metrics and price changes
- **Technical Charts**: Advanced multi-panel analysis
- **Performance**: Risk metrics and cumulative returns
- **AI Predictions**: Machine learning price forecasts
- **Market Analysis**: AI-generated insights

### 3. **Understand the Insights**
- 🟢 **Green indicators**: Bullish signals
- 🔴 **Red indicators**: Bearish signals  
- 🟡 **Yellow indicators**: Neutral/mixed signals
- ⚠️ **Warning indicators**: Overbought/oversold conditions


## 🧠 Machine Learning Model

Our AI uses a **Random Forest Regressor** trained on 30+ features including:

- **Price-based features**: Returns, volatility, price changes
- **Technical indicators**: RSI, MACD, moving averages
- **Volume features**: Volume ratios and trends  
- **Lag features**: Historical price and volume data
- **Statistical features**: Rolling means and standard deviations

**Model Performance:**
- Real-time training on historical data
- Cross-validation with train/test splits
- Feature importance analysis
- Confidence metrics displayed


## 📊 Technical Indicators

| Indicator | Purpose | Interpretation |
|-----------|---------|----------------|
| **RSI** | Momentum | >70 Overbought, <30 Oversold |
| **MACD** | Trend | Signal line crossovers |
| **Bollinger Bands** | Volatility | Price vs. bands position |
| **Moving Averages** | Trend | Price vs. MA relationships |
| **Stochastic** | Momentum | %K and %D oscillator |
| **Volume** | Confirmation | Volume vs. average ratios |

## 🎯 Use Cases

### 📈 **For Traders**
- Quick technical analysis of any stock
- AI-powered price predictions for next trading day
- Volume confirmation signals
- Multiple timeframe analysis

### 💼 **For Investors**
- Long-term performance metrics
- Risk assessment (volatility, drawdown)
- Company fundamental information
- Market trend analysis

### 🎓 **For Learning**
- Understanding technical indicators
- Machine learning in finance
- Market behavior patterns
- Professional chart analysis


## ⚠️ Disclaimer

**This tool is for educational and informational purposes only.**

- Not financial advice or investment recommendations
- Past performance doesn't guarantee future results
- Always do your own research before investing
- Consider consulting with financial professionals
- Markets involve risk and potential loss of capital

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🌟 Acknowledgments

- **Yahoo Finance** for providing free stock data
- **Streamlit** for the amazing web framework
- **Plotly** for interactive visualizations
- **scikit-learn** for machine learning capabilities

## 📞 Support

If you find this project helpful, please give it a ⭐ on GitHub!

For questions or issues:
- Open an [Issue](https://github.com/Trevisx07/MarketMind/issues)

---

<div align="center">

**Built with ❤️ and Python**

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)]([https://github.com/erikthiart](https://github.com/Trevisx07/MarketMind/))
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/het-patel-65a272252/)

</div>
