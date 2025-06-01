# FinWell - Multi-Agent Financial Analysis System

A sophisticated multi-agent financial analysis platform that provides comprehensive stock market analysis, risk assessment, and investment recommendations using advanced AI agents and technical indicators.

## Problem Statement

Financial markets are complex and require analysis across multiple dimensions - technical patterns, fundamental valuation, market sentiment, and risk assessment. Individual investors and financial professionals face several key challenges:

- **Information Overload**: Markets generate vast amounts of data from prices, news, financial reports, and economic indicators
- **Multi-Dimensional Analysis**: Effective investment decisions require combining technical analysis, fundamental analysis, sentiment analysis, and risk assessment
- **Time Constraints**: Manual analysis of multiple stocks across different timeframes is time-intensive
- **Emotional Bias**: Human decision-making can be influenced by emotions and cognitive biases
- **Risk Management**: Proper portfolio risk assessment requires complex mathematical calculations and historical analysis

FinWell addresses these challenges by deploying specialized AI agents that work collaboratively to provide comprehensive, unbiased, and data-driven financial analysis and investment recommendations.

## Tech Stack

### Core Technologies

- **CrewAI** - Multi-agent orchestration framework
- **Google Gemini** - Large language model for AI analysis
- **Rich** - Terminal UI and formatting
- **Click** - Command-line interface

### Data & Analysis Libraries

- **yfinance** - Yahoo Finance data retrieval
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **textblob** - Natural language processing for sentiment analysis
- **feedparser** - RSS news feed parsing
- **requests** - HTTP API interactions

### APIs & Data Sources

- **Yahoo Finance** - Stock prices, financial data
- **Google Gemini API** - AI-powered analysis
- **Finnhub API** - Professional market data and news
- **RSS Feeds** - Real-time news sentiment

## Project Structure

```bash
FinWell/
├── agents/                     # AI Agent implementations
│   ├── analysis_agent.py          # Technical & fundamental analysis
│   ├── data_agent.py              # Data collection and processing
│   └── risk_agent.py              # Risk assessment and recommendations
│
├── core/                       # Core system components
│   ├── __init__.py
│   ├── config.py                  # Configuration management
│   ├── gemini.py                  # Google Gemini integration
│   └── orchestrator.py            # Main orchestration logic
│
├── tools/                      # Analysis tools and utilities
│   ├── __init__.py
│   ├── financial_data.py          # Stock data retrieval
│   ├── news_sentiment.py          # News sentiment analysis
│   └── technical_analysis.py      # Technical indicators
│
├── main.py                        # Application entry point
├── requirements.txt               # Python dependencies
├── pyproject.toml                 # UV project configuration
├── uv.lock                        # UV lockfile                # 
```

## Agent Interactions & Workflow

FinWell uses a multi-agent architecture where specialized AI agents collaborate to provide comprehensive financial analysis:

### Agent Collaboration Flow

```bash
DataAgent → AnalysisAgent → RiskAgent → Final Report
```

### 1. **DataAgent** (Data Collection & Processing)

- **Primary Role**: Collects and preprocesses financial data
- **Data Sources**: Stock prices, financial statements, market indices, news feeds
- **Output**: Clean, structured financial datasets
- **Handoff**: Provides standardized data to AnalysisAgent

### 2. **AnalysisAgent** (Technical & Fundamental Analysis)

- **Input**: Receives processed data from DataAgent
- **Technical Analysis**: Calculates RSI, MACD, Bollinger Bands, moving averages, support/resistance
- **Fundamental Analysis**: Computes financial ratios, valuation metrics, growth rates
- **Sentiment Analysis**: Processes news sentiment and market indicators
- **Output**: Technical signals, fundamental scores, and market sentiment
- **Handoff**: Provides analysis results to RiskAgent

### 3. **RiskAgent** (Risk Assessment & Recommendations)

- **Input**: Receives analysis results from AnalysisAgent
- **Risk Calculations**: VaR, Sharpe ratio, beta, volatility, maximum drawdown
- **Portfolio Analysis**: Position sizing, allocation recommendations, correlation analysis
- **Final Scoring**: Combines technical, fundamental, and risk metrics into actionable recommendations
- **Output**: Investment recommendations (STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL) with confidence scores

### Agent Communication Protocol

- **Shared Memory**: Agents communicate through structured data objects
- **Error Handling**: Each agent validates inputs and provides fallback mechanisms
- **Logging**: All agent interactions are logged for debugging and audit trails
- **Orchestration**: CrewAI coordinates agent execution and manages workflows

## Features

### Core Capabilities

- **Multi-Agent Architecture**: Specialized AI agents for data collection, technical analysis, fundamental analysis, and risk assessment
- **Real-Time Data Integration**: Yahoo Finance, Alpha Vantage, Finnhub, and RSS news feeds
- **Fundamental Analysis**: Financial ratios, valuation metrics, profitability analysis
- **Risk Assessment**: VaR, Sharpe ratio, beta calculation, portfolio risk metrics
- **News Sentiment Analysis**: Real-time news sentiment analysis for market impact assessment
- **Portfolio Optimization**: Intelligent position sizing and allocation recommendations
- **Professional Reporting**: Rich terminal output and detailed JSON reports

### Supported Analysis Types

- **Individual Stock Analysis**: Deep-dive analysis of single stocks
- **Multi-Stock Portfolio Analysis**: Comparative analysis across multiple securities
- **Risk-Adjusted Recommendations**: BUY/SELL/HOLD recommendations with confidence scores
- **Market Trend Analysis**: Sector performance and market outlook assessment

## Prerequisites

### Package Manager (Recommended)

- **UV**: Fast Python package installer and resolver ([Install UV](https://docs.astral.sh/uv/getting-started/installation/))

### API Keys

- **Google Gemini API**: For AI-powered analysis (required)
- **Finnhub API**: Professional market data and news

## 🛠 Installation

### Using UV (Recommended)

#### 1. Install UV

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Alternative: Using pip
pip install uv
```

#### 2. Clone the Repository

```bash
git clone https://github.com/Abhinavexists/FinWell.git
cd FinWell
```

#### 3. Create Project and Install Dependencies

```bash
uv sync
```

### or

#### Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Environment Configuration

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key

FINNHUB_API_KEY=your_finnhub_key
TIMEOUT_SECONDS=30
```

#### 5. Run the Application

```bash
uv run python main.py --help
## or
python main.py --help
```

#### Expected Output

![expected_output](/images/expected_output.png)

## Quick Start

#### Basic Stock Analysis

```bash
python main.py analyze AAPL --period 1y
```

#### Expected Stock Output

![basic_stock_analysis](/images/basic_stock_analysis.png)

```bash
python main.py analyze <SYMBOLS> [OPTIONS]
```

#### Arguments

- `SYMBOLS`: Space-separated list of stock symbols (e.g., `AAPL MSFT GOOGL`)

#### Options

| Option | Description | Default | Examples |
|--------|-------------|---------|----------|
| `--period` | Analysis time period | `1y` | `1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y` |
| `--use-crew` | Use CrewAI orchestration | `use-crew` | `--use-crew` |
| `--help` | Show help message | | `--help` |

#### Multi-Stock Portfolio Analysis

```bash
python main.py analyze AAPL MSFT GOOGL NVDA --period 6mo
```

#### Expected Multi-Stock Output

![multi_stock_output](/images/multi_stock_analysis.png)

#### Advanced Analysis Options

```bash
# Use CrewAI for collaborative agent analysis
python main.py analyze AAPL --period 1y --use-crew

# Short-term trading analysis
python main.py analyze AAPL NVDA --period 5d

# Long-term investment analysis
python main.py analyze BRK-B SPY --period 5y
```

## Analysis Components

### 1. Technical Analysis

**Indicators:** RSI, MACD, Bollinger Bands, Moving Averages (SMA/EMA), Support/Resistance, Volume

### 2. Fundamental Analysis

**Metrics:** P/E, P/B, ROE, ROA, Debt Ratios, Profit Margins, Growth Rates

### 3. Risk Assessment

**Calculations:** Volatility, VaR, Maximum Drawdown, Sharpe Ratio, Beta

### 4. Investment Recommendations

**Scale:** STRONG_BUY (≥70) • BUY (≥60) • HOLD (≥40) • SELL (≥30) • STRONG_SELL (<30)

## Output Files

### JSON Analysis Reports

All analyses generate detailed JSON reports saved automatically:

**Filename Format:**

```bash
financial_analysis_{SYMBOLS}_{TIMESTAMP}.json
```

**File Structure:**

```json
{
  "symbols": ["AAPL"],
  "analysis_period": "1y",
  "status": "success",
  "timestamp": "2025-06-01T19:01:16.424056",
  "data_collection": { ... },
  "analysis": { ... },
  "risk_assessment": { ... },
  "executive_summary": { ... }
}
```

### Report Sections

1. **Data Collection**: Raw financial data, news, market indices
2. **Technical Analysis**: All technical indicators and signals
3. **Fundamental Analysis**: Financial ratios and scores
4. **Risk Assessment**: Risk metrics and portfolio analysis
5. **Executive Summary**: Key findings and recommendations

### Debug Mode

```bash
export LOG_LEVEL=DEBUG
python main.py analyze AAPL --period 1y
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
