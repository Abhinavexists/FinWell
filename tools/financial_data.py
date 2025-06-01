import logging
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Any
from core.config import settings

logger = logging.getLogger(__name__)

class FinancialDataTool:    
    def __init__(self):
        self.finnhub_key = settings.finnhub_api_key
    
    def convert_dataframe_to_dict(self, df: pd.DataFrame) -> Dict[str, Any]:
        if df.empty:
            return {}
        
        df_copy = df.copy()
        if isinstance(df_copy.index, pd.DatetimeIndex):
            df_copy.index = df_copy.index.strftime('%Y-%m-%d')
        
        result = df_copy.to_dict()
        
        def convert_value(obj):
            if obj is None or pd.isna(obj):
                return None
            elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
                return str(obj)
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif hasattr(obj, 'item') and callable(getattr(obj, 'item')):
                try:
                    return obj.item()
                except (ValueError, TypeError):
                    return str(obj)
            elif isinstance(obj, dict):
                new_dict = {}
                for k, v in obj.items():
                    if isinstance(k, (pd.Timestamp, pd.Timedelta)):
                        new_key = str(k)
                    elif hasattr(k, 'item') and callable(getattr(k, 'item')):
                        try:
                            new_key = k.item()
                        except (ValueError, TypeError):
                            new_key = str(k)
                    else:
                        new_key = k
                    new_dict[new_key] = convert_value(v)
                return new_dict
            elif isinstance(obj, (list, tuple)):
                return [convert_value(v) for v in obj]
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            else:
                return obj
        
        return convert_value(result)
    
    def get_stock_data(self, 
                      symbol: str, 
                      period: str = "1y") -> Dict[str, Any]:
        try:
            stock = yf.Ticker(symbol)
            
            hist = stock.history(period=period)
            info = stock.info
            
            current_price = hist['Close'].iloc[-1] if not hist.empty else None
            price_change = hist['Close'].iloc[-1] - hist['Close'].iloc[-2] if len(hist) > 1 else 0
            price_change_pct = (price_change / hist['Close'].iloc[-2] * 100) if len(hist) > 1 else 0
            
            financials = stock.financials
            balance_sheet = stock.balance_sheet
            cashflow = stock.cashflow
            
            return {
                "symbol": symbol,
                "current_price": float(current_price) if current_price is not None else None,
                "price_change": float(price_change) if price_change is not None else 0,
                "price_change_percent": float(price_change_pct) if price_change_pct is not None else 0,
                "historical_data": self.convert_dataframe_to_dict(hist),
                "company_info": info,
                "financials": self.convert_dataframe_to_dict(financials),
                "balance_sheet": self.convert_dataframe_to_dict(balance_sheet),
                "cashflow": self.convert_dataframe_to_dict(cashflow),
                "volume": float(hist['Volume'].iloc[-1]) if not hist.empty else None,
                "market_cap": info.get('marketCap'),
                "pe_ratio": info.get('trailingPE'),
                "dividend_yield": info.get('dividendYield'),
                "52_week_high": info.get('fiftyTwoWeekHigh'),
                "52_week_low": info.get('fiftyTwoWeekLow')
            }
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return {"error": f"Failed to fetch data for {symbol}: {str(e)}"}
    
    def get_market_overview(self) -> Dict[str, Any]:
        indices = {
            "S&P 500": "^GSPC",
            "Dow Jones": "^DJI", 
            "NASDAQ": "^IXIC",
            "Russell 2000": "^RUT",
            "VIX": "^VIX"
        }
        
        market_data = {}
        
        for name, symbol in indices.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="5d")
                
                if not hist.empty:
                    current = hist['Close'].iloc[-1]
                    previous = hist['Close'].iloc[-2] if len(hist) > 1 else current
                    change = current - previous
                    change_pct = (change / previous * 100) if previous != 0 else 0
                    
                    market_data[name] = {
                        "symbol": symbol,
                        "current": float(current),
                        "change": float(change),
                        "change_percent": float(change_pct)
                    }
            except Exception as e:
                logger.error(f"Error fetching {name} data: {str(e)}")
                market_data[name] = {"error": str(e)}
        
        return market_data
    
    def get_sector_performance(self) -> Dict[str, Any]:
        sector_etfs = {
            "Technology": "XLK",
            "Healthcare": "XLV",
            "Financial": "XLF",
            "Energy": "XLE",
            "Consumer Discretionary": "XLY",
            "Consumer Staples": "XLP",
            "Industrial": "XLI",
            "Materials": "XLB",
            "Real Estate": "XLRE",
            "Utilities": "XLU",
            "Communication": "XLC"
        }
        
        sector_data = {}
        
        for sector, etf in sector_etfs.items():
            try:
                data = self.get_stock_data(etf, "1mo")
                if "error" not in data:
                    sector_data[sector] = {
                        "etf_symbol": etf,
                        "current_price": data["current_price"],
                        "price_change_percent": data["price_change_percent"]
                    }
            except Exception as e:
                logger.error(f"Error fetching {sector} sector data: {str(e)}")
        
        return sector_data
    
    def search_stocks(self, query: str, limit: int = 10) -> List[Dict[str, str]]:
        try:
            ticker = yf.Ticker(query.upper())
            info = ticker.info
            
            if info and 'symbol' in info:
                return [{
                    "symbol": info.get('symbol', query.upper()),
                    "name": info.get('longName', 'Unknown'),
                    "sector": info.get('sector', 'Unknown'),
                    "industry": info.get('industry', 'Unknown')
                }]
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error searching for stocks with query '{query}': {str(e)}")
            return []

financial_data_tool = FinancialDataTool()
