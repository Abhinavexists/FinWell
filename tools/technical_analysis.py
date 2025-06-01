import logging
import pandas as pd
import numpy as np
from typing import Dict, Any
from tools.financial_data import financial_data_tool

logger = logging.getLogger(__name__)

class TechnicalAnalysisTool:
    def __init__(self):
        pass
    
    def convert_to_native_type(self, value):
        if pd.isna(value):
            return None
        elif isinstance(value, (np.integer, np.floating)):
            return value.item()
        elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
            return str(value)
        elif hasattr(value, 'item'):
            return value.item()
        else:
            return value
    
    def analyze_stock(self, symbol: str, period: str = "6mo") -> Dict[str, Any]:
        try:
            stock_data = financial_data_tool.get_stock_data(symbol, period)
            
            if "error" in stock_data:
                return stock_data
            
            hist_data = pd.DataFrame(stock_data["historical_data"])
            
            if hist_data.empty:
                return {"error": "No historical data available"}
            
            indicators = {}
            
            indicators["moving_averages"] = self.calculate_moving_averages(hist_data)
            indicators["rsi"] = self.calculate_rsi(hist_data)
            indicators["macd"] = self.calculate_macd(hist_data)
            indicators["bollinger_bands"] = self.calculate_bollinger_bands(hist_data)
            indicators["support_resistance"] = self.find_support_resistance(hist_data)
            indicators["volume_analysis"] = self.analyze_volume(hist_data)
            signals = self.generate_signals(hist_data, indicators)
            
            return {
                "symbol": symbol,
                "analysis_period": period,
                "current_price": self.convert_to_native_type(stock_data["current_price"]),
                "technical_indicators": indicators,
                "trading_signals": signals,
                "summary": self.generate_summary(indicators, signals)
            }
            
        except Exception as e:
            logger.error(f"Error in technical analysis for {symbol}: {str(e)}")
            return {"error": f"Technical analysis failed: {str(e)}"}
    
    def calculate_moving_averages(self, df: pd.DataFrame) -> Dict[str, Any]:
        close_prices = df['Close']
        
        ma_data = {
            "sma_20": self.convert_to_native_type(close_prices.rolling(window=20).mean().iloc[-1]),
            "sma_50": self.convert_to_native_type(close_prices.rolling(window=50).mean().iloc[-1]),
            "sma_200": self.convert_to_native_type(close_prices.rolling(window=200).mean().iloc[-1]),
            "ema_12": self.convert_to_native_type(close_prices.ewm(span=12).mean().iloc[-1]),
            "ema_26": self.convert_to_native_type(close_prices.ewm(span=26).mean().iloc[-1])
        }
        
        current_price = self.convert_to_native_type(close_prices.iloc[-1])
        
        signals = []
        if ma_data["sma_20"] and current_price > ma_data["sma_20"]:
            signals.append("Price above 20-day SMA (bullish)")
        elif ma_data["sma_20"] and current_price < ma_data["sma_20"]:
            signals.append("Price below 20-day SMA (bearish)")
        
        if ma_data["sma_50"] and ma_data["sma_200"]:
            if ma_data["sma_50"] > ma_data["sma_200"]:
                signals.append("Golden Cross: 50-day SMA above 200-day SMA (bullish)")
            else:
                signals.append("Death Cross: 50-day SMA below 200-day SMA (bearish)")
        
        ma_data["signals"] = signals
        return ma_data
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> Dict[str, Any]:
        close_prices = df['Close']
        delta = close_prices.diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        current_rsi = self.convert_to_native_type(rsi.iloc[-1])
        
        signals = []
        if current_rsi > 70:
            signals.append("RSI indicates overbought condition")
        elif current_rsi < 30:
            signals.append("RSI indicates oversold condition")
        else:
            signals.append("RSI in neutral range")
        
        return {
            "current_rsi": current_rsi,
            "signals": signals,
            "interpretation": "overbought" if current_rsi > 70 else "oversold" if current_rsi < 30 else "neutral"
        }
    
    def calculate_macd(self, df: pd.DataFrame) -> Dict[str, Any]:
        close_prices = df['Close']
        
        ema_12 = close_prices.ewm(span=12).mean()
        ema_26 = close_prices.ewm(span=26).mean()
        
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9).mean()
        histogram = macd_line - signal_line
        
        current_macd = self.convert_to_native_type(macd_line.iloc[-1])
        current_signal = self.convert_to_native_type(signal_line.iloc[-1])
        current_histogram = self.convert_to_native_type(histogram.iloc[-1])
        
        signals = []
        if current_macd > current_signal:
            signals.append("MACD above signal line (bullish)")
        else:
            signals.append("MACD below signal line (bearish)")
        
        if len(histogram) > 1:
            if current_histogram > self.convert_to_native_type(histogram.iloc[-2]):
                signals.append("MACD histogram increasing (momentum building)")
            else:
                signals.append("MACD histogram decreasing (momentum weakening)")
        
        return {
            "macd_line": current_macd,
            "signal_line": current_signal,
            "histogram": current_histogram,
            "signals": signals
        }
    
    def calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: int = 2) -> Dict[str, Any]:
        close_prices = df['Close']
        
        sma = close_prices.rolling(window=period).mean()
        std = close_prices.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        current_price = self.convert_to_native_type(close_prices.iloc[-1])
        current_upper = self.convert_to_native_type(upper_band.iloc[-1])
        current_lower = self.convert_to_native_type(lower_band.iloc[-1])
        current_middle = self.convert_to_native_type(sma.iloc[-1])
        
        band_width = current_upper - current_lower
        position = (current_price - current_lower) / band_width if band_width > 0 else 0.5
        
        signals = []
        if current_price > current_upper:
            signals.append("Price above upper Bollinger Band (potentially overbought)")
        elif current_price < current_lower:
            signals.append("Price below lower Bollinger Band (potentially oversold)")
        else:
            signals.append("Price within Bollinger Bands (normal range)")
        
        return {
            "upper_band": current_upper,
            "middle_band": current_middle,
            "lower_band": current_lower,
            "current_price": current_price,
            "band_position": self.convert_to_native_type(position),
            "signals": signals
        }
    
    def find_support_resistance(self, df: pd.DataFrame) -> Dict[str, Any]:
        high_prices = df['High']
        low_prices = df['Low']
        close_prices = df['Close']
        
        recent_high = self.convert_to_native_type(high_prices.tail(20).max())
        recent_low = self.convert_to_native_type(low_prices.tail(20).min())
        current_price = self.convert_to_native_type(close_prices.iloc[-1])
        
        resistance_distance = (recent_high - current_price) / current_price * 100
        support_distance = (current_price - recent_low) / current_price * 100
        
        signals = []
        if resistance_distance < 2:
            signals.append("Price near resistance level")
        if support_distance < 2:
            signals.append("Price near support level")
        
        return {
            "resistance_level": recent_high,
            "support_level": recent_low,
            "current_price": current_price,
            "distance_to_resistance_pct": self.convert_to_native_type(resistance_distance),
            "distance_to_support_pct": self.convert_to_native_type(support_distance),
            "signals": signals
        }
    
    def analyze_volume(self, df: pd.DataFrame) -> Dict[str, Any]:
        volume = df['Volume']
        close_prices = df['Close']
        
        avg_volume = self.convert_to_native_type(volume.rolling(window=20).mean().iloc[-1])
        current_volume = self.convert_to_native_type(volume.iloc[-1])
        
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        price_change = self.convert_to_native_type(close_prices.pct_change().iloc[-1])
        
        signals = []
        if volume_ratio > 1.5:
            signals.append("High volume activity")
            if price_change > 0:
                signals.append("High volume with price increase (bullish)")
            else:
                signals.append("High volume with price decrease (bearish)")
        elif volume_ratio < 0.5:
            signals.append("Low volume activity")
        
        return {
            "current_volume": current_volume,
            "average_volume": avg_volume,
            "volume_ratio": self.convert_to_native_type(volume_ratio),
            "signals": signals
        }
    
    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        bullish_signals = 0
        bearish_signals = 0
        
        for indicator_name, indicator_data in indicators.items():
            if "signals" in indicator_data:
                for signal in indicator_data["signals"]:
                    signal_lower = signal.lower()
                    if any(word in signal_lower for word in ["bullish", "above", "increasing", "oversold"]):
                        bullish_signals += 1
                    elif any(word in signal_lower for word in ["bearish", "below", "decreasing", "overbought"]):
                        bearish_signals += 1
        
        if bullish_signals > bearish_signals:
            overall_signal = "BULLISH"
            confidence = min(bullish_signals / (bullish_signals + bearish_signals), 1.0)
        elif bearish_signals > bullish_signals:
            overall_signal = "BEARISH"
            confidence = min(bearish_signals / (bullish_signals + bearish_signals), 1.0)
        else:
            overall_signal = "NEUTRAL"
            confidence = 0.5
        
        return {
            "overall_signal": overall_signal,
            "confidence": self.convert_to_native_type(confidence),
            "bullish_indicators": bullish_signals,
            "bearish_indicators": bearish_signals,
            "recommendation": self.get_recommendation(overall_signal, confidence)
        }
    
    def get_recommendation(self, signal: str, confidence: float) -> str:
        if signal == "BULLISH" and confidence > 0.7:
            return "STRONG BUY"
        elif signal == "BULLISH" and confidence > 0.6:
            return "BUY"
        elif signal == "BEARISH" and confidence > 0.7:
            return "STRONG SELL"
        elif signal == "BEARISH" and confidence > 0.6:
            return "SELL"
        else:
            return "HOLD"
    
    def generate_summary(self, indicators: Dict[str, Any], signals: Dict[str, Any]) -> str:
        summary_parts = []
        
        summary_parts.append(f"Overall Signal: {signals['overall_signal']} (Confidence: {signals['confidence']:.1%})")
        summary_parts.append(f"Recommendation: {signals['recommendation']}")
        
        if "rsi" in indicators:
            rsi_val = indicators["rsi"]["current_rsi"]
            summary_parts.append(f"RSI: {rsi_val:.1f} ({indicators['rsi']['interpretation']})")
        
        if "moving_averages" in indicators:
            ma_20 = indicators["moving_averages"]["sma_20"]
            if ma_20:
                summary_parts.append(f"20-day SMA: ${ma_20:.2f}")
        
        return " | ".join(summary_parts)

technical_analysis_tool = TechnicalAnalysisTool()
