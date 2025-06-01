from crewai import Agent, Task
from typing import Dict, List, Any
import logging
from datetime import datetime
from tools.financial_data import financial_data_tool
from tools.news_sentiment import news_sentiment_tool
from core.gemini import get_gemini

logger = logging.getLogger(__name__)

class DataAgent:
    def __init__(self):
        self.agent = Agent(
            role="Financial Data Collector",
            goal="Gather comprehensive financial data, market information, and news sentiment for investment analysis",
            backstory="""You are an expert financial data analyst with deep knowledge of market data sources, 
            financial metrics, and news analysis. You excel at collecting accurate, timely, and relevant 
            financial information from multiple sources and organizing it for analysis.""",
            verbose=True,
            allow_delegation=False,
            llm=get_gemini()
        )
    
    def create_data_collection_task(self, symbols: List[str], analysis_period: str = "1y") -> Task:
        return Task(
            description=f"""
            Collect comprehensive financial data for the following stocks: {', '.join(symbols)}
            
            Your tasks:
            1. Gather current stock prices, historical data, and key financial metrics
            2. Collect recent news articles and analyze sentiment for each stock
            3. Obtain market overview data including major indices and sector performance
            4. Compile company fundamental data including financials, balance sheets, and cash flow
            5. Organize all data in a structured format for analysis
            
            Analysis period: {analysis_period}
            
            Provide detailed data collection results with proper organization and any data quality notes.
            """,
            agent=self.agent,
            expected_output="Comprehensive financial data report with stock data, news sentiment, market overview, and fundamental analysis data"
        )
    
    def collect_stock_data(self, symbols: List[str], period: str = "1y") -> Dict[str, Any]:
        try:
            stock_data = {}
            
            for symbol in symbols:
                logger.info(f"Collecting data for {symbol}")
                data = financial_data_tool.get_stock_data(symbol, period)
                stock_data[symbol] = data
            
            return {
                "stocks": stock_data,
                "collection_timestamp": datetime.now().isoformat(),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error collecting stock data: {str(e)}")
            return {"error": str(e), "status": "failed"}
    
    def collect_market_data(self) -> Dict[str, Any]:
        try:
            market_overview = financial_data_tool.get_market_overview()
            sector_performance = financial_data_tool.get_sector_performance()
            
            return {
                "market_indices": market_overview,
                "sector_performance": sector_performance,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error collecting market data: {str(e)}")
            return {"error": str(e), "status": "failed"}
    
    def collect_news_sentiment(self, symbols: List[str]) -> Dict[str, Any]:
        try:
            news_data = {}
            
            for symbol in symbols:
                logger.info(f"Collecting news for {symbol}")
                news = news_sentiment_tool.get_stock_news(symbol)
                news_data[symbol] = news
            
            market_sentiment = news_sentiment_tool.get_market_sentiment_summary(symbols)
            
            return {
                "individual_news": news_data,
                "market_sentiment": market_sentiment,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error collecting news sentiment: {str(e)}")
            return {"error": str(e), "status": "failed"}
    
    def execute_data_collection(self, symbols: List[str], period: str = "1y") -> Dict[str, Any]:
        logger.info(f"Starting data collection for symbols: {symbols}")
        
        results = {
            "symbols": symbols,
            "period": period,
            "timestamp": datetime.now().isoformat()
        }
        
        stock_data = self.collect_stock_data(symbols, period)
        results["stock_data"] = stock_data
        
        market_data = self.collect_market_data()
        results["market_data"] = market_data
        
        news_data = self.collect_news_sentiment(symbols)
        results["news_sentiment"] = news_data
        
        results["summary"] = self.generate_data_summary(results)
        
        return results
    
    def generate_data_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        summary = {
            "total_symbols": len(data.get("symbols", [])),
            "data_quality": "good",
            "issues": []
        }
        
        if "stock_data" in data and data["stock_data"].get("status") != "success":
            summary["issues"].append("Stock data collection failed")
            summary["data_quality"] = "poor"
        
        if "market_data" in data and data["market_data"].get("status") != "success":
            summary["issues"].append("Market data collection failed")
            summary["data_quality"] = "fair" if summary["data_quality"] == "good" else "poor"
        
        if "news_sentiment" in data and data["news_sentiment"].get("status") != "success":
            summary["issues"].append("News sentiment collection failed")
            summary["data_quality"] = "fair" if summary["data_quality"] == "good" else "poor"
        
        return summary

data_agent = DataAgent()
