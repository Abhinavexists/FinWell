import logging
from crewai import Agent, Task
from typing import Dict, List, Any
from tools.technical_analysis import technical_analysis_tool
from core.gemini import get_gemini

logger = logging.getLogger(__name__)

class AnalysisAgent:
    def __init__(self):
        self.agent = Agent(
            role="Financial Analysis Specialist",
            goal="Perform comprehensive technical and fundamental analysis to identify investment opportunities and risks",
            backstory="""You are a seasoned financial analyst with expertise in both technical and fundamental analysis. 
            You have years of experience in evaluating stocks using various analytical methods including chart patterns, 
            technical indicators, financial ratios, and market trends. You excel at synthesizing complex financial data 
            into actionable insights.""",
            verbose=True,
            allow_delegation=False,
            llm=get_gemini()
        )
    
    def create_analysis_task(self, collected_data: Dict[str, Any]) -> Task:
        symbols = collected_data.get("symbols", [])
        
        return Task(
            description=f"""
            Perform comprehensive financial analysis on the collected data for: {', '.join(symbols)}
            
            Your analysis should include:
            1. Technical analysis using indicators like RSI, MACD, moving averages, and Bollinger Bands
            2. Fundamental analysis of financial metrics, ratios, and company performance
            3. Market trend analysis and sector comparison
            4. News sentiment impact assessment
            5. Identification of key strengths, weaknesses, opportunities, and threats
            6. Price target estimation and trend predictions
            
            Use the provided financial data, market data, and news sentiment to create detailed analysis reports.
            
            Provide specific, actionable insights with supporting evidence from the data.
            """,
            agent=self.agent,
            expected_output="Detailed financial analysis report with technical indicators, fundamental metrics, market analysis, and investment insights"
        )
    
    def perform_technical_analysis(self, symbols: List[str], period: str = "6mo") -> Dict[str, Any]:
        try:
            technical_results = {}
            
            for symbol in symbols:
                logger.info(f"Performing technical analysis for {symbol}")
                analysis = technical_analysis_tool.analyze_stock(symbol, period)
                technical_results[symbol] = analysis
            
            return {
                "technical_analysis": technical_results,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error in technical analysis: {str(e)}")
            return {"error": str(e), "status": "failed"}
    
    def perform_fundamental_analysis(self, stock_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            fundamental_results = {}
            
            for symbol, data in stock_data.items():
                if "error" in data:
                    fundamental_results[symbol] = {"error": data["error"]}
                    continue
                
                logger.info(f"Performing fundamental analysis for {symbol}")
                
                info = data.get("company_info", {})
                
                analysis = {
                    "valuation_metrics": {
                        "market_cap": info.get("marketCap"),
                        "pe_ratio": info.get("trailingPE"),
                        "forward_pe": info.get("forwardPE"),
                        "peg_ratio": info.get("pegRatio"),
                        "price_to_book": info.get("priceToBook"),
                        "price_to_sales": info.get("priceToSalesTrailing12Months"),
                        "enterprise_value": info.get("enterpriseValue"),
                        "ev_to_revenue": info.get("enterpriseToRevenue"),
                        "ev_to_ebitda": info.get("enterpriseToEbitda")
                    },
                    "profitability_metrics": {
                        "profit_margin": info.get("profitMargins"),
                        "operating_margin": info.get("operatingMargins"),
                        "return_on_assets": info.get("returnOnAssets"),
                        "return_on_equity": info.get("returnOnEquity"),
                        "revenue_growth": info.get("revenueGrowth"),
                        "earnings_growth": info.get("earningsGrowth")
                    },
                    "financial_health": {
                        "total_cash": info.get("totalCash"),
                        "total_debt": info.get("totalDebt"),
                        "debt_to_equity": info.get("debtToEquity"),
                        "current_ratio": info.get("currentRatio"),
                        "quick_ratio": info.get("quickRatio"),
                        "free_cashflow": info.get("freeCashflow")
                    },
                    "dividend_info": {
                        "dividend_yield": info.get("dividendYield"),
                        "dividend_rate": info.get("dividendRate"),
                        "payout_ratio": info.get("payoutRatio"),
                        "five_year_avg_dividend_yield": info.get("fiveYearAvgDividendYield")
                    }
                }
                
                analysis["fundamental_score"] = self.calculate_fundamental_score(analysis)
                analysis["fundamental_summary"] = self.generate_fundamental_summary(analysis)
                
                fundamental_results[symbol] = analysis
            
            return {
                "fundamental_analysis": fundamental_results,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error in fundamental analysis: {str(e)}")
            return {"error": str(e), "status": "failed"}
    
    def analyze_market_trends(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            market_indices = market_data.get("market_indices", {})
            sector_performance = market_data.get("sector_performance", {})
            
            market_trend = "neutral"
            positive_indices = 0
            total_indices = 0
            
            for index_data in market_indices:
                if "error" not in index_data:
                    total_indices += 1
                    if index_data.get("change_percent", 0) > 0:
                        positive_indices += 1
            
            if total_indices > 0:
                positive_ratio = positive_indices / total_indices
                if positive_ratio > 0.6:
                    market_trend = "bullish"
                elif positive_ratio < 0.4:
                    market_trend = "bearish"
            
            sector_trends = {}
            for sector, data in sector_performance.items():
                change_pct = data.get("price_change_percent", 0)
                if change_pct > 2:
                    sector_trends[sector] = "strong_positive"
                elif change_pct > 0:
                    sector_trends[sector] = "positive"
                elif change_pct > -2:
                    sector_trends[sector] = "neutral"
                else:
                    sector_trends[sector] = "negative"
            
            return {
                "market_trend": market_trend,
                "market_sentiment": "bullish" if positive_ratio > 0.5 else "bearish",
                "sector_trends": sector_trends,
                "leading_sectors": self.get_leading_sectors(sector_performance),
                "lagging_sectors": self.get_lagging_sectors(sector_performance),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error in market trend analysis: {str(e)}")
            return {"error": str(e), "status": "failed"}
    
    def execute_comprehensive_analysis(self, collected_data: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Starting comprehensive financial analysis")
        
        symbols = collected_data.get("symbols", [])
        stock_data = collected_data.get("stock_data", {}).get("stocks", {})
        market_data = collected_data.get("market_data", {})
        news_sentiment = collected_data.get("news_sentiment", {})
        
        results = {
            "symbols": symbols,
            "analysis_timestamp": collected_data.get("timestamp")
        }
        
        technical_analysis = self.perform_technical_analysis(symbols)
        results["technical_analysis"] = technical_analysis
        
        fundamental_analysis = self.perform_fundamental_analysis(stock_data)
        results["fundamental_analysis"] = fundamental_analysis
        
        market_analysis = self.analyze_market_trends(market_data)
        results["market_analysis"] = market_analysis
        
        sentiment_analysis = self.analyze_sentiment_impact(news_sentiment)
        results["sentiment_analysis"] = sentiment_analysis
        
        results["analysis_summary"] = self.generate_analysis_summary(results)
        
        return results
    
    def calculate_fundamental_score(self, analysis: Dict[str, Any]) -> float:
        score = 50
        
        valuation = analysis.get("valuation_metrics", {})
        profitability = analysis.get("profitability_metrics", {})
        health = analysis.get("financial_health", {})
        
        pe_ratio = valuation.get("pe_ratio")
        if pe_ratio:
            if 10 <= pe_ratio <= 20:
                score += 10
            elif pe_ratio < 10 or pe_ratio > 30:
                score -= 10
        
        roe = profitability.get("return_on_equity")
        if roe and roe > 0.15:
            score += 15
        elif roe and roe < 0.05:
            score -= 15
        
        profit_margin = profitability.get("profit_margin")
        if profit_margin and profit_margin > 0.1:
            score += 10
        elif profit_margin and profit_margin < 0:
            score -= 20
        
        debt_to_equity = health.get("debt_to_equity")
        if debt_to_equity and debt_to_equity < 0.5:
            score += 10
        elif debt_to_equity and debt_to_equity > 2:
            score -= 15
        
        current_ratio = health.get("current_ratio")
        if current_ratio and current_ratio > 1.5:
            score += 5
        elif current_ratio and current_ratio < 1:
            score -= 10
        
        return max(0, min(100, score))
    
    def generate_fundamental_summary(self, analysis: Dict[str, Any]) -> str:
        score = analysis.get("fundamental_score", 50)
        
        if score >= 70:
            return "Strong fundamental profile with good valuation and financial health"
        elif score >= 60:
            return "Good fundamental profile with some positive indicators"
        elif score >= 40:
            return "Mixed fundamental profile with both strengths and weaknesses"
        else:
            return "Weak fundamental profile with concerning metrics"
    
    def get_leading_sectors(self, sector_performance: Dict[str, Any]) -> List[str]:
        sectors = []
        for sector, data in sector_performance.items():
            change_pct = data.get("price_change_percent", 0)
            sectors.append((sector, change_pct))
        
        sectors.sort(key=lambda x: x[1], reverse=True)
        return [sector for sector, _ in sectors[:3]]
    
    def get_lagging_sectors(self, sector_performance: Dict[str, Any]) -> List[str]:
        sectors = []
        for sector, data in sector_performance.items():
            change_pct = data.get("price_change_percent", 0)
            sectors.append((sector, change_pct))
        
        sectors.sort(key=lambda x: x[1])
        return [sector for sector, _ in sectors[:3]]
    
    def analyze_sentiment_impact(self, news_sentiment: Dict[str, Any]) -> Dict[str, Any]:
        try:
            market_sentiment = news_sentiment.get("market_sentiment", {})
            individual_news = news_sentiment.get("individual_news", {})
            
            sentiment_impact = {}
            
            for symbol, news_list in individual_news.items():
                if not news_list:
                    continue
                
                sentiments = [article.get("sentiment", {}) for article in news_list if "sentiment" in article]
                
                if sentiments:
                    avg_polarity = sum(s.get("polarity", 0) for s in sentiments) / len(sentiments)
                    
                    impact_level = "neutral"
                    if avg_polarity > 0.2:
                        impact_level = "positive"
                    elif avg_polarity < -0.2:
                        impact_level = "negative"
                    
                    sentiment_impact[symbol] = {
                        "sentiment_score": avg_polarity,
                        "impact_level": impact_level,
                        "article_count": len(sentiments),
                        "recommendation": "Consider positive sentiment" if avg_polarity > 0.1 else "Monitor negative sentiment" if avg_polarity < -0.1 else "Neutral sentiment impact"
                    }
            
            return {
                "individual_sentiment_impact": sentiment_impact,
                "overall_market_sentiment": market_sentiment.get("overall_sentiment", {}),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment impact: {str(e)}")
            return {"error": str(e), "status": "failed"}
    
    def generate_analysis_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        symbols = results.get("symbols", [])
        
        summary = {
            "total_symbols_analyzed": len(symbols),
            "analysis_quality": "good",
            "key_findings": [],
            "overall_market_outlook": "neutral"
        }
        
        issues = []
        if results.get("technical_analysis", {}).get("status") != "success":
            issues.append("Technical analysis incomplete")
        if results.get("fundamental_analysis", {}).get("status") != "success":
            issues.append("Fundamental analysis incomplete")
        if results.get("market_analysis", {}).get("status") != "success":
            issues.append("Market analysis incomplete")
        
        if issues:
            summary["analysis_quality"] = "partial"
            summary["issues"] = issues
        
        market_trend = results.get("market_analysis", {}).get("market_trend", "neutral")
        summary["overall_market_outlook"] = market_trend
        
        return summary

analysis_agent = AnalysisAgent()
