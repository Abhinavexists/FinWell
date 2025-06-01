import json
import logging
from crewai import Crew, Process
from typing import Dict, List, Any, Optional
from datetime import datetime
from core.config import settings
from core.gemini import get_gemini
from agents.data_agent import DataAgent
from agents.analysis_agent import AnalysisAgent
from agents.risk_agent import RiskAgent

logger = logging.getLogger(__name__)

class FinancialAnalysisOrchestrator:
    def __init__(self):
        self.data_agent = DataAgent()
        self.analysis_agent = AnalysisAgent()
        self.risk_agent = RiskAgent()
        self.crew = None
        self.validate_setup()
    
    def validate_setup(self):
        if not settings.google_api_key:
            raise ValueError("Gemini not configured. Please set up API keys in .env file.")
        
        logger.info("Financial Analysis Orchestrator initialized successfully")
        logger.info(f"Available LLM providers: {get_gemini()}")
    
    def create_analysis_crew(self, symbols: List[str], analysis_period: str = "1y") -> Crew:
        data_task = self.data_agent.create_data_collection_task(symbols, analysis_period)
        
        analysis_task = self.analysis_agent.create_analysis_task({"symbols": symbols})
        analysis_task.context = [data_task]
        
        risk_task = self.risk_agent.create_risk_assessment_task({"symbols": symbols})
        risk_task.context = [data_task, analysis_task]
        
        crew = Crew(
            agents=[
                self.data_agent.agent,
                self.analysis_agent.agent,
                self.risk_agent.agent
            ],
            tasks=[data_task, analysis_task, risk_task],
            process=Process.sequential,
            verbose=True
        )
        
        return crew
    
    def analyze_stocks(self, 
                      symbols: List[str], 
                      analysis_period: str = "1y",
                      use_crew: bool = False) -> Dict[str, Any]:
        
        logger.info(f"Starting financial analysis for symbols: {symbols}")
        
        start_time = datetime.now()
        
        try:
            if use_crew:
                return self.analyze_with_crew(symbols, analysis_period)
            else:
                return self.analyze_direct(symbols, analysis_period)
                
        except Exception as e:
            logger.error(f"Error in stock analysis: {str(e)}")
            return {
                "error": str(e),
                "status": "failed",
                "timestamp": datetime.now().isoformat()
            }
        finally:
            end_time = datetime.now()
            logger.info(f"Analysis completed in {(end_time - start_time).total_seconds():.2f} seconds")
    
    def analyze_with_crew(self, symbols: List[str], analysis_period: str) -> Dict[str, Any]:
        crew = self.create_analysis_crew(symbols, analysis_period)
        
        result = crew.kickoff()
        
        return {
            "symbols": symbols,
            "analysis_period": analysis_period,
            "crew_result": result,
            "method": "crew_orchestration",
            "status": "success",
            "timestamp": datetime.now().isoformat()
        }
    
    def analyze_direct(self, symbols: List[str], analysis_period: str) -> Dict[str, Any]:
        logger.info("Step 1: Collecting financial data...")
        collected_data = self.data_agent.execute_data_collection(symbols, analysis_period)
        
        if "error" in collected_data:
            return {
                "error": f"Data collection failed: {collected_data['error']}",
                "status": "failed",
                "timestamp": datetime.now().isoformat()
            }
        
        logger.info("Step 2: Performing financial analysis...")
        analysis_results = self.analysis_agent.execute_comprehensive_analysis(collected_data)
        
        if "error" in analysis_results:
            return {
                "error": f"Analysis failed: {analysis_results['error']}",
                "status": "failed",
                "timestamp": datetime.now().isoformat()
            }
        
        logger.info("Step 3: Conducting risk assessment...")
        risk_results = self.risk_agent.execute_risk_assessment(analysis_results)
        
        if "error" in risk_results:
            return {
                "error": f"Risk assessment failed: {risk_results['error']}",
                "status": "failed",
                "timestamp": datetime.now().isoformat()
            }
        
        final_results = {
            "symbols": symbols,
            "analysis_period": analysis_period,
            "data_collection": collected_data,
            "analysis": analysis_results,
            "risk_assessment": risk_results,
            "executive_summary": self.generate_executive_summary(
                symbols, collected_data, analysis_results, risk_results
            ),
            "method": "direct_orchestration",
            "status": "success",
            "timestamp": datetime.now().isoformat()
        }
        
        return final_results
    
    def generate_executive_summary(self, 
                                  symbols: List[str],
                                  data: Dict[str, Any],
                                  analysis: Dict[str, Any], 
                                  risk: Dict[str, Any]) -> Dict[str, Any]:
        
        summary = {
            "analysis_overview": {
                "symbols_analyzed": len(symbols),
                "analysis_period": data.get("period", "Unknown"),
                "data_quality": data.get("summary", {}).get("data_quality", "Unknown"),
                "analysis_quality": analysis.get("analysis_summary", {}).get("analysis_quality", "Unknown")
            },
            "key_findings": [],
            "top_recommendations": [],
            "risk_highlights": [],
            "portfolio_summary": {}
        }
        
        market_analysis = analysis.get("market_analysis", {})
        if market_analysis.get("status") == "success":
            market_trend = market_analysis.get("market_trend", "neutral")
            summary["key_findings"].append(f"Overall market trend: {market_trend}")
            
            leading_sectors = market_analysis.get("leading_sectors", [])
            if leading_sectors:
                summary["key_findings"].append(f"Leading sectors: {', '.join(leading_sectors[:3])}")
        
        recommendations = risk.get("recommendations", {}).get("individual_recommendations", {})
        buy_recommendations = []
        sell_recommendations = []
        
        for symbol, rec in recommendations.items():
            action = rec.get("recommendation", "HOLD")
            if action in ["STRONG_BUY", "BUY"]:
                buy_recommendations.append({
                    "symbol": symbol,
                    "action": action,
                    "confidence": rec.get("confidence", 0),
                    "target_price": rec.get("target_price")
                })
            elif action in ["STRONG_SELL", "SELL"]:
                sell_recommendations.append({
                    "symbol": symbol,
                    "action": action,
                    "confidence": rec.get("confidence", 0)
                })
        
        buy_recommendations.sort(key=lambda x: x["confidence"], reverse=True)
        sell_recommendations.sort(key=lambda x: x["confidence"], reverse=True)
        
        summary["top_recommendations"] = {
            "buy": buy_recommendations[:3],
            "sell": sell_recommendations[:3]
        }
        
        portfolio_risk = risk.get("portfolio_risk", {}).get("portfolio_risk", {})
        if portfolio_risk:
            risk_level = portfolio_risk.get("risk_level", "UNKNOWN")
            summary["risk_highlights"].append(f"Portfolio risk level: {risk_level}")
            
            avg_risk = portfolio_risk.get("average_risk_score", 0)
            summary["risk_highlights"].append(f"Average risk score: {avg_risk:.1f}/100")
        
        portfolio_allocation = risk.get("recommendations", {}).get("portfolio_allocation", {})
        if portfolio_allocation:
            summary["portfolio_summary"] = {
                "recommended_stock_allocation": f"{portfolio_allocation.get('total_invested', 0):.1f}%",
                "recommended_cash_allocation": f"{portfolio_allocation.get('cash_allocation', 0):.1f}%",
                "number_of_positions": len(portfolio_allocation.get('stock_allocations', {}))
            }
        
        overall_strategy = risk.get("recommendations", {}).get("overall_strategy", {})
        if overall_strategy:
            summary["investment_strategy"] = overall_strategy.get("strategy", "BALANCED")
            summary["market_outlook"] = overall_strategy.get("market_outlook", "NEUTRAL")
        
        return summary
    
    def get_quick_analysis(self, symbol: str) -> Dict[str, Any]:
        logger.info(f"Performing quick analysis for {symbol}")
        
        try:
            stock_data = self.data_agent.collect_stock_data([symbol], "3mo")
            news_data = self.data_agent.collect_news_sentiment([symbol])
            
            technical_analysis = self.analysis_agent.perform_technical_analysis([symbol], "3mo")
            
            risk_metrics = self.risk_agent.calculate_risk_metrics([symbol], 
                                                                stock_data.get("stocks", {}))
            
            return {
                "symbol": symbol,
                "quick_analysis": {
                    "current_price": stock_data.get("stocks", {}).get(symbol, {}).get("current_price"),
                    "price_change_percent": stock_data.get("stocks", {}).get(symbol, {}).get("price_change_percent"),
                    "technical_signal": technical_analysis.get("technical_analysis", {}).get(symbol, {}).get("trading_signals", {}).get("overall_signal"),
                    "risk_level": self.risk_agent.classify_risk_level(
                        risk_metrics.get("risk_metrics", {}).get(symbol, {}).get("risk_score", 50)
                    ),
                    "news_sentiment": news_data.get("market_sentiment", {}).get("symbol_sentiments", {}).get(symbol, {}).get("sentiment_label")
                },
                "status": "success",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in quick analysis for {symbol}: {str(e)}")
            return {
                "symbol": symbol,
                "error": str(e),
                "status": "failed",
                "timestamp": datetime.now().isoformat()
            }
    
    def save_analysis_results(self, results: Dict[str, Any], filename: Optional[str] = None) -> str:
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            symbols_str = "_".join(results.get("symbols", ["unknown"]))
            filename = f"financial_analysis_{symbols_str}_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Analysis results saved to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise

orchestrator = FinancialAnalysisOrchestrator()
