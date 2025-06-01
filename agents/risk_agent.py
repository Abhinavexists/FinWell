import logging
import numpy as np
from crewai import Agent, Task
from typing import Dict, List, Any, Tuple
from core.gemini import get_gemini

logger = logging.getLogger(__name__)

class RiskAgent:
    def __init__(self):
        self.agent = Agent(
            role="Risk Assessment & Investment Advisor",
            goal="Evaluate investment risks, calculate risk metrics, and provide actionable investment recommendations",
            backstory="""You are a senior risk management specialist and investment advisor with extensive experience 
            in portfolio management, risk assessment, and investment strategy. You excel at identifying potential risks, 
            calculating risk-adjusted returns, and providing clear, actionable investment recommendations based on 
            comprehensive analysis.""",
            verbose=True,
            allow_delegation=False,
            llm=get_gemini()
        )
    
    def create_risk_assessment_task(self, analysis_results: Dict[str, Any]) -> Task:
        symbols = analysis_results.get("symbols", [])
        
        return Task(
            description=f"""
            Perform comprehensive risk assessment and generate investment recommendations for: {', '.join(symbols)}
            
            Your assessment should include:
            1. Calculate risk metrics including volatility, beta, VaR, and Sharpe ratio
            2. Identify key risk factors for each investment
            3. Assess portfolio diversification and correlation risks
            4. Evaluate market timing and macroeconomic risks
            5. Consider liquidity and operational risks
            6. Generate specific investment recommendations with position sizing
            7. Provide risk management strategies and stop-loss levels
            8. Create portfolio allocation suggestions
            
            Use the provided technical analysis, fundamental analysis, and market data to make informed risk assessments.
            
            Provide clear, actionable recommendations with specific risk parameters and investment guidelines.
            """,
            agent=self.agent,
            expected_output="Comprehensive risk assessment report with investment recommendations, risk metrics, and portfolio allocation guidance"
        )
    
    def calculate_risk_metrics(self, symbols: List[str], stock_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            risk_metrics = {}
            
            for symbol in symbols:
                if symbol not in stock_data or "error" in stock_data[symbol]:
                    risk_metrics[symbol] = {"error": "No data available for risk calculation"}
                    continue
                
                logger.info(f"Calculating risk metrics for {symbol}")
                
                hist_data = stock_data[symbol].get("historical_data", {})
                if not hist_data or "Close" not in hist_data:
                    risk_metrics[symbol] = {"error": "Insufficient historical data"}
                    continue
                
                prices = np.array(list(hist_data["Close"].values()))
                returns = np.diff(prices) / prices[:-1]
                
                metrics = {
                    "volatility": {
                        "daily": np.std(returns),
                        "annualized": np.std(returns) * np.sqrt(252)
                    },
                    "var": {
                        "95_percent": np.percentile(returns, 5),
                        "99_percent": np.percentile(returns, 1)
                    },
                    "max_drawdown": self.calculate_max_drawdown(prices),
                    "sharpe_ratio": self.calculate_sharpe_ratio(returns),
                    "beta": self.calculate_beta(returns, symbol),
                    "risk_score": self.calculate_risk_score(returns, prices)
                }
                
                risk_metrics[symbol] = metrics
            
            return {
                "risk_metrics": risk_metrics,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            return {"error": str(e), "status": "failed"}
    
    def assess_portfolio_risk(self, symbols: List[str], risk_metrics: Dict[str, Any]) -> Dict[str, Any]:
        try:
            individual_risks = []
            valid_symbols = []
            
            for symbol in symbols:
                if symbol in risk_metrics and "error" not in risk_metrics[symbol]:
                    risk_score = risk_metrics[symbol].get("risk_score", 50)
                    individual_risks.append(risk_score)
                    valid_symbols.append(symbol)
            
            if not individual_risks:
                return {"error": "No valid risk data for portfolio assessment"}
            
            portfolio_risk = {
                "average_risk_score": np.mean(individual_risks),
                "risk_concentration": np.std(individual_risks),
                "diversification_score": self.calculate_diversification_score(valid_symbols),
                "portfolio_volatility": self.estimate_portfolio_volatility(risk_metrics, valid_symbols),
                "risk_level": self.classify_risk_level(np.mean(individual_risks))
            }
            
            return {
                "portfolio_risk": portfolio_risk,
                "recommendations": self.generate_portfolio_recommendations(portfolio_risk),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error assessing portfolio risk: {str(e)}")
            return {"error": str(e), "status": "failed"}
    
    def generate_investment_recommendations(self, 
                                          analysis_results: Dict[str, Any], 
                                          risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        try:
            symbols = analysis_results.get("symbols", [])
            technical_analysis = analysis_results.get("technical_analysis", {}).get("technical_analysis", {})
            fundamental_analysis = analysis_results.get("fundamental_analysis", {}).get("fundamental_analysis", {})
            risk_metrics = risk_assessment.get("risk_metrics", {})
            
            recommendations = {}
            
            for symbol in symbols:
                logger.info(f"Generating recommendations for {symbol}")
                
                tech_data = technical_analysis.get(symbol, {})
                fund_data = fundamental_analysis.get(symbol, {})
                risk_data = risk_metrics.get(symbol, {})
                
                if "error" in tech_data or "error" in fund_data or "error" in risk_data:
                    recommendations[symbol] = {
                        "recommendation": "INSUFFICIENT_DATA",
                        "reason": "Incomplete analysis data"
                    }
                    continue
                
                technical_signal = tech_data.get("trading_signals", {}).get("overall_signal", "NEUTRAL")
                technical_confidence = tech_data.get("trading_signals", {}).get("confidence", 0.5)
                fundamental_score = fund_data.get("fundamental_score", 50)
                risk_score = risk_data.get("risk_score", 50)
                
                recommendation = self.generate_stock_recommendation(
                    technical_signal, technical_confidence, fundamental_score, risk_score
                )
                
                position_size = self.calculate_position_size(risk_score, technical_confidence)
                
                current_price = tech_data.get("current_price")
                stop_loss, target_price = self.calculate_price_targets(
                    current_price, technical_signal, risk_score
                )
                
                recommendations[symbol] = {
                    "recommendation": recommendation["action"],
                    "confidence": recommendation["confidence"],
                    "reasoning": recommendation["reasoning"],
                    "position_size": position_size,
                    "current_price": current_price,
                    "target_price": target_price,
                    "stop_loss": stop_loss,
                    "risk_level": self.classify_risk_level(risk_score),
                    "time_horizon": recommendation["time_horizon"],
                    "key_risks": self.identify_key_risks(tech_data, fund_data, risk_data)
                }
            
            portfolio_allocation = self.generate_portfolio_allocation(recommendations)
            
            return {
                "individual_recommendations": recommendations,
                "portfolio_allocation": portfolio_allocation,
                "overall_strategy": self.generate_overall_strategy(recommendations),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return {"error": str(e), "status": "failed"}
    
    def execute_risk_assessment(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Starting comprehensive risk assessment")
        
        symbols = analysis_results.get("symbols", [])
        
        technical_analysis = analysis_results.get("technical_analysis", {}).get("technical_analysis", {})
        
        stock_data = {}
        for symbol in symbols:
            if symbol in technical_analysis:
                tech_data = technical_analysis[symbol]
                if "error" not in tech_data:
                    stock_data[symbol] = {
                        "historical_data": {"Close": {}},
                        "current_price": tech_data.get("current_price")
                    }
                    
                    logger.warning(f"Risk calculation for {symbol} may be limited without direct access to historical data")
                else:
                    stock_data[symbol] = {"error": tech_data.get("error", "No data available")}
        
        results = {
            "symbols": symbols,
            "assessment_timestamp": analysis_results.get("analysis_timestamp")
        }
        
        if not any("historical_data" in data and data["historical_data"]["Close"] 
                  for data in stock_data.values() if "error" not in data):
            logger.warning("Insufficient historical data for detailed risk analysis, using fallback assessment")
            risk_metrics = self._generate_fallback_risk_metrics(symbols, technical_analysis)
        else:
            risk_metrics = self.calculate_risk_metrics(symbols, stock_data)
        
        results["risk_metrics"] = risk_metrics
        
        portfolio_risk = self.assess_portfolio_risk(symbols, risk_metrics.get("risk_metrics", {}))
        results["portfolio_risk"] = portfolio_risk
        
        recommendations = self.generate_investment_recommendations(analysis_results, risk_metrics)
        results["recommendations"] = recommendations
        
        risk_management = self.generate_risk_management_plan(results)
        results["risk_management"] = risk_management
        
        return results
    
    def _generate_fallback_risk_metrics(self, symbols: List[str], technical_analysis: Dict[str, Any]) -> Dict[str, Any]:
        risk_metrics = {}
        
        for symbol in symbols:
            tech_data = technical_analysis.get(symbol, {})
            if "error" in tech_data:
                risk_metrics[symbol] = {"error": "No technical data available"}
                continue
            
            technical_signal = tech_data.get("trading_signals", {}).get("overall_signal", "NEUTRAL")
            confidence = tech_data.get("trading_signals", {}).get("confidence", 0.5)
            
            base_risk = 50  # Default moderate risk
            
            if technical_signal == "BULLISH" and confidence > 0.7:
                risk_score = 35  # Lower risk for strong bullish signals
            elif technical_signal == "BEARISH" and confidence > 0.7:
                risk_score = 75  # Higher risk for strong bearish signals
            elif confidence < 0.3:
                risk_score = 65  # Higher risk for uncertain signals
            else:
                risk_score = base_risk
        
            risk_metrics[symbol] = {
                "volatility": {
                    "daily": 0.02,
                    "annualized": 0.32
                },
                "var": {
                    "95_percent": -0.03,
                    "99_percent": -0.05
                },
                "max_drawdown": -0.15,
                "sharpe_ratio": 0.8,
                "beta": 1.1,
                "risk_score": risk_score,
                "estimation_note": "Risk metrics estimated from technical analysis due to limited historical data"
            }
        
        return {
            "risk_metrics": risk_metrics,
            "status": "success"
        }
    
    def calculate_max_drawdown(self, prices: np.ndarray) -> float:
        peak = np.maximum.accumulate(prices)
        drawdown = (prices - peak) / peak
        return float(np.min(drawdown))
    
    def calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        excess_returns = np.mean(returns) - (risk_free_rate / 252)  # Daily risk-free rate
        return float(excess_returns / np.std(returns) * np.sqrt(252))
    
    def calculate_beta(self, returns: np.ndarray, symbol: str) -> float:
        market_volatility = 0.16  # Assumed market volatility
        stock_volatility = np.std(returns) * np.sqrt(252)
        
        return float(stock_volatility / market_volatility) if market_volatility > 0 else 1.0
    
    def calculate_risk_score(self, returns: np.ndarray, prices: np.ndarray) -> float:
        if len(returns) == 0:
            return 50.0
        
        volatility = np.std(returns) * np.sqrt(252)
        max_dd = abs(self.calculate_max_drawdown(prices))
        
        vol_score = min(volatility * 100, 50)  # Cap at 50
        dd_score = min(max_dd * 100, 50)      
        
        return float(vol_score + dd_score)
    
    def calculate_diversification_score(self, symbols: List[str]) -> float:
        num_stocks = len(symbols)
        if num_stocks >= 10:
            return 90.0
        elif num_stocks >= 5:
            return 70.0
        elif num_stocks >= 3:
            return 50.0
        else:
            return 20.0
    
    def estimate_portfolio_volatility(self, risk_metrics: Dict[str, Any], symbols: List[str]) -> float:
        volatilities = []
        for symbol in symbols:
            if symbol in risk_metrics and "volatility" in risk_metrics[symbol]:
                vol = risk_metrics[symbol]["volatility"].get("annualized", 0.2)
                volatilities.append(vol)
        
        if not volatilities:
            return 0.2
        
        avg_vol = np.mean(volatilities)
        diversification_benefit = 0.8 if len(volatilities) > 1 else 1.0
        
        return float(avg_vol * diversification_benefit)
    
    def classify_risk_level(self, risk_score: float) -> str:
        if risk_score < 30:
            return "LOW"
        elif risk_score < 60:
            return "MODERATE"
        elif risk_score < 80:
            return "HIGH"
        else:
            return "VERY_HIGH"
    
    def generate_portfolio_recommendations(self, portfolio_risk: Dict[str, Any]) -> List[str]:
        recommendations = []
        
        risk_level = portfolio_risk.get("risk_level", "MODERATE")
        diversification = portfolio_risk.get("diversification_score", 50)
        
        if risk_level in ["HIGH", "VERY_HIGH"]:
            recommendations.append("Consider reducing position sizes due to high portfolio risk")
            recommendations.append("Implement strict stop-loss orders")
        
        if diversification < 50:
            recommendations.append("Increase diversification across sectors and asset classes")
        
        if portfolio_risk.get("risk_concentration", 0) > 20:
            recommendations.append("Reduce concentration risk by rebalancing positions")
        
        return recommendations
    
    def generate_stock_recommendation(self, 
                                     technical_signal: str, 
                                     technical_confidence: float,
                                     fundamental_score: float, 
                                     risk_score: float) -> Dict[str, Any]:
        
        tech_weight = 0.4
        fund_weight = 0.4
        risk_weight = 0.2
        
        tech_score = 80 if technical_signal == "BULLISH" else 20 if technical_signal == "BEARISH" else 50
        tech_score *= technical_confidence
        
        risk_adjusted_score = 100 - risk_score
        
        combined_score = (tech_score * tech_weight + 
                         fundamental_score * fund_weight + 
                         risk_adjusted_score * risk_weight)
        
        if combined_score >= 70:
            action = "STRONG_BUY"
            time_horizon = "3-6 months"
        elif combined_score >= 60:
            action = "BUY"
            time_horizon = "1-3 months"
        elif combined_score >= 40:
            action = "HOLD"
            time_horizon = "Monitor"
        elif combined_score >= 30:
            action = "SELL"
            time_horizon = "1-2 weeks"
        else:
            action = "STRONG_SELL"
            time_horizon = "Immediate"
        
        confidence = min(technical_confidence + 0.2, 1.0)
        
        reasoning = f"Technical: {technical_signal} ({technical_confidence:.1%}), "
        reasoning += f"Fundamental Score: {fundamental_score:.0f}, "
        reasoning += f"Risk Score: {risk_score:.0f}"
        
        return {
            "action": action,
            "confidence": confidence,
            "reasoning": reasoning,
            "time_horizon": time_horizon
        }
    
    def calculate_position_size(self, risk_score: float, confidence: float) -> float:
        base_size = 10.0
        
        risk_multiplier = max(0.2, (100 - risk_score) / 100)
        confidence_multiplier = confidence
        
        position_size = base_size * risk_multiplier * confidence_multiplier
        
        return min(max(position_size, 1.0), 25.0)
    
    def calculate_price_targets(self, 
                               current_price: float, 
                               signal: str, 
                               risk_score: float) -> Tuple[float, float]:
        if not current_price:
            return None, None
        
        stop_loss_pct = 0.05 + (risk_score / 100) * 0.15
        
        if signal == "BULLISH":
            target_pct = 0.15 + (1 - risk_score / 100) * 0.15
        elif signal == "BEARISH":
            target_pct = -0.10
        else:
            target_pct = 0.05
        
        stop_loss = current_price * (1 - stop_loss_pct)
        target_price = current_price * (1 + target_pct)
        
        return round(stop_loss, 2), round(target_price, 2)
    
    def identify_key_risks(self, tech_data: Dict, fund_data: Dict, risk_data: Dict) -> List[str]:
        risks = []
        
        if tech_data.get("trading_signals", {}).get("overall_signal") == "BEARISH":
            risks.append("Negative technical momentum")
        
        fund_score = fund_data.get("fundamental_score", 50)
        if fund_score < 40:
            risks.append("Weak fundamental metrics")
        
        risk_score = risk_data.get("risk_score", 50)
        if risk_score > 70:
            risks.append("High volatility and drawdown risk")
        
        volatility = risk_data.get("volatility", {}).get("annualized", 0)
        if volatility > 0.4:
            risks.append("Extremely high volatility")
        
        return risks
    
    def generate_portfolio_allocation(self, recommendations: Dict[str, Any]) -> Dict[str, Any]:
        total_allocation = 0
        allocations = {}
        
        for symbol, rec in recommendations.items():
            if rec["recommendation"] in ["STRONG_BUY", "BUY"]:
                allocations[symbol] = rec["position_size"]
                total_allocation += rec["position_size"]
        
        if total_allocation > 100:
            for symbol in allocations:
                allocations[symbol] = (allocations[symbol] / total_allocation) * 100
        
        cash_allocation = max(0, 100 - sum(allocations.values()))
        
        return {
            "stock_allocations": allocations,
            "cash_allocation": cash_allocation,
            "total_invested": sum(allocations.values())
        }
    
    def generate_overall_strategy(self, recommendations: Dict[str, Any]) -> Dict[str, Any]:
        actions = [rec["recommendation"] for rec in recommendations.values()]
        
        buy_count = sum(1 for action in actions if action in ["BUY", "STRONG_BUY"])
        sell_count = sum(1 for action in actions if action in ["SELL", "STRONG_SELL"])
        hold_count = sum(1 for action in actions if action == "HOLD")
        
        if buy_count > sell_count:
            strategy = "AGGRESSIVE_GROWTH"
        elif sell_count > buy_count:
            strategy = "DEFENSIVE"
        else:
            strategy = "BALANCED"
        
        return {
            "strategy": strategy,
            "buy_signals": buy_count,
            "sell_signals": sell_count,
            "hold_signals": hold_count,
            "market_outlook": "BULLISH" if buy_count > sell_count else "BEARISH" if sell_count > buy_count else "NEUTRAL"
        }
    
    def generate_risk_management_plan(self, results: Dict[str, Any]) -> Dict[str, Any]:
        portfolio_risk = results.get("portfolio_risk", {}).get("portfolio_risk", {})
        recommendations = results.get("recommendations", {}).get("individual_recommendations", {})
        
        plan = {
            "position_limits": {
                "max_single_position": "25%",
                "max_sector_exposure": "40%",
                "cash_reserve": "10-20%"
            },
            "stop_loss_strategy": "Implement trailing stops for all positions",
            "rebalancing_frequency": "Monthly review, quarterly rebalancing",
            "risk_monitoring": [
                "Daily P&L monitoring",
                "Weekly portfolio risk assessment",
                "Monthly correlation analysis"
            ]
        }
        
        risk_level = portfolio_risk.get("risk_level", "MODERATE")
        if risk_level in ["HIGH", "VERY_HIGH"]:
            plan["immediate_actions"] = [
                "Reduce position sizes",
                "Implement tighter stop losses",
                "Consider hedging strategies"
            ]
        
        return plan

risk_agent = RiskAgent()
