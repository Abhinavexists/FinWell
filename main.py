import json
import typer
import logging
from typing import List, Dict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from core.orchestrator import orchestrator
from core.config import settings
from tools.financial_data import financial_data_tool

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

console = Console()
app = typer.Typer(help="FinWell - Multi-Agent Investment Analysis System")

def validate_setup():
    if not settings.google_api_key:
        console.print("[red]Gemini not configured![/red]")
        console.print("Please set up the GOOGLE_API_KEY in your .env file.")
        raise typer.Exit(1)
    
    console.print("[green]Gemini setup validated[/green]")

@app.command()
def analyze(
    symbols: List[str] = typer.Argument(..., help="Stock symbols to analyze (e.g., AAPL GOOGL MSFT)"),
    period: str = typer.Option("1y", help="Analysis period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y)"),
    save_results: bool = typer.Option(True, help="Save results to JSON file"),
    use_crew: bool = typer.Option(False, help="Use CrewAI orchestration (experimental)"),
    output_format: str = typer.Option("rich", help="Output format: rich, json, summary")
):
    
    validate_setup()
    
    symbols = [symbol.upper() for symbol in symbols]

    console.print(f"\n[bold blue] Starting Financial Analysis[/bold blue]")
    console.print(f"Symbols: {', '.join(symbols)}")
    console.print(f"Period: {period}")
    console.print(f"Method: {'CrewAI Orchestration' if use_crew else 'Direct Orchestration'}")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        task = progress.add_task("Analyzing stocks...", total=None)
        
        try:
            results = orchestrator.analyze_stocks(
                symbols=symbols,
                analysis_period=period,
                use_crew=use_crew
            )
            
            progress.update(task, description="Analysis complete!")
            
            if results.get("status") == "failed":
                console.print(f"[red] Analysis failed: {results.get('error')}[/red]")
                raise typer.Exit(1)
            
            if output_format == "json":
                console.print(json.dumps(results, indent=2, default=str))
            elif output_format == "summary":
                display_summary(results)
            else:
                display_rich_results(results)
            
            if save_results:
                filename = orchestrator.save_analysis_results(results)
                console.print(f"\n[green]Results saved to: {filename}[/green]")
                
        except Exception as e:
            progress.update(task, description="Analysis failed!")
            console.print(f"[red] Error: {str(e)}[/red]")
            raise typer.Exit(1)

@app.command()
def quick(
    symbol: str = typer.Argument(..., help="Stock symbol for quick analysis"),
):
    
    validate_setup()
    
    symbol = symbol.upper()
    console.print(f"\n[bold blue]⚡ Quick Analysis for {symbol}[/bold blue]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        task = progress.add_task("Getting quick analysis...", total=None)
        
        try:
            results = orchestrator.get_quick_analysis(symbol)
            progress.update(task, description="Quick analysis complete!")
            
            if results.get("status") == "failed":
                console.print(f"[red] Quick analysis failed: {results.get('error')}[/red]")
                raise typer.Exit(1)
            
            display_quick_results(results)
            
        except Exception as e:
            progress.update(task, description="Quick analysis failed!")
            console.print(f"[red] Error: {str(e)}[/red]")
            raise typer.Exit(1)

@app.command()
def market_overview():
    console.print("\n[bold blue] Market Overview[/bold blue]")
    
    try:
        market_data = financial_data_tool.get_market_overview()
        sector_data = financial_data_tool.get_sector_performance()
        
        indices_table = Table(title="Major Market Indices")
        indices_table.add_column("Index", style="cyan")
        indices_table.add_column("Current", justify="right")
        indices_table.add_column("Change", justify="right")
        indices_table.add_column("Change %", justify="right")
        
        for name, data in market_data.items():
            if "error" not in data:
                current = f"{data['current']:.2f}"
                change = f"{data['change']:+.2f}"
                change_pct = f"{data['change_percent']:+.2f}%"
                
                color = "green" if data['change'] >= 0 else "red"
                indices_table.add_row(
                    name, 
                    current,
                    f"[{color}]{change}[/{color}]",
                    f"[{color}]{change_pct}[/{color}]"
                )
        
        console.print(indices_table)
        
        sector_table = Table(title="Sector Performance")
        sector_table.add_column("Sector", style="cyan")
        sector_table.add_column("ETF", style="dim")
        sector_table.add_column("Price", justify="right")
        sector_table.add_column("Change %", justify="right")
        
        for sector, data in sector_data.items():
            change_pct = data.get("price_change_percent", 0)
            color = "green" if change_pct >= 0 else "red"
            
            sector_table.add_row(
                sector,
                data.get("etf_symbol", ""),
                f"${data.get('current_price', 0):.2f}",
                f"[{color}]{change_pct:+.2f}%[/{color}]"
            )
        
        console.print(sector_table)
        
    except Exception as e:
        console.print(f"[red] Error getting market overview: {str(e)}[/red]")
        raise typer.Exit(1)

@app.command()
def search(
    query: str = typer.Argument(..., help="Company name or stock symbol to search"),
    limit: int = typer.Option(10, help="Maximum number of results")
):
    
    console.print(f"\n[bold blue]Searching for: {query}[/bold blue]")
    
    try:
        results = financial_data_tool.search_stocks(query, limit)
        
        if not results:
            console.print("[yellow]No results found[/yellow]")
            return
        
        search_table = Table(title="Search Results")
        search_table.add_column("Symbol", style="cyan")
        search_table.add_column("Company Name")
        search_table.add_column("Sector", style="dim")
        search_table.add_column("Industry", style="dim")
        
        for result in results:
            search_table.add_row(
                result.get("symbol", ""),
                result.get("name", ""),
                result.get("sector", ""),
                result.get("industry", "")
            )
        
        console.print(search_table)
        
    except Exception as e:
        console.print(f"[red] Error searching: {str(e)}[/red]")
        raise typer.Exit(1)

def display_rich_results(results: Dict):
    exec_summary = results.get("executive_summary", {})
    
    summary_panel = Panel(
        f"""[bold]Analysis Overview[/bold]
Symbols: {', '.join(results.get('symbols', []))}
Period: {results.get('analysis_period', 'Unknown')}
Status: [green]{results.get('status', 'Unknown')}[/green]
Timestamp: {results.get('timestamp', 'Unknown')}

[bold]Investment Strategy[/bold]: {exec_summary.get('investment_strategy', 'Unknown')}
[bold]Market Outlook[/bold]: {exec_summary.get('market_outlook', 'Unknown')}""",
        title="Financial Analysis Results",
        border_style="blue"
    )
    console.print(summary_panel)
    
    recommendations = exec_summary.get("top_recommendations", {})
    
    if recommendations.get("buy"):
        buy_table = Table(title="Top Buy Recommendations")
        buy_table.add_column("Symbol", style="cyan")
        buy_table.add_column("Action", style="green")
        buy_table.add_column("Confidence", justify="right")
        buy_table.add_column("Target Price", justify="right")
        
        for rec in recommendations["buy"]:
            buy_table.add_row(
                rec["symbol"],
                rec["action"],
                f"{rec['confidence']:.1%}",
                f"${rec.get('target_price', 0):.2f}" if rec.get('target_price') else "N/A"
            )
        
        console.print(buy_table)
    
    if recommendations.get("sell"):
        sell_table = Table(title="Top Sell Recommendations")
        sell_table.add_column("Symbol", style="cyan")
        sell_table.add_column("Action", style="red")
        sell_table.add_column("Confidence", justify="right")
        
        for rec in recommendations["sell"]:
            sell_table.add_row(
                rec["symbol"],
                rec["action"],
                f"{rec['confidence']:.1%}"
            )
        
        console.print(sell_table)
    
    portfolio_summary = exec_summary.get("portfolio_summary", {})
    if portfolio_summary:
        portfolio_panel = Panel(
            f"""[bold]Recommended Portfolio Allocation[/bold]
Stock Allocation: {portfolio_summary.get('recommended_stock_allocation', 'N/A')}
Cash Allocation: {portfolio_summary.get('recommended_cash_allocation', 'N/A')}
Number of Positions: {portfolio_summary.get('number_of_positions', 'N/A')}""",
            title="Portfolio Summary",
            border_style="green"
        )
        console.print(portfolio_panel)
    
    risk_highlights = exec_summary.get("risk_highlights", [])
    if risk_highlights:
        risk_text = "\n".join(f"• {highlight}" for highlight in risk_highlights)
        risk_panel = Panel(
            risk_text,
            title="Risk Highlights",
            border_style="yellow"
        )
        console.print(risk_panel)

def display_summary(results: Dict):
    exec_summary = results.get("executive_summary", {})
    
    console.print(f"\n[bold]Financial Analysis Summary[/bold]")
    console.print(f"Symbols: {', '.join(results.get('symbols', []))}")
    console.print(f"Status: {results.get('status', 'Unknown')}")
    console.print(f"Investment Strategy: {exec_summary.get('investment_strategy', 'Unknown')}")
    console.print(f"Market Outlook: {exec_summary.get('market_outlook', 'Unknown')}")
    
    recommendations = exec_summary.get("top_recommendations", {})
    if recommendations.get("buy"):
        console.print(f"\nTop Buy Recommendations:")
        for rec in recommendations["buy"][:3]:
            console.print(f"  • {rec['symbol']}: {rec['action']} (Confidence: {rec['confidence']:.1%})")
    
    if recommendations.get("sell"):
        console.print(f"\nTop Sell Recommendations:")
        for rec in recommendations["sell"][:3]:
            console.print(f"  • {rec['symbol']}: {rec['action']} (Confidence: {rec['confidence']:.1%})")

def display_quick_results(results: Dict):
    symbol = results.get("symbol", "Unknown")
    quick_data = results.get("quick_analysis", {})
    
    quick_panel = Panel(
        f"""[bold]{symbol} Quick Analysis[/bold]

Current Price: ${quick_data.get('current_price', 0):.2f}
Price Change: {quick_data.get('price_change_percent', 0):+.2f}%
Technical Signal: {quick_data.get('technical_signal', 'Unknown')}
Risk Level: {quick_data.get('risk_level', 'Unknown')}
News Sentiment: {quick_data.get('news_sentiment', 'Unknown')}""",
        title="Quick Analysis",
        border_style="cyan"
    )
    console.print(quick_panel)

@app.command()
def config():
    console.print("\n[bold blue]Current Configuration[/bold blue]")
    
    config_table = Table()

    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value")
    
    config_table.add_row("Default LLM Provider", "Google Gemini")
    config_table.add_row("Default Model", settings.default_model)
    config_table.add_row("Fallback Model", settings.fallback_model)
    config_table.add_row("Log Level", settings.log_level)
    config_table.add_row("Max Retries", str(settings.max_retries))
    config_table.add_row("Timeout (seconds)", str(settings.timeout_seconds))
    
    config_table.add_row("Available LLM Provider", "Google Gemini" if settings.google_api_key else "None")
    
    console.print(config_table)

def main():
    app()

if __name__ == "__main__":
    main()
