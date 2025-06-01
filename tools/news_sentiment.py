import feedparser
import requests
from textblob import TextBlob
from typing import Dict, List, Any
import logging
from datetime import datetime, timedelta
from core.config import settings

logger = logging.getLogger(__name__)

class NewsSentimentTool:
    
    def __init__(self):
        self.finnhub_key = settings.finnhub_api_key
        self.news_sources = [
        "https://feeds.finance.yahoo.com/rss/2.0/headline",           
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",      
        "https://www.marketwatch.com/rss/topstories",
    ]
    
    def get_stock_news(self, symbol: str, days_back: int = 7) -> List[Dict[str, Any]]:
        news_articles = []
        
        if self.finnhub_key:
            try:
                news_articles.extend(self.get_finnhub_news(symbol, days_back))
            except Exception as e:
                logger.error(f"Error fetching Finnhub news: {str(e)}")
        
        try:
            news_articles.extend(self.get_rss_news(symbol, days_back))
        except Exception as e:
            logger.error(f"Error fetching RSS news: {str(e)}")
        
        for article in news_articles:
            article['sentiment'] = self.analyze_sentiment(article.get('summary', ''))
        
        news_articles.sort(key=lambda x: x.get('published', ''), reverse=True)
        
        return news_articles[:20]
    
    def get_finnhub_news(self, symbol: str, days_back: int) -> List[Dict[str, Any]]:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        url = "https://finnhub.io/api/v1/company-news"
        params = {
            "symbol": symbol,
            "from": start_date.strftime("%Y-%m-%d"),
            "to": end_date.strftime("%Y-%m-%d"),
            "token": self.finnhub_key
        }
        
        response = requests.get(url, params=params, timeout=settings.timeout_seconds)
        response.raise_for_status()
        
        articles = []
        for item in response.json():
            articles.append({
                "title": item.get("headline", ""),
                "summary": item.get("summary", ""),
                "url": item.get("url", ""),
                "published": datetime.fromtimestamp(item.get("datetime", 0)).isoformat(),
                "source": item.get("source", "Finnhub"),
                "image": item.get("image", "")
            })
        
        return articles
    
    def get_rss_news(self, symbol: str, days_back: int) -> List[Dict[str, Any]]:
        articles = []
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        for feed_url in self.news_sources:
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries:
                    title = entry.get('title', '').lower()
                    summary = entry.get('summary', '').lower()
                    
                    if (symbol.lower() in title or 
                        symbol.lower() in summary or
                        self.company_name(symbol, title + " " + summary)):
                        
                        pub_date = None
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            pub_date = datetime(*entry.published_parsed[:6])
                        
                        if pub_date and pub_date < cutoff_date:
                            continue
                        
                        articles.append({
                            "title": entry.get('title', ''),
                            "summary": entry.get('summary', ''),
                            "url": entry.get('link', ''),
                            "published": pub_date.isoformat() if pub_date else '',
                            "source": feed.feed.get('title', 'RSS Feed')
                        })
                        
            except Exception as e:
                logger.error(f"Error parsing RSS feed {feed_url}: {str(e)}")
        
        return articles
    
    def company_name(self, symbol: str, text: str) -> bool:
        common_mappings = {
        "AAPL": ["apple", "iphone", "ipad", "mac", "ios", "app store", "tim cook"],
        "GOOGL": ["google", "alphabet", "youtube", "android", "search engine", "chrome", "google cloud"],
        "MSFT": ["microsoft", "windows", "azure", "office", "xbox", "surface", "linkedin"],
        "AMZN": ["amazon", "aws", "prime", "kindle", "alexa", "whole foods", "amazon web services"],
        "TSLA": ["tesla", "elon musk", "electric vehicle", "model s", "model 3", "solar city", "gigafactory"],
        "META": ["meta", "facebook", "instagram", "whatsapp", "oculus", "social media", "mark zuckerberg"],
        "NVDA": ["nvidia", "gpu", "ai chip", "graphics card", "cuda", "tensorrt", "rtx"],
        "NFLX": ["netflix", "streaming", "originals", "series", "movies"],
        "DIS": ["disney", "pixar", "marvel", "star wars", "disney+", "abc", "espn"],
        "PYPL": ["paypal", "payments", "venmo", "digital wallet", "e-commerce"],
        "INTC": ["intel", "semiconductor", "chip", "processor", "core i7", "x86"],
        "CSCO": ["cisco", "networking", "router", "switch", "webex", "security"],
        "IBM": ["ibm", "watson", "cloud", "ai", "mainframe"],
        "ORCL": ["oracle", "database", "cloud", "software", "java"],
        "ADBE": ["adobe", "photoshop", "acrobat", "creative cloud", "illustrator"],
        "CRM": ["salesforce", "crm", "cloud", "sales", "service cloud"],
        "SHOP": ["shopify", "e-commerce", "storefront", "payments"],
        "UBER": ["uber", "ride-sharing", "food delivery", "uber eats"],
        "SQ": ["square", "payments", "cash app", "block"],
        "BA": ["boeing", "aerospace", "aircraft", "defense"],
        "XOM": ["exxon", "oil", "energy", "petroleum", "gas"],
        "CVX": ["chevron", "oil", "energy", "gas", "refinery"],
    }

        if symbol.upper() in common_mappings:
            return any(term in text.lower() for term in common_mappings[symbol.upper()])
        
        return False
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        if not text:
            return {"polarity": 0, "subjectivity": 0, "label": "neutral"}
        
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        if polarity > 0.1:
            label = "positive"
        elif polarity < -0.1:
            label = "negative"
        else:
            label = "neutral"
        
        return {
            "polarity": polarity,
            "subjectivity": subjectivity,
            "label": label
        }
    
    def get_market_sentiment_summary(self, symbols: List[str]) -> Dict[str, Any]:
        all_sentiments = []
        symbol_sentiments = {}
        
        for symbol in symbols:
            try:
                news = self.get_stock_news(symbol, days_back=3)
                sentiments = [article['sentiment'] for article in news if 'sentiment' in article]
                
                if sentiments:
                    avg_polarity = sum(s['polarity'] for s in sentiments) / len(sentiments)
                    avg_subjectivity = sum(s['subjectivity'] for s in sentiments) / len(sentiments)
                    
                    symbol_sentiments[symbol] = {
                        "average_polarity": avg_polarity,
                        "average_subjectivity": avg_subjectivity,
                        "article_count": len(sentiments),
                        "sentiment_label": "positive" if avg_polarity > 0.1 else "negative" if avg_polarity < -0.1 else "neutral"
                    }
                    
                    all_sentiments.extend(sentiments)
                    
            except Exception as e:
                logger.error(f"Error analyzing sentiment for {symbol}: {str(e)}")
        
        if all_sentiments:
            overall_polarity = sum(s['polarity'] for s in all_sentiments) / len(all_sentiments)
            overall_subjectivity = sum(s['subjectivity'] for s in all_sentiments) / len(all_sentiments)
            overall_label = "positive" if overall_polarity > 0.1 else "negative" if overall_polarity < -0.1 else "neutral"
        else:
            overall_polarity = 0
            overall_subjectivity = 0
            overall_label = "neutral"
        
        return {
            "overall_sentiment": {
                "polarity": overall_polarity,
                "subjectivity": overall_subjectivity,
                "label": overall_label,
                "total_articles": len(all_sentiments)
            },
            "symbol_sentiments": symbol_sentiments
        }

news_sentiment_tool = NewsSentimentTool()
