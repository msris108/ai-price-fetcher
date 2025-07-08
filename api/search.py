import requests
from bs4 import BeautifulSoup
from urllib.parse import quote, urlparse, parse_qs, unquote
from rapidfuzz import fuzz
import re
import json
import ollama
from openai import OpenAI
from typing import Dict, List, Optional, Union
import logging
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load .env file
load_dotenv()

class LLMProvider(Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"

@dataclass
class PriceResult:
    price: Optional[str]
    currency: Optional[str]
    confidence: str

@dataclass
class SearchResult:
    link: str
    product_name: str
    match_score: int
    price: Optional[PriceResult]

class PriceExtractor:
    def __init__(self, openai_api_key: Optional[str] = None, ollama_model: str = "llama3.2"):
        self.openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None
        self.ollama_model = ollama_model
        # Default to Ollama, only use OpenAI if API key is provided
        self.current_provider = LLMProvider.OLLAMA
        
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        # Common price element selectors
        self.price_selectors = [
            'span.a-price-whole',
            'span.a-price-fraction',
            'div.price-display',
            'span.price',
            'span.selling-price',
            'div.product-price',
            'meta[property="product:price:amount"]',
            'meta[itemprop="price"]',
            'span[data-price]',
            'div.BNeaBox.snPzQe > div:nth-child(1)',
            'div._30jeq3._16Jk6d',
            'div.a-offscreen',
            'span.discount-price',
            'p.product-page__price'
        ]
        
        # Price patterns for regex fallback
        self.price_patterns = [
            r'(?:₹|Rs\.?|INR\s?)\s*\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?',
            r'(?:\$|\€)\s*\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?',
            r'\b\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?\b'
        ]

    def _resolve_duckduckgo_url(self, raw_url: str) -> str:
        """Resolve DuckDuckGo redirect URLs to actual URLs."""
        parsed = urlparse(raw_url)
        if parsed.netloc == "duckduckgo.com" and parsed.path.startswith("/l/"):
            qs = parse_qs(parsed.query)
            if "uddg" in qs:
                return unquote(qs["uddg"][0])
        return raw_url

    def _search_duckduckgo(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search DuckDuckGo and return results."""
        url = f"https://html.duckduckgo.com/html/?q={quote(query)}"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            
            results = []
            for a in soup.select(".result__a")[:max_results]:
                results.append({
                    "title": a.get_text(strip=True),
                    "link": self._resolve_duckduckgo_url(a.get("href"))
                })
            return results
            
        except Exception as e:
            logger.error(f"Error searching DuckDuckGo: {e}")
            return []

    def _get_llm_prompt(self, product_name: str, text: str) -> str:
        """Generate prompt for LLM price extraction."""
        return f"""
You are a machine. A user is looking for the price of the product: "{product_name}" from this web page text:

---
{text[:6000]}
---

Your task:
1. Find the price of the product (if mentioned).
2. Respond ONLY in **strict JSON** format. No extra text, markdown, or explanation.
3. If unsure, return "null" for fields and set confidence to "low".

Your response format MUST be exactly:

{{
"price": "...",
"currency": "USD/INR/EUR etc.",
"confidence": "high" | "medium" | "low"
}}
"""

    def _ask_openai_for_price(self, product_name: str, text: str) -> Optional[str]:
        """Ask OpenAI for price extraction."""
        if not self.openai_client:
            return None
            
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{
                    "role": "user",
                    "content": self._get_llm_prompt(product_name, text)
                }],
                max_tokens=150,
                temperature=0
            )
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return None

    def _ask_ollama_for_price(self, product_name: str, text: str) -> Optional[str]:
        """Ask Ollama for price extraction."""
        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{
                    "role": "user",
                    "content": self._get_llm_prompt(product_name, text)
                }],
            )
            return response['message']['content']
            
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            return None

    def _parse_llm_response(self, llm_output: str) -> PriceResult:
        """Parse LLM response into PriceResult."""
        try:
            # Extract first JSON block from response
            match = re.search(r'\{.*?\}', llm_output, re.DOTALL)
            if match:
                json_str = match.group(0)
                data = json.loads(json_str)
                return PriceResult(
                    price=data.get("price"),
                    currency=data.get("currency"),
                    confidence=data.get("confidence", "low")
                )
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
        
        return PriceResult(price=None, currency=None, confidence="low")

    def _extract_price_with_llm(self, url: str, product_name: str) -> Optional[PriceResult]:
        """Extract price using LLM (alternating between Ollama and OpenAI)."""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            # Remove script and style tags
            for tag in soup(['style', 'script']):
                tag.decompose()
            
            text = ' '.join(soup.stripped_strings)
            
            # Try current provider first
            llm_response = None
            if self.current_provider == LLMProvider.OLLAMA:
                llm_response = self._ask_ollama_for_price(product_name, text)
                if not llm_response and self.openai_client:
                    # Fallback to OpenAI only if API key is available
                    llm_response = self._ask_openai_for_price(product_name, text)
            else:
                llm_response = self._ask_openai_for_price(product_name, text)
                if not llm_response:
                    # Fallback to Ollama
                    llm_response = self._ask_ollama_for_price(product_name, text)
            
            # Toggle provider for next call (only if OpenAI is available)
            if self.openai_client:
                self.current_provider = (LLMProvider.OPENAI if self.current_provider == LLMProvider.OLLAMA else LLMProvider.OLLAMA)
            # If no OpenAI client, stay with Ollama
            
            if llm_response:
                return self._parse_llm_response(llm_response)
                
        except Exception as e:
            logger.error(f"Error extracting price with LLM from {url}: {e}")
        
        return None

    def _extract_price_with_regex(self, url: str, product_name: str) -> Optional[str]:
        """Extract price using regex patterns as fallback."""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Strategy 1: Look for specific HTML elements
            found_prices = []
            for selector in self.price_selectors:
                elements = soup.select(selector)
                for el in elements:
                    price_text = el.get('content') if el.name == 'meta' and 'content' in el.attrs else el.get_text(strip=True)
                    
                    if price_text and self._is_valid_price(price_text):
                        found_prices.append(price_text)
            
            if found_prices:
                return sorted(found_prices, key=len)[0]
            
            # Strategy 2: Regex fallback on entire text
            text = soup.get_text(separator=' ', strip=True)
            
            for pattern in self.price_patterns:
                for match in re.finditer(pattern, text):
                    matched_price = match.group(0).strip()
                    
                    # Check for price context
                    context = text[max(0, match.start() - 50):min(len(text), match.end() + 50)]
                    if any(keyword in context.lower() for keyword in ["price", "mrp", "m.r.p.", "sale price", "deal price"]):
                        if self._is_valid_price(matched_price):
                            return matched_price
                    
                    # Validate as reasonable price
                    if self._is_reasonable_price(matched_price):
                        return matched_price
            
        except Exception as e:
            logger.error(f"Error extracting price with regex from {url}: {e}")
        
        return None

    def _is_valid_price(self, price_text: str) -> bool:
        """Check if price text is valid."""
        try:
            cleaned = re.sub(r'[^\d.,]', '', price_text)
            if ',' in cleaned and '.' not in cleaned:
                temp = cleaned.replace('.', '').replace(',', '.')
            else:
                temp = cleaned.replace(',', '')
            
            float_price = float(temp)
            return float_price > 0
        except ValueError:
            return False

    def _is_reasonable_price(self, price_text: str) -> bool:
        """Check if price is within reasonable bounds."""
        try:
            cleaned = re.sub(r'[^\d.,]', '', price_text)
            if ',' in cleaned and '.' not in cleaned:
                temp = cleaned.replace('.', '').replace(',', '.')
            else:
                temp = cleaned.replace(',', '')
            
            float_price = float(temp)
            return 0 < float_price < 10000000  # Reasonable price range
        except ValueError:
            return False

    def fetch_prices(self, product_name: str, country: str = "India") -> List[SearchResult]:
        """Fetch prices for a product from multiple sources."""
        query = f"{product_name} buy online in {country}"
        search_results = self._search_duckduckgo(query)
        
        scored_results = []
        
        for result in search_results:
            url = result["link"]
            title = result["title"]
            
            # Calculate relevance score
            score = fuzz.token_sort_ratio(product_name.lower(), title.lower())
            
            # Try LLM extraction first
            price_result = self._extract_price_with_llm(url, product_name)
            
            # Fallback to regex if LLM fails
            if not price_result or not price_result.price:
                regex_price = self._extract_price_with_regex(url, product_name)
                if regex_price:
                    price_result = PriceResult(
                        price=regex_price,
                        currency="Unknown",
                        confidence="low"
                    )
            
            if price_result and price_result.price:
                scored_results.append(SearchResult(
                    link=url,
                    product_name=title,
                    match_score=score,
                    price=price_result
                ))

        # Define confidence priority
        confidence_order = {
            "high": 0,
            "medium": 1,
            "low": 2,
            "unknown": 3,
            None: 4
        }

        def clean_price(p):
            if isinstance(p, float) or isinstance(p, int): 
                return p
            # Remove currency symbols and commas and convert to float
            if not p:
                return float('inf')
            p = re.sub(r"[^\d.]", "", p)
            return float(p) if p else float('inf')
                
        return sorted(scored_results, key=lambda x: (confidence_order.get(x.price.confidence, 5), clean_price(x.price.price), -x.match_score))

# Global extractor instance
_extractor = None

def get_extractor() -> PriceExtractor:
    """Get or create the global extractor instance."""
    global _extractor
    if _extractor is None:
        # Try to get OpenAI API key from environment
        import os
        openai_key = os.getenv("OPENAI_API_KEY")
        _extractor = PriceExtractor(openai_key)
    return _extractor

def fetch_prices(product_name: str, country: str = "India") -> List[Dict]:
    """
    Fetch prices for a product - compatible with original function signature.
    
    Args:
        product_name: Name of the product to search for
        country: Country to search in (default: India)
    
    Returns:
        List of dictionaries containing price information
    """
    extractor = get_extractor()
    results = extractor.fetch_prices(product_name, country)
    
    # Convert SearchResult objects to dictionaries for API response
    return [
        {
            "link": result.link,
            "productName": result.product_name,
            "matchScore": result.match_score,
            "price": {
                "price": result.price.price,
                "currency": result.price.currency,
                "confidence": result.price.confidence
            } if result.price else None
        }
        for result in results
    ]

# Usage example
if __name__ == "__main__":
    # Test the fetch_prices function
    results = fetch_prices("iPhone 15", "India")
    
    # Print results
    for result in results:
        print(f"Product: {result['productName']}")
        if result['price']:
            print(f"Price: {result['price']['price']} {result['price']['currency']}")
            print(f"Confidence: {result['price']['confidence']}")
        print(f"Match Score: {result['matchScore']}")
        print(f"Link: {result['link']}")
        print("-" * 50)