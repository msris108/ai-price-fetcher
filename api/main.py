import uvicorn
from fastapi import FastAPI, Query, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
import os
import hashlib
import time
import json
from contextlib import asynccontextmanager
from dotenv import load_dotenv


# Load .env file
load_dotenv()

# Import your search module
try:
    from api.search import fetch_prices
except ImportError:
    # Fallback if running directly
    from search import fetch_prices

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory cache
class InMemoryCache:
    def __init__(self, default_ttl: int = 3600):  # 1 hour default TTL
        self.cache = {}
        self.default_ttl = default_ttl
    
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            data, expiry = self.cache[key]
            if time.time() < expiry:
                return data
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        if ttl is None:
            ttl = self.default_ttl
        expiry = time.time() + ttl
        self.cache[key] = (value, expiry)
    
    def delete(self, key: str) -> bool:
        if key in self.cache:
            del self.cache[key]
            return True
        return False
    
    def clear(self) -> None:
        self.cache.clear()
    
    def size(self) -> int:
        # Clean expired items first
        current_time = time.time()
        expired_keys = [k for k, (_, expiry) in self.cache.items() if current_time >= expiry]
        for k in expired_keys:
            del self.cache[k]
        return len(self.cache)

# Global cache instance
cache = InMemoryCache(default_ttl=1800)  # 30 minutes default

# Simple Bearer Token Auth
class SimpleAuth:
    def __init__(self):
        # Load configuration from environment or use defaults
        self.whitelisted_emails = self._load_whitelisted_emails()
        self.secret_key = os.getenv("AUTH_SECRET_KEY", "your-secret-key-change-this")
        self.token_ttl = int(os.getenv("TOKEN_TTL", "86400"))  # 24 hours default
        
    def _load_whitelisted_emails(self) -> List[str]:
        """Load whitelisted emails from environment variable."""
        emails_str = os.getenv("WHITELISTED_EMAILS", "admin@example.com,user@example.com")
        return [email.strip() for email in emails_str.split(",") if email.strip()]
    
    def generate_token(self, email: str) -> str:
        """Generate a simple token for an email."""
        if email not in self.whitelisted_emails:
            raise ValueError(f"Email {email} not whitelisted")
        
        # Create a simple token: hash(email + secret + timestamp)
        timestamp = str(int(time.time()))
        token_data = f"{email}:{timestamp}:{self.secret_key}"
        token_hash = hashlib.sha256(token_data.encode()).hexdigest()
        
        # Format: email:timestamp:hash
        return f"{email}:{timestamp}:{token_hash}"
    
    def verify_token(self, token: str) -> Optional[str]:
        """Verify token and return email if valid."""
        try:
            parts = token.split(":")
            if len(parts) != 3:
                return None
            
            email, timestamp, token_hash = parts
            
            # Check if email is whitelisted
            if email not in self.whitelisted_emails:
                return None
            
            # Check if token is expired
            token_time = int(timestamp)
            if time.time() - token_time > self.token_ttl:
                return None
            
            # Verify hash
            expected_token_data = f"{email}:{timestamp}:{self.secret_key}"
            expected_hash = hashlib.sha256(expected_token_data.encode()).hexdigest()
            
            if token_hash == expected_hash:
                return email
            
        except Exception as e:
            logger.error(f"Token verification error: {e}")
        
        return None

# Global auth instance
auth = SimpleAuth()
security = HTTPBearer()

# Auth dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify bearer token and return user email."""
    token = credentials.credentials
    email = auth.verify_token(token)
    
    if not email:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return email

# Cache key generator
def generate_cache_key(product_name: str, country: str) -> str:
    """Generate cache key for price search."""
    return f"prices:{hashlib.md5(f'{product_name}:{country}'.encode()).hexdigest()}"

# Pydantic models for API documentation
class PriceInfo(BaseModel):
    price: Optional[str | float] = Field(None, description="Extracted price as string")
    currency: Optional[str] = Field(None, description="Currency code (USD, INR, EUR, etc.)")
    confidence: str = Field("low", description="Confidence level: high, medium, or low")

class PriceResult(BaseModel):
    link: str = Field(..., description="URL of the product page")
    productName: str = Field(..., description="Name/title of the product from the webpage")
    matchScore: float = Field(..., description="Relevance score (0-100) based on search query match")
    price: Optional[PriceInfo] = Field(None, description="Price information if found")

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")

class AuthTokenRequest(BaseModel):
    email: str = Field(..., description="Your whitelisted email address")

class AuthTokenResponse(BaseModel):
    token: str = Field(..., description="Bearer token for API access")
    email: str = Field(..., description="Email address")
    expires_in: int = Field(..., description="Token expiration time in seconds")

class CacheStatsResponse(BaseModel):
    size: int = Field(..., description="Number of cached items")
    ttl: int = Field(..., description="Default TTL in seconds")

# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Price Extractor API...")
    
    # Check if required dependencies are available
    try:
        # Test if we can import required modules
        import requests
        import ollama
        logger.info("✓ All dependencies available")
    except ImportError as e:
        logger.error(f"✗ Missing dependency: {e}")
    
    # Check environment variables
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        logger.info("✓ OpenAI API key found - will alternate between OpenAI and Ollama")
    else:
        logger.info("✓ No OpenAI API key - will use Ollama only")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Price Extractor API...")

# Create FastAPI app with custom documentation
app = FastAPI(
    title="Price Extractor API",
    description="""
    **Price Extractor API** - Extract product prices from e-commerce websites
    
    This API searches for products online and extracts their prices using:
    - **LLM-based extraction** (Ollama by default, OpenAI if API key provided)
    - **Regex-based fallback** for when LLM extraction fails
    - **Smart alternating** between providers for load balancing
    
    ## Features
    - 🔍 **Smart Search**: Uses DuckDuckGo to find relevant product pages
    - 🤖 **AI-Powered**: Leverages LLMs for accurate price extraction
    - 🔄 **Fallback System**: Multiple extraction methods for reliability
    - 📊 **Relevance Scoring**: Results sorted by match relevance
    - 🌍 **Multi-Country**: Search in different countries/regions
    - 🔒 **Bearer Token Auth**: Simple token-based authentication
    - ⚡ **In-Memory Caching**: Fast response times with automatic cache expiration
    
    ## Authentication
    1. Get your bearer token from `/auth/token` with your whitelisted email
    2. Include it in requests: `Authorization: Bearer <your-token>`
    
    ## Setup
    - **Ollama**: Install locally for free LLM inference
    - **OpenAI** (optional): Set `OPENAI_API_KEY` environment variable
    - **Auth**: Set `WHITELISTED_EMAILS` (comma-separated) and `AUTH_SECRET_KEY`
    
    ## Rate Limits
    - Be mindful of website rate limits
    - DuckDuckGo search has built-in delays
    """,
    version="1.0.0",
    contact={
        "name": "Price Extractor API",
        "email": "support@example.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Price Extractor API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "fetch_prices": "/fetch-prices"
        }
    }

@app.post(
    "/auth/token",
    response_model=AuthTokenResponse,
    responses={
        200: {"description": "Token generated successfully"},
        400: {"model": ErrorResponse, "description": "Invalid email or not whitelisted"},
        500: {"model": ErrorResponse, "description": "Token generation failed"},
    },
    summary="Generate Bearer Token",
    description="Generate a bearer token for API access using your whitelisted email",
    tags=["Authentication"]
)
async def generate_token(request: AuthTokenRequest):
    """Generate a bearer token for API access."""
    try:
        token = auth.generate_token(request.email)
        return AuthTokenResponse(
            token=token,
            email=request.email,
            expires_in=auth.token_ttl
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Token generation error: {e}")
        raise HTTPException(status_code=500, detail="Token generation failed")

@app.get(
    "/cache/stats",
    response_model=CacheStatsResponse,
    summary="Get Cache Statistics",
    description="Get current cache statistics including size and TTL",
    tags=["Cache Management"]
)
async def get_cache_stats(current_user: str = Depends(get_current_user)):
    """Get cache statistics."""
    return CacheStatsResponse(
        size=cache.size(),
        ttl=cache.default_ttl
    )

@app.delete(
    "/cache/clear",
    summary="Clear Cache",
    description="Clear all cached price data",
    tags=["Cache Management"]
)
async def clear_cache(current_user: str = Depends(get_current_user)):
    """Clear all cached data."""
    cache.clear()
    return {"message": "Cache cleared successfully", "cleared_by": current_user}
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Test if Ollama is accessible
        import ollama
        ollama.list()  # This will fail if Ollama is not running
        ollama_status = "✓ Running"
    except Exception as e:
        ollama_status = f"✗ Error: {str(e)}"
    
    # Check OpenAI availability
    openai_status = "✓ Available" if os.getenv("OPENAI_API_KEY") else "✗ No API key"
    
    return {
        "status": "healthy",
        "services": {
            "ollama": ollama_status,
            "openai": openai_status,
            "cache": f"✓ {cache.size()} items cached",
            "auth": f"✓ {len(auth.whitelisted_emails)} whitelisted emails: {auth.whitelisted_emails}"
        },
        "timestamp": "2025-07-08T00:00:00Z"  # You can use datetime.now() here
    }

@app.get(
    "/fetch-prices",
    response_model=List[PriceResult],
    responses={
        200: {
            "description": "Successfully extracted prices",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "link": "https://example.com/product",
                            "productName": "iPhone 15 Pro 128GB",
                            "matchScore": 95,
                            "price": {
                                "price": "₹1,34,900",
                                "currency": "INR",
                                "confidence": "high"
                            }
                        }
                    ]
                }
            }
        },
        400: {"model": ErrorResponse, "description": "Invalid request parameters"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Extract product prices from e-commerce websites",
    description="""
    Search for a product and extract prices from multiple e-commerce websites.
    
    **Parameters:**
    - `product_name`: The name of the product to search for
    - `country`: Country/region to search in (affects search results and currency)
    
    **Returns:**
    - List of price results sorted by relevance score
    - Each result includes price, currency, confidence level, and source link
    
    **Process:**
    1. Search DuckDuckGo for product listings
    2. Extract prices using LLM (Ollama/OpenAI)
    3. Fallback to regex extraction if LLM fails
    4. Return results sorted by match relevance
    """,
    tags=["Price Extraction"]
)
async def get_prices(
    product_name: str = Query(
        ...,
        description="Name of the product to search for",
        example="iPhone 15",
        min_length=1,
        max_length=200
    ),
    country: str = Query(
        "India",
        description="Country or region to search in",
        example="India",
        max_length=100
    ),
    current_user: str = Depends(get_current_user)
):
    """
    Extract prices for a product from multiple e-commerce websites.
    """
    try:
        # Validate input
        if not product_name or not product_name.strip():
            raise HTTPException(
                status_code=400,
                detail="Product name cannot be empty"
            )
        
        # Clean inputs
        product_name = product_name.strip()
        country = country.strip()
        
        # Check cache first
        cache_key = generate_cache_key(product_name, country)
        cached_result = cache.get(cache_key)
        
        if cached_result:
            logger.info(f"Cache hit for '{product_name}' in {country} (user: {current_user})")
            return cached_result
        
        logger.info(f"Fetching prices for '{product_name}' in {country} (user: {current_user})")

        # Call the price extraction function
        results = fetch_prices(product_name, country)
        
        # Cache the results (30 minutes TTL)
        cache.set(cache_key, results, ttl=1800)
        
        # Log results
        logger.info(f"Found {len(results)} results for '{product_name}' (cached for reuse)")
        
        if not results:
            logger.warning(f"No results found for '{product_name}' in {country}")
            return []
        
        return results
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error fetching prices for '{product_name}': {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler for unexpected errors."""
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred"
        }
    )

if __name__ == "__main__":
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("RELOAD", "true").lower() == "true"
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "app.main:app",  # Use "main:app" if running this file directly
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )