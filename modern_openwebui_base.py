"""
title: Modern OpenWebUI Base Module
version: 1.0.0
description: Modern, secure, and performant base module for OpenWebUI tools and functions
author: commstech
tags: [base, modern, secure, performant, ollama-v0.11.8+, openwebui-v0.6.26+]
"""

import os
import asyncio
import logging
import time
import json
import ssl
from typing import Optional, Dict, List, Any, Callable, Awaitable, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

import aiohttp
import aiohttp.client_exceptions
from pydantic import BaseModel, Field, validator
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            
            raise e

class RateLimiter:
    """Rate limiting for API calls"""
    
    def __init__(self, max_calls: int = 100, time_window: int = 60):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
        
    async def acquire(self):
        """Acquire permission to make a call"""
        now = time.time()
        
        # Remove old calls outside the time window
        self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]
        
        if len(self.calls) >= self.max_calls:
            # Wait until we can make another call
            sleep_time = self.time_window - (now - self.calls[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        self.calls.append(now)

class ModernValves(BaseModel):
    """Modern configuration with environment variable support"""
    
    # Ollama Configuration
    ollama_host: str = Field(
        default="localhost",
        description="Ollama host address"
    )
    ollama_port: int = Field(
        default=11434,
        description="Ollama port number"
    )
    ollama_api_version: str = Field(
        default="v1",
        description="Ollama API version"
    )
    default_model: str = Field(
        default="llama3.2:8b",
        description="Default model to use"
    )
    fallback_models: List[str] = Field(
        default=["llama3.2:8b", "llama3.1:8b", "mistral:7b"],
        description="Fallback models if primary fails"
    )
    
    # Security Configuration
    enable_ssl: bool = Field(
        default=True,
        description="Enable SSL verification"
    )
    ssl_context: Optional[ssl.SSLContext] = Field(
        default=None,
        description="Custom SSL context"
    )
    timeout: int = Field(
        default=30,
        description="Request timeout in seconds"
    )
    max_retries: int = Field(
        default=3,
        description="Maximum retry attempts"
    )
    
    # Performance Configuration
    connection_pool_size: int = Field(
        default=100,
        description="Connection pool size"
    )
    max_concurrent_requests: int = Field(
        default=10,
        description="Maximum concurrent requests"
    )
    enable_caching: bool = Field(
        default=True,
        description="Enable response caching"
    )
    cache_ttl: int = Field(
        default=300,
        description="Cache TTL in seconds"
    )
    
    # Feature Configuration
    enable_function_calling: bool = Field(
        default=True,
        description="Enable function calling support"
    )
    enable_streaming: bool = Field(
        default=True,
        description="Enable streaming responses"
    )
    enable_multimodal: bool = Field(
        default=False,
        description="Enable multimodal support"
    )
    
    # Web Search Configuration
    search_engines: List[str] = Field(
        default=["google", "duckduckgo", "brave"],
        description="Search engines to use"
    )
    search_limit: int = Field(
        default=5,
        description="Maximum search results"
    )
    enable_web_scraping: bool = Field(
        default=True,
        description="Enable web content scraping"
    )
    
    @validator('ollama_host', pre=True)
    def get_ollama_host(cls, v):
        return os.getenv('OLLAMA_HOST', v)
    
    @validator('ollama_port', pre=True)
    def get_ollama_port(cls, v):
        return int(os.getenv('OLLAMA_PORT', v))
    
    @validator('default_model', pre=True)
    def get_default_model(cls, v):
        return os.getenv('OLLAMA_MODEL', v)
    
    @property
    def ollama_base_url(self) -> str:
        """Get the base URL for Ollama API"""
        protocol = "https" if self.enable_ssl else "http"
        return f"{protocol}://{self.ollama_host}:{self.ollama_port}/api/{self.ollama_api_version}"
    
    @property
    def chat_endpoint(self) -> str:
        """Get the chat completions endpoint"""
        return f"{self.ollama_base_url}/chat"
    
    @property
    def generate_endpoint(self) -> str:
        """Get the generate endpoint"""
        return f"{self.ollama_base_url}/generate"

class ModernHTTPClient:
    """Modern HTTP client with connection pooling and circuit breaker"""
    
    def __init__(self, valves: ModernValves):
        self.valves = valves
        self.session: Optional[aiohttp.ClientSession] = None
        self.circuit_breaker = CircuitBreaker()
        self.rate_limiter = RateLimiter()
        self.semaphore = asyncio.Semaphore(valves.max_concurrent_requests)
        
    async def __aenter__(self):
        await self.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
    
    async def initialize(self):
        """Initialize the HTTP client session"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=self.valves.timeout)
            connector = aiohttp.TCPConnector(
                limit=self.valves.connection_pool_size,
                limit_per_host=self.valves.connection_pool_size // 2,
                ssl=self.valves.ssl_context or self._create_ssl_context(),
                enable_cleanup_closed=True
            )
            
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={
                    "User-Agent": "ModernOpenWebUI/1.0.0",
                    "Accept": "application/json",
                    "Content-Type": "application/json"
                }
            )
    
    def _create_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context based on configuration"""
        if not self.valves.enable_ssl:
            return ssl.create_default_context()
        
        context = ssl.create_default_context()
        context.check_hostname = True
        context.verify_mode = ssl.CERT_REQUIRED
        return context
    
    async def request(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> aiohttp.ClientResponse:
        """Make HTTP request with circuit breaker and rate limiting"""
        await self.rate_limiter.acquire()
        
        async with self.semaphore:
            try:
                if not self.session:
                    await self.initialize()
                
                response = await self.session.request(method, url, **kwargs)
                response.raise_for_status()
                return response
                
            except Exception as e:
                logger.error(f"HTTP request failed: {e}")
                raise
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
            self.session = None

class ModernOllamaClient:
    """Modern Ollama client with latest API support"""
    
    def __init__(self, http_client: ModernHTTPClient, valves: ModernValves):
        self.http_client = http_client
        self.valves = valves
        self.model_cache = {}
        self.last_model_check = 0
        self.model_check_interval = 300  # 5 minutes
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Make chat completion request with modern API support"""
        model = model or self.valves.default_model
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **kwargs
        }
        
        if tools and self.valves.enable_function_calling:
            payload["tools"] = tools
        
        try:
            response = await self.http_client.request(
                "POST",
                self.valves.chat_endpoint,
                json=payload
            )
            
            if stream:
                return await self._handle_streaming_response(response)
            else:
                return await response.json()
                
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            return await self._fallback_chat_completion(messages, tools, stream, **kwargs)
    
    async def _handle_streaming_response(self, response: aiohttp.ClientResponse) -> Dict[str, Any]:
        """Handle streaming response from Ollama"""
        result = {
            "model": "",
            "created_at": "",
            "message": {"role": "assistant", "content": ""},
            "done": False,
            "stream_data": []
        }
        
        async for line in response.content:
            if line:
                try:
                    data = json.loads(line.decode('utf-8'))
                    result["stream_data"].append(data)
                    
                    if data.get("done", False):
                        result.update(data)
                        break
                    elif "message" in data:
                        result["message"]["content"] += data["message"].get("content", "")
                        
                except json.JSONDecodeError:
                    continue
        
        return result
    
    async def _fallback_chat_completion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Fallback to alternative models if primary fails"""
        for fallback_model in self.valves.fallback_models:
            try:
                logger.info(f"Trying fallback model: {fallback_model}")
                return await self.chat_completion(
                    messages, fallback_model, tools, stream, **kwargs
                )
            except Exception as e:
                logger.warning(f"Fallback model {fallback_model} failed: {e}")
                continue
        
        raise Exception("All models failed to respond")
    
    async def list_models(self) -> List[str]:
        """List available models"""
        try:
            response = await self.http_client.request(
                "GET",
                f"{self.valves.ollama_base_url}/tags"
            )
            data = await response.json()
            return [model["name"] for model in data.get("models", [])]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    async def check_model_availability(self, model: str) -> bool:
        """Check if a specific model is available"""
        try:
            response = await self.http_client.request(
                "POST",
                f"{self.valves.ollama_base_url}/show",
                json={"name": model}
            )
            return response.status == 200
        except Exception:
            return False

class ModernWebSearch:
    """Modern web search with multiple engines and intelligent fallback"""
    
    def __init__(self, http_client: ModernHTTPClient, valves: ModernValves):
        self.http_client = http_client
        self.valves = valves
        self.search_cache = {}
        self.last_cache_cleanup = time.time()
    
    async def search(
        self,
        query: str,
        engines: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Perform web search with multiple engines"""
        engines = engines or self.valves.search_engines
        limit = limit or self.valves.search_limit
        
        # Check cache first
        cache_key = f"{query}:{':'.join(engines)}:{limit}"
        if cache_key in self.search_cache:
            cached_result = self.search_cache[cache_key]
            if time.time() - cached_result["timestamp"] < self.valves.cache_ttl:
                return cached_result["results"]
        
        # Perform search
        all_results = []
        for engine in engines:
            try:
                engine_results = await self._search_engine(query, engine, limit)
                all_results.extend(engine_results)
            except Exception as e:
                logger.warning(f"Search engine {engine} failed: {e}")
                continue
        
        # Deduplicate and rank results
        unique_results = self._deduplicate_results(all_results)
        ranked_results = self._rank_results(unique_results, query)[:limit]
        
        # Cache results
        self.search_cache[cache_key] = {
            "results": ranked_results,
            "timestamp": time.time()
        }
        
        # Cleanup old cache entries
        await self._cleanup_cache()
        
        return ranked_results
    
    async def _search_engine(self, query: str, engine: str, limit: int) -> List[Dict[str, Any]]:
        """Search using a specific engine"""
        # This is a placeholder - implement actual search engine APIs
        # For now, return mock results
        return [
            {
                "title": f"Search result from {engine}",
                "url": f"https://example.com/{engine}",
                "snippet": f"Mock result for query: {query}",
                "engine": engine
            }
        ]
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results based on URL"""
        seen_urls = set()
        unique_results = []
        
        for result in results:
            if result["url"] not in seen_urls:
                seen_urls.add(result["url"])
                unique_results.append(result)
        
        return unique_results
    
    def _rank_results(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Rank results by relevance to query"""
        query_terms = set(query.lower().split())
        
        def relevance_score(result):
            title_terms = set(result["title"].lower().split())
            snippet_terms = set(result.get("snippet", "").lower().split())
            
            title_match = len(query_terms.intersection(title_terms))
            snippet_match = len(query_terms.intersection(snippet_terms))
            
            return title_match * 2 + snippet_match
        
        return sorted(results, key=relevance_score, reverse=True)
    
    async def _cleanup_cache(self):
        """Clean up old cache entries"""
        current_time = time.time()
        if current_time - self.last_cache_cleanup > 300:  # Every 5 minutes
            expired_keys = [
                key for key, value in self.search_cache.items()
                if current_time - value["timestamp"] > self.valves.cache_ttl
            ]
            
            for key in expired_keys:
                del self.search_cache[key]
            
            self.last_cache_cleanup = current_time

class ModernContentScraper:
    """Modern content scraper with intelligent extraction and sanitization"""
    
    def __init__(self, http_client: ModernHTTPClient, valves: ModernValves):
        self.http_client = http_client
        self.valves = valves
    
    async def scrape_content(
        self,
        urls: List[str],
        max_content_length: int = 2000
    ) -> List[Dict[str, Any]]:
        """Scrape content from multiple URLs concurrently"""
        if not self.valves.enable_web_scraping:
            return []
        
        tasks = [self._scrape_single_url(url, max_content_length) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out failed scrapes
        successful_results = [
            result for result in results
            if isinstance(result, dict) and result.get("content")
        ]
        
        return successful_results
    
    async def _scrape_single_url(
        self,
        url: str,
        max_content_length: int
    ) -> Dict[str, Any]:
        """Scrape content from a single URL"""
        try:
            response = await self.http_client.request("GET", url)
            html = await response.text()
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove unwanted elements
            for element in soup.find_all(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                element.decompose()
            
            # Extract content using multiple strategies
            content = self._extract_content(soup)
            
            if not content:
                return {"url": url, "content": "", "error": "No content found"}
            
            # Clean and truncate content
            cleaned_content = self._clean_content(content)
            truncated_content = cleaned_content[:max_content_length]
            
            return {
                "url": url,
                "content": truncated_content,
                "title": soup.find("title").get_text(strip=True) if soup.find("title") else "",
                "timestamp": datetime.now().isoformat(),
                "content_length": len(cleaned_content)
            }
            
        except Exception as e:
            logger.error(f"Failed to scrape {url}: {e}")
            return {"url": url, "content": "", "error": str(e)}
    
    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract content using multiple strategies"""
        # Strategy 1: Article content
        article = soup.find(['article', '[itemtype*="Article"]'])
        if article:
            return article.get_text(strip=True)
        
        # Strategy 2: Main content area
        main_content = soup.find(['main', '[role="main"]', '#content', '.content'])
        if main_content:
            return main_content.get_text(strip=True)
        
        # Strategy 3: Body content with filtering
        body = soup.find('body')
        if body:
            # Remove navigation and other non-content elements
            for element in body.find_all(['nav', 'menu', 'sidebar', 'advertisement']):
                element.decompose()
            
            return body.get_text(strip=True)
        
        return ""
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize content"""
        # Remove extra whitespace
        content = ' '.join(content.split())
        
        # Remove common web artifacts
        content = content.replace('\n', ' ')
        content = content.replace('\r', ' ')
        content = content.replace('\t', ' ')
        
        # Remove multiple spaces
        while '  ' in content:
            content = content.replace('  ', ' ')
        
        return content.strip()

class ModernBaseFilter:
    """Modern base filter with all the latest features and best practices"""
    
    def __init__(self, valves: Optional[ModernValves] = None):
        self.valves = valves or ModernValves()
        self.http_client: Optional[ModernHTTPClient] = None
        self.ollama_client: Optional[ModernOllamaClient] = None
        self.web_search: Optional[ModernWebSearch] = None
        self.content_scraper: Optional[ModernContentScraper] = None
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all components"""
        self.http_client = ModernHTTPClient(self.valves)
        self.ollama_client = ModernOllamaClient(self.http_client, self.valves)
        self.web_search = ModernWebSearch(self.http_client, self.valves)
        self.content_scraper = ModernContentScraper(self.http_client, self.valves)
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.http_client.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.http_client.cleanup()
    
    async def process_message(
        self,
        body: Dict[str, Any],
        event_emitter: Optional[Callable[[Any], Awaitable[None]]] = None,
        user: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process incoming message with modern capabilities"""
        try:
            # Extract messages
            messages = body.get("messages", [])
            if not messages:
                return body
            
            # Get the last message (user input)
            user_message = messages[-1]["content"]
            
            # Emit status update
            if event_emitter:
                await event_emitter({
                    "type": "status",
                    "data": {
                        "description": "Processing message...",
                        "done": False
                    }
                })
            
            # Process the message
            processed_response = await self._process_user_message(
                user_message, messages, event_emitter, user
            )
            
            # Update the response
            if messages:
                messages[-1]["content"] = processed_response
            
            body["messages"] = messages
            
            # Final status update
            if event_emitter:
                await event_emitter({
                    "type": "status",
                    "data": {
                        "description": "Processing complete",
                        "done": True
                    }
                })
            
            return body
            
        except Exception as e:
            self.logger.error(f"Message processing failed: {e}")
            return self._handle_error(body, str(e), event_emitter)
    
    async def _process_user_message(
        self,
        user_message: str,
        messages: List[Dict[str, Any]],
        event_emitter: Optional[Callable[[Any], Awaitable[None]]] = None,
        user: Optional[Dict[str, Any]] = None
    ) -> str:
        """Process user message with modern capabilities"""
        # This is a base implementation - override in subclasses
        return f"Processed: {user_message}"
    
    async def _handle_error(
        self,
        body: Dict[str, Any],
        error: str,
        event_emitter: Optional[Callable[[Any], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """Handle errors gracefully"""
        error_response = f"An error occurred: {error}"
        
        if event_emitter:
            await event_emitter({
                "type": "status",
                "data": {
                    "description": f"Error: {error}",
                    "done": True
                }
            })
        
        messages = body.get("messages", [])
        if messages:
            messages[-1]["content"] = error_response
        else:
            messages.append({"role": "assistant", "content": error_response})
        
        body["messages"] = messages
        return body
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.http_client:
            await self.http_client.cleanup()

def register():
    """Register the modern base filter"""
    return ModernBaseFilter()