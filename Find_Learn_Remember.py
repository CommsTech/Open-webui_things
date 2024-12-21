"""
title: Find Learn Remember - Self-Learning Module
author: commstech
version: 0.3.2
"""

from typing import Optional, Dict, List, Any, Callable, Awaitable
import aiohttp
import logging
import time
import json
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup

class Filter:
    class Valves(BaseModel):
        enable_autolearn: bool = Field(
            default=True, 
            description="Enable or disable real-time learning"
        )
        model: str = Field(
            default="luna-tic:base",
            description="Model to use for processing"
        )
        api_url: str = Field(
            default="http://localhost:11434",
            description="API endpoint"
        )
        search_url: str = Field(
            default="https://search.commsnet.org/search",
            description="Search endpoint"
        )

    def __init__(self):
        self.valves = self.Valves()
        self.logger = logging.getLogger("research_filter")
        self.session = None
        self.knowledge_base = []

    async def outlet(
        self,
        body: Dict[str, Any],
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
        __user__: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process outgoing messages"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            messages = body.get("messages", [])
            if not messages:
                return body

            last_message = messages[-1]["content"]
            start_time = time.time()

            # Search and scrape relevant information
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": "Searching for relevant information...",
                        "done": False
                    }
                })

            search_results = await self._search(last_message)
            scraped_content = await self._scrape_pages(search_results)

            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": "Processing information...",
                        "done": False
                    }
                })

            # Process and store the information
            processed_info = await self._process_information(last_message, scraped_content)
            self.knowledge_base.append({
                "query": last_message,
                "info": processed_info,
                "timestamp": time.time()
            })

            duration = time.time() - start_time
            
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": f"Completed in {duration:.2f}s",
                        "done": True
                    }
                })

            messages[-1]["content"] = processed_info
            body["messages"] = messages
            return body

        except Exception as e:
            self.logger.error(f"Outlet error: {str(e)}")
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": f"Error: {str(e)}",
                        "done": True
                    }
                })
            return body

    async def _search(self, query: str) -> List[Dict]:
        """Perform web search"""
        try:
            params = {
                "q": query,
                "format": "json",
                "engines": "google,duckduckgo,brave",
                "limit": 5
            }
            
            async with self.session.get(
                self.valves.search_url,
                params=params,
                timeout=10
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("results", [])[:5]
                return []
        except Exception as e:
            self.logger.error(f"Search error: {str(e)}")
            return []

    async def _scrape_pages(self, search_results: List[Dict]) -> List[Dict]:
        """Scrape content from search results with improved resilience"""
        scraped_data = []
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        }
        
        async def verify_content(content: str, query: str) -> bool:
            """Verify content relevance"""
            # Basic relevance check
            query_terms = set(query.lower().split())
            content_terms = set(content.lower().split())
            return len(query_terms.intersection(content_terms)) > 0

        for result in search_results:
            try:
                async with self.session.get(
                    result["url"], 
                    timeout=15,
                    headers=headers,
                    ssl=False,  # Handle SSL issues
                    compress=True  # Enable compression including Brotli
                ) as response:
                    if response.status == 200:
                        try:
                            html = await response.text()
                            soup = BeautifulSoup(html, 'html.parser')
                            
                            # Remove unwanted elements
                            for element in soup.find_all(['script', 'style', 'nav', 'footer', 'header']):
                                element.decompose()
                            
                            # Try multiple content extraction methods
                            content = ""
                            
                            # Method 1: Schema.org Article content
                            article = soup.find(['article', '[itemtype*="Article"]'])
                            if article:
                                content = article.get_text(strip=True)
                            
                            # Method 2: Main content area
                            if not content:
                                main_content = soup.find(['main', '[role="main"]', '#content', '.content'])
                                if main_content:
                                    content = main_content.get_text(strip=True)
                            
                            # Method 3: Fallback to paragraphs with filtering
                            if not content:
                                paragraphs = soup.find_all('p')
                                filtered_paragraphs = []
                                for p in paragraphs:
                                    text = p.get_text(strip=True)
                                    # Filter out short or irrelevant paragraphs
                                    if len(text) > 50 and not any(skip in text.lower() for skip in ['cookie', 'privacy policy', 'terms of service']):
                                        filtered_paragraphs.append(text)
                                content = ' '.join(filtered_paragraphs)
                            
                            # Clean up the content
                            content = ' '.join(content.split())  # Remove extra whitespace
                            
                            # Verify content relevance
                            if content and await verify_content(content, result.get("title", "")):
                                scraped_data.append({
                                    "title": result["title"],
                                    "url": result["url"],
                                    "content": content[:1000],  # Increased length for better context
                                    "date_scraped": time.strftime("%Y-%m-%d"),
                                    "timestamp": int(time.time())
                                })
                        except Exception as e:
                            self.logger.error(f"Content parsing error for {result['url']}: {str(e)}")
                            continue
                            
            except aiohttp.ClientError as e:
                self.logger.error(f"Network error for {result['url']}: {str(e)}")
                continue
            except Exception as e:
                self.logger.error(f"Unexpected error for {result['url']}: {str(e)}")
                continue
                
        return scraped_data

    async def _process_information(self, query: str, scraped_data: List[Dict]) -> str:
        """Process scraped information using LLM with current date context"""
        try:
            current_date = time.strftime("%B %d, %Y")
            
            # Sort data by timestamp
            scraped_data.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
            
            context = f"Current date: {current_date}\n\n"
            context += "Recent information:\n"
            for data in scraped_data:
                context += f"\nSource ({data['date_scraped']}): {data['title']}\n{data['content']}\n"
            
            system_prompt = """You are a helpful AI assistant with access to current information.
            Critical instructions:
            1. The current date is VERY important - always mention it
            2. If information is from an older date, explicitly acknowledge this
            3. If you can't find recent information, clearly state this
            4. Distinguish between historical facts and current developments
            5. Be transparent about information gaps
            6. If information seems outdated, recommend checking official sources
            
            Format your response to:
            1. Start with the current date
            2. Provide the most recent information first
            3. Clearly indicate information dates
            4. Note any gaps or uncertainties
            5. Suggest follow-up sources if needed"""

            url = f"{self.valves.api_url}/v1/chat/completions"
            payload = {
                "model": self.valves.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
                ]
            }

            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    answer = data["choices"][0]["message"]["content"]
                    
                    # Add sources with dates
                    sources = "\n\n<details>\n<summary>Sources and Dates</summary>\n"
                    for data in scraped_data:
                        sources += f"- [{data['title']}]({data['url']}) (Accessed: {data['date_scraped']})\n"
                    sources += "</details>"
                    
                    # Add current date footer
                    footer = f"\n\n<sub>Information as of {current_date}</sub>"
                    
                    return f"{answer}\n{sources}{footer}"
                return "Error processing information"

        except Exception as e:
            self.logger.error(f"Processing error: {str(e)}")
            return f"Error processing information: {str(e)}"

    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
            self.session = None

def register():
    """Register the filter"""
    return Filter()
