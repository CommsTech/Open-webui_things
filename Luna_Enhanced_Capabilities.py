"""
title: Luna Enhanced Capabilities
version: 0.1.5
description: Enhances Luna's responses with real-time research, emotional intelligence, learning, reasoning, and memory capabilities
author: commstech
tags: [filter, research, learning, memory, emotional intelligence, AGI]
"""

from typing import Optional, Dict, List, Any, Callable, Awaitable
import aiohttp
import logging
import time
import asyncio
from datetime import datetime
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup

class FilterException(Exception):
    """Custom exception for filter-specific errors"""
    pass

class ContextMemory:
    def __init__(self):
        self.current_date = time.strftime("%B %d, %Y")
        self.learned_facts = {}
        self.last_update = time.time()
        self.update_interval = 300  # 5 minutes

    def update_context(self):
        current_time = time.time()
        if current_time - self.last_update > self.update_interval:
            self.current_date = time.strftime("%B %d, %Y")
            self.last_update = current_time

    def add_fact(self, key: str, value: Any, importance: float = 1.0):
        self.learned_facts[key] = {
            'value': value,
            'timestamp': time.time(),
            'importance': importance,
            'access_count': 0
        }

    def get_fact(self, key: str) -> Optional[Any]:
        if key in self.learned_facts:
            self.learned_facts[key]['access_count'] += 1
            return self.learned_facts[key]['value']
        return None

class EmotionalSystem:
    """Handles emotional intelligence capabilities"""
    def __init__(self):
        self.emotional_states = {
            'empathy': 0.5,
            'tone': 'neutral',
            'confidence': 0.5
        }
    
    def evaluate_emotional_state(self, content: str) -> Dict:
        # Enhanced emotion detection using sentiment analysis
        emotions = {
            'empathy': self._calculate_empathy(content),
            'tone': self._detect_tone(content),
            'confidence': self._assess_confidence(content)
        }
        return emotions

    def _calculate_empathy(self, content: str) -> float:
        # Improved empathy calculation
        empathy_keywords = ['understand', 'feel', 'appreciate', 'sorry', 'help']
        return min(1.0, sum(word in content.lower() for word in empathy_keywords) * 0.2)

    def _detect_tone(self, content: str) -> str:
        # Advanced tone detection
        if any(word in content.lower() for word in ['error', 'sorry', 'unfortunately']):
            return 'apologetic'
        elif any(word in content.lower() for word in ['great', 'excellent', 'wonderful']):
            return 'positive'
        return 'neutral'

    def _assess_confidence(self, content: str) -> float:
        # Confidence assessment
        uncertainty_markers = ['maybe', 'perhaps', 'might', 'could', 'unsure']
        return 1.0 - min(1.0, sum(word in content.lower() for word in uncertainty_markers) * 0.2)

class MemorySystem:
    """Handles memory retention and retrieval"""
    def __init__(self):
        self.short_term = []
        self.long_term = {}
        self.max_short_term = 5
        self.max_long_term = 100

    def add_memory(self, content: str):
        timestamp = time.time()
        self.short_term.append({'content': content, 'timestamp': timestamp})
        
        if len(self.short_term) > self.max_short_term:
            # Move oldest short-term memory to long-term
            oldest = self.short_term.pop(0)
            self.long_term[oldest['content']] = oldest['timestamp']

        # Cleanup old long-term memories
        if len(self.long_term) > self.max_long_term:
            oldest_key = min(self.long_term, key=self.long_term.get)
            del self.long_term[oldest_key]

    def get_relevant_memory(self, query: str) -> Optional[str]:
        # Search through both short and long term memory
        relevant_memories = []
        
        # Check short-term memory
        for memory in self.short_term:
            if any(word in memory['content'].lower() for word in query.lower().split()):
                relevant_memories.append(memory['content'])

        # Check long-term memory
        for content, _ in self.long_term.items():
            if any(word in content.lower() for word in query.lower().split()):
                relevant_memories.append(content)

        return '\n'.join(relevant_memories[-3:]) if relevant_memories else None

class ReasoningSystem:
    """Handles analysis and reasoning capabilities"""
    def __init__(self):
        self.context_history = []

    def analyze_response(self, 
                        query: str, 
                        research: str, 
                        historical_context: List[str]) -> Dict:
        confidence = self._calculate_confidence(query, research)
        completeness = self._assess_completeness(research, historical_context)
        
        return {
            'confidence': confidence,
            'completeness': completeness,
            'has_sufficient_context': confidence > 0.7 and completeness > 0.6
        }

    def _calculate_confidence(self, query: str, research: str) -> float:
        if not research:
            return 0.0
        # Calculate confidence based on research relevance
        query_terms = set(query.lower().split())
        research_terms = set(research.lower().split())
        overlap = len(query_terms.intersection(research_terms))
        return min(1.0, overlap / len(query_terms) if query_terms else 0)

    def _assess_completeness(self, research: str, historical_context: List[str]) -> float:
        if not research:
            return 0.0
        # Assess completeness based on research and historical context
        has_research = bool(research)
        has_history = bool(historical_context)
        return (has_research * 0.7 + has_history * 0.3)

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
        emotional_intelligence: bool = Field(
            default=True,
            description="Enable emotional understanding and response"
        )
        memory_retention: bool = Field(
            default=True,
            description="Enable long-term memory retention"
        )
        max_retries: int = Field(
            default=3,
            description="Maximum number of retries for failed requests"
        )
        timeout: int = Field(
            default=30,
            description="Timeout for requests in seconds"
        )

    def __init__(self):
        self.valves = self.Valves()
        self.logger = logging.getLogger("luna_filter")
        self.session = None
        self.context_memory = ContextMemory()
        self.emotional_system = EmotionalSystem()
        self.memory_system = MemorySystem()
        self.reasoning_system = ReasoningSystem()

    async def initialize_session(self):
        """Initialize aiohttp session with proper timeout"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=self.valves.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)

    async def outlet(
        self,
        body: Dict[str, Any],
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
        __user__: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process outgoing messages with enhanced capabilities"""
        try:
            await self.initialize_session()
            self.context_memory.update_context()
            current_date = self.context_memory.current_date

            messages = body.get("messages", [])
            if not messages:
                return body

            # Get messages and analyze emotional context
            luna_response = messages[-1]["content"]
            user_message = messages[-2]["content"] if len(messages) > 1 else ""
            
            emotional_state = self.emotional_system.evaluate_emotional_state(user_message)
            
            # Check memory for relevant context
            relevant_memory = self.memory_system.get_relevant_memory(user_message)
            
            # Apply reasoning to determine if research is needed
            reasoning_result = self.reasoning_system.analyze_response(
                user_message, 
                relevant_memory or "", 
                [msg["content"] for msg in messages[:-1]]
            )

            if reasoning_result['has_sufficient_context']:
                enhanced_response = self._enhance_response(
                    luna_response,
                    emotional_state,
                    relevant_memory
                )
            else:
                # Conduct research if needed
                enhanced_response = await self._research_and_enhance(
                    luna_response,
                    user_message,
                    emotional_state,
                    __event_emitter__
                )

            # Store interaction in memory
            self.memory_system.add_memory(enhanced_response)
            
            # Update message content
            messages[-1]["content"] = enhanced_response
            body["messages"] = messages
            return body

        except FilterException as e:
            self.logger.error(f"Filter error: {str(e)}")
            return self._handle_error(body, str(e))
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
            return self._handle_error(body, "An unexpected error occurred")
        finally:
            if __event_emitter__:
                await __event_emitter__({"type": "status", "data": {"description": "Processing complete", "done": True}})

    async def _research_and_enhance(
        self,
        original_response: str,
        user_message: str,
        emotional_state: Dict,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None
    ) -> str:
        """Conduct research and enhance response"""
        try:
            if __event_emitter__:
                await __event_emitter__({"type": "status", "data": {"description": "🔍 Researching...", "done": False}})

            search_results = await self._search_with_retry(user_message)
            if not search_results:
                return self._format_response(original_response, [], emotional_state)

            scraped_content = await self._scrape_pages(search_results)
            if not scraped_content:
                return self._format_response(original_response, [], emotional_state)

            processed_info = await self._process_information(user_message, scraped_content)
            
            return self._format_response(original_response, processed_info, emotional_state)

        except Exception as e:
            self.logger.error(f"Research error: {str(e)}")
            return original_response

    async def _search_with_retry(self, query: str, retries: int = None) -> List[Dict]:
        """Perform web search with retry mechanism"""
        retries = retries or self.valves.max_retries
        last_error = None

        for attempt in range(retries):
            try:
                return await self._search(query)
            except aiohttp.ClientError as e:
                last_error = e
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                continue

        self.logger.error(f"Search failed after {retries} attempts: {last_error}")
        return []

    def _format_response(
        self,
        original_response: str,
        research_info: Any,
        emotional_state: Dict
    ) -> str:
        """Format the final response with emotional context and research"""
        current_date = self.context_memory.current_date
        
        # Adjust tone based on emotional state
        tone = emotional_state.get('tone', 'neutral')
        
        response = f"As of {current_date}, "
        if tone == 'apologetic':
            response += "I apologize, but "
        elif tone == 'positive':
            response += "I'm happy to tell you that "
            
        response += original_response
        
        if research_info:
            response += f"\n\nBased on my research:\n{research_info}"
            
        return response

    def _handle_error(self, body: Dict, error: str) -> Dict:
        """Handle errors gracefully"""
        messages = body.get("messages", [])
        current_date = self.context_memory.current_date
        
        error_response = (
            f"As of {current_date}, I encountered an issue while processing your request. "
            f"Error: {error}\n\nWould you like to try again? 🔧"
        )
        
        if messages:
            messages[-1]["content"] = error_response
        else:
            messages.append({"role": "assistant", "content": error_response})
            
        body["messages"] = messages
        return body

    async def _search(self, query: str) -> List[Dict]:
        """Perform web search"""
        try:
            params = {
                "q": query,
                "format": "json",
                "engines": "google,duckduckgo,brave",
                "limit": 5,
            }

            async with self.session.get(
                self.valves.search_url, params=params, timeout=10
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("results", [])[:5]
                return []
        except Exception as e:
            self.logger.error(f"Search error: {str(e)}")
            return []

    async def _scrape_pages(self, search_results: List[Dict]) -> List[Dict]:
        """Scrape content from search results"""
        scraped_data = []
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

        for result in search_results:
            try:
                async with self.session.get(result["url"], headers=headers, timeout=10) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, "html.parser")
                        content = ""

                        # Extract main content
                        main_content = soup.find(["article", "main", '[role="main"]', "#content", ".content"])
                        if main_content:
                            content = main_content.get_text(strip=True)

                        # Fallback to paragraphs with filtering
                        if not content:
                            paragraphs = soup.find_all("p")
                            filtered_paragraphs = [
                                p.get_text(strip=True) for p in paragraphs
                                if len(p.get_text(strip=True)) > 50 and not any(
                                    skip in p.get_text(strip=True).lower()
                                    for skip in ["cookie", "privacy policy", "terms of service"]
                                )
                            ]
                            content = " ".join(filtered_paragraphs)

                        # Clean up the content
                        content = " ".join(content.split())  # Remove extra whitespace

                        # Limit content length and summarize
                        content_summary = content[:1000]  # Adjust length as needed

                        scraped_data.append({
                            "title": result["title"],
                            "url": result["url"],
                            "content": content_summary,
                            "date_scraped": time.strftime("%Y-%m-%d"),
                            "timestamp": int(time.time()),
                        })

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
            6. If information seems outdated, recommend checking official sources"""

            url = f"{self.valves.api_url}/v1/chat/completions"
            payload = {
                "model": self.valves.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
                ],
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

    def _is_response_coherent(self, response: str) -> bool:
        """Check if the response is coherent and makes sense"""
        # Implement logic to evaluate response coherence
        return "error" not in response.lower() and len(response.split()) > 10

def register():
    """Register the filter"""
    return Filter()
