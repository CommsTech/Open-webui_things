"""
title: Modern Find Learn Remember - Enhanced Self-Learning Module
version: 1.0.0
description: Modern, secure, and performant self-learning module with latest Ollama and OpenWebUI integration
author: commstech
tags: [filter, research, learning, memory, modern, ollama-v0.11.8+, openwebui-v0.6.26+]
"""

import asyncio
import time
from typing import Optional, Dict, List, Any, Callable, Awaitable
from datetime import datetime

from modern_openwebui_base import (
    ModernBaseFilter, ModernValves, ModernOllamaClient, 
    ModernWebSearch, ModernContentScraper
)

class EnhancedLearningValves(ModernValves):
    """Enhanced configuration for learning capabilities"""
    
    # Learning Configuration
    enable_autolearn: bool = Field(
        default=True,
        description="Enable or disable real-time learning"
    )
    learning_rate: float = Field(
        default=0.1,
        description="Learning rate for knowledge updates"
    )
    max_knowledge_items: int = Field(
        default=1000,
        description="Maximum number of knowledge items to store"
    )
    knowledge_expiry_days: int = Field(
        default=30,
        description="Days before knowledge items expire"
    )
    
    # Research Configuration
    research_depth: str = Field(
        default="balanced",  # shallow, balanced, deep
        description="Depth of research to perform"
    )
    enable_cross_referencing: bool = Field(
        default=True,
        description="Enable cross-referencing between sources"
    )
    min_confidence_threshold: float = Field(
        default=0.7,
        description="Minimum confidence threshold for information"
    )
    
    # Content Processing Configuration
    max_content_length: int = Field(
        default=3000,
        description="Maximum content length for processing"
    )
    enable_content_summarization: bool = Field(
        default=True,
        description="Enable automatic content summarization"
    )
    enable_fact_extraction: bool = Field(
        default=True,
        description="Enable automatic fact extraction"
    )

class KnowledgeItem:
    """Represents a piece of learned knowledge"""
    
    def __init__(
        self,
        query: str,
        content: str,
        sources: List[Dict[str, Any]],
        confidence: float,
        timestamp: float = None
    ):
        self.query = query
        self.content = content
        self.sources = sources
        self.confidence = confidence
        self.timestamp = timestamp or time.time()
        self.access_count = 0
        self.last_accessed = self.timestamp
        self.related_queries = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "query": self.query,
            "content": self.content,
            "sources": self.sources,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "related_queries": self.related_queries
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeItem':
        """Create from dictionary"""
        item = cls(
            query=data["query"],
            content=data["content"],
            sources=data["sources"],
            confidence=data["confidence"],
            timestamp=data["timestamp"]
        )
        item.access_count = data.get("access_count", 0)
        item.last_accessed = data.get("last_accessed", item.timestamp)
        item.related_queries = data.get("related_queries", [])
        return item

class KnowledgeBase:
    """Intelligent knowledge base with learning capabilities"""
    
    def __init__(self, valves: EnhancedLearningValves):
        self.valves = valves
        self.knowledge_items: Dict[str, KnowledgeItem] = {}
        self.query_index: Dict[str, List[str]] = {}  # term -> query_ids
        self.last_cleanup = time.time()
        self.cleanup_interval = 3600  # 1 hour
    
    def add_knowledge(
        self,
        query: str,
        content: str,
        sources: List[Dict[str, Any]],
        confidence: float
    ) -> None:
        """Add new knowledge to the base"""
        # Create knowledge item
        knowledge_item = KnowledgeItem(query, content, sources, confidence)
        
        # Index by query
        self.knowledge_items[query] = knowledge_item
        
        # Index by terms for search
        terms = self._extract_terms(query)
        for term in terms:
            if term not in self.query_index:
                self.query_index[term] = []
            if query not in self.query_index[term]:
                self.query_index[term].append(query)
        
        # Cleanup if needed
        self._cleanup_if_needed()
    
    def find_knowledge(self, query: str, min_confidence: float = None) -> List[KnowledgeItem]:
        """Find relevant knowledge for a query"""
        min_confidence = min_confidence or self.valves.min_confidence_threshold
        
        # Extract terms from query
        query_terms = self._extract_terms(query)
        
        # Find matching queries
        matching_queries = set()
        for term in query_terms:
            if term in self.query_index:
                matching_queries.update(self.query_index[term])
        
        # Get knowledge items and rank by relevance
        knowledge_items = []
        for query_id in matching_queries:
            if query_id in self.knowledge_items:
                item = self.knowledge_items[query_id]
                if item.confidence >= min_confidence:
                    # Update access statistics
                    item.access_count += 1
                    item.last_accessed = time.time()
                    knowledge_items.append(item)
        
        # Rank by relevance and recency
        knowledge_items.sort(
            key=lambda x: (
                x.confidence * 0.6 +  # Confidence weight
                (time.time() - x.last_accessed) / 86400 * 0.2 +  # Recency weight
                x.access_count / 100 * 0.2  # Popularity weight
            ),
            reverse=True
        )
        
        return knowledge_items[:10]  # Return top 10
    
    def _extract_terms(self, text: str) -> List[str]:
        """Extract meaningful terms from text"""
        # Simple term extraction - can be enhanced with NLP
        words = text.lower().split()
        # Filter out common stop words and short terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        terms = [word for word in words if len(word) > 2 and word not in stop_words]
        return list(set(terms))
    
    def _cleanup_if_needed(self):
        """Clean up old knowledge items if needed"""
        current_time = time.time()
        if current_time - self.last_cleanup > self.cleanup_interval:
            self._cleanup_expired_items()
            self.last_cleanup = current_time
    
    def _cleanup_expired_items(self):
        """Remove expired knowledge items"""
        current_time = time.time()
        expiry_seconds = self.valves.knowledge_expiry_days * 86400
        
        expired_queries = []
        for query, item in self.knowledge_items.items():
            if current_time - item.timestamp > expiry_seconds:
                expired_queries.append(query)
        
        # Remove expired items
        for query in expired_queries:
            del self.knowledge_items[query]
        
        # Clean up index
        for term, queries in self.query_index.items():
            self.query_index[term] = [
                q for q in queries if q in self.knowledge_items
            ]
            if not self.query_index[term]:
                del self.query_index[term]
        
        # Limit total items if needed
        if len(self.knowledge_items) > self.valves.max_knowledge_items:
            # Remove least relevant items
            sorted_items = sorted(
                self.knowledge_items.items(),
                key=lambda x: x[1].confidence + x[1].access_count / 100
            )
            
            items_to_remove = len(self.knowledge_items) - self.valves.max_knowledge_items
            for i in range(items_to_remove):
                query = sorted_items[i][0]
                del self.knowledge_items[query]
        
        logger.info(f"Cleaned up {len(expired_queries)} expired knowledge items")

class ModernFindLearnRemember(ModernBaseFilter):
    """Modern Find Learn Remember filter with enhanced capabilities"""
    
    def __init__(self, valves: Optional[EnhancedLearningValves] = None):
        self.valves = valves or EnhancedLearningValves()
        super().__init__(self.valves)
        
        # Initialize learning components
        self.knowledge_base = KnowledgeBase(self.valves)
        self.research_cache = {}
        self.last_research_cleanup = time.time()
    
    async def _process_user_message(
        self,
        user_message: str,
        messages: List[Dict[str, Any]],
        event_emitter: Optional[Callable[[Any], Awaitable[None]]] = None,
        user: Optional[Dict[str, Any]] = None
    ) -> str:
        """Process user message with learning capabilities"""
        try:
            # Check if we have existing knowledge
            existing_knowledge = self.knowledge_base.find_knowledge(user_message)
            
            if existing_knowledge and self._is_knowledge_sufficient(existing_knowledge, user_message):
                # Use existing knowledge
                return self._format_existing_knowledge(existing_knowledge, user_message)
            
            # Perform research if needed
            if self.valves.enable_autolearn:
                return await self._research_and_learn(user_message, event_emitter)
            else:
                return f"I don't have enough information about '{user_message}'. Research is currently disabled."
                
        except Exception as e:
            self.logger.error(f"Error processing user message: {e}")
            return f"I encountered an error while processing your request: {str(e)}"
    
    def _is_knowledge_sufficient(
        self,
        knowledge_items: List[KnowledgeItem],
        query: str
    ) -> bool:
        """Check if existing knowledge is sufficient for the query"""
        if not knowledge_items:
            return False
        
        # Check if any item has high confidence
        high_confidence_items = [
            item for item in knowledge_items
            if item.confidence >= self.valves.min_confidence_threshold
        ]
        
        if not high_confidence_items:
            return False
        
        # Check if content is recent enough
        current_time = time.time()
        recent_items = [
            item for item in high_confidence_items
            if current_time - item.timestamp < 86400 * 7  # Within 7 days
        ]
        
        return len(recent_items) > 0
    
    def _format_existing_knowledge(
        self,
        knowledge_items: List[KnowledgeItem],
        query: str
    ) -> str:
        """Format existing knowledge for response"""
        if not knowledge_items:
            return f"I don't have specific information about '{query}'."
        
        # Get the best knowledge item
        best_item = knowledge_items[0]
        
        response = f"Based on my knowledge about '{query}':\n\n"
        response += best_item.content
        
        # Add source information
        if best_item.sources:
            response += "\n\n**Sources:**\n"
            for source in best_item.sources[:3]:  # Show top 3 sources
                response += f"- {source.get('title', 'Unknown')}\n"
        
        # Add confidence and timestamp
        confidence_text = "high" if best_item.confidence >= 0.8 else "moderate" if best_item.confidence >= 0.6 else "low"
        response += f"\n\n*Confidence: {confidence_text} | Last updated: {datetime.fromtimestamp(best_item.timestamp).strftime('%Y-%m-%d')}*"
        
        return response
    
    async def _research_and_learn(
        self,
        query: str,
        event_emitter: Optional[Callable[[Any], Awaitable[None]]] = None
    ) -> str:
        """Perform research and learn new information"""
        try:
            # Update status
            if event_emitter:
                await event_emitter({
                    "type": "status",
                    "data": {
                        "description": "🔍 Researching for new information...",
                        "done": False
                    }
                })
            
            # Perform web search
            search_results = await self.web_search.search(query)
            
            if not search_results:
                return f"I couldn't find any information about '{query}' through web search."
            
            # Update status
            if event_emitter:
                await event_emitter({
                    "type": "status",
                    "data": {
                        "description": "📖 Extracting content from sources...",
                        "done": False
                    }
                })
            
            # Extract URLs from search results
            urls = [result["url"] for result in search_results if result.get("url")]
            
            # Scrape content
            scraped_content = await self.content_scraper.scrape_content(
                urls, self.valves.max_content_length
            )
            
            if not scraped_content:
                return f"I found search results for '{query}' but couldn't extract meaningful content."
            
            # Update status
            if event_emitter:
                await event_emitter({
                    "type": "status",
                    "data": {
                        "description": "🧠 Processing and learning new information...",
                        "done": False
                    }
                })
            
            # Process information with LLM
            processed_info = await self._process_with_llm(query, scraped_content, search_results)
            
            # Learn the new information
            confidence = self._calculate_confidence(scraped_content, query)
            self.knowledge_base.add_knowledge(
                query=query,
                content=processed_info,
                sources=search_results,
                confidence=confidence
            )
            
            return processed_info
            
        except Exception as e:
            self.logger.error(f"Research and learning failed: {e}")
            return f"I encountered an error while researching '{query}': {str(e)}"
    
    async def _process_with_llm(
        self,
        query: str,
        scraped_content: List[Dict[str, Any]],
        search_results: List[Dict[str, Any]]
    ) -> str:
        """Process scraped content with LLM for learning"""
        try:
            # Prepare context for LLM
            context = self._prepare_llm_context(scraped_content, search_results)
            
            # Create system prompt
            system_prompt = """You are an intelligent AI assistant that processes and learns from web content.

Your task is to:
1. Analyze the provided web content
2. Extract key facts and information
3. Synthesize a comprehensive response
4. Ensure accuracy and relevance
5. Provide clear, well-structured information

Important guidelines:
- Always verify information from multiple sources when possible
- Be transparent about any uncertainties
- Focus on factual, verifiable information
- Organize information logically
- Include relevant context and dates when available"""

            # Create messages for LLM
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {query}\n\nContext:\n{context}"}
            ]
            
            # Get LLM response
            response = await self.ollama_client.chat_completion(
                messages=messages,
                stream=False,
                temperature=0.3,  # Lower temperature for more factual responses
                max_tokens=2000
            )
            
            if "message" in response and "content" in response["message"]:
                return response["message"]["content"]
            else:
                return "I processed the information but couldn't generate a proper response."
                
        except Exception as e:
            self.logger.error(f"LLM processing failed: {e}")
            # Fallback to basic content processing
            return self._fallback_content_processing(scraped_content, search_results)
    
    def _prepare_llm_context(
        self,
        scraped_content: List[Dict[str, Any]],
        search_results: List[Dict[str, Any]]
    ) -> str:
        """Prepare context for LLM processing"""
        context = "Web Content Sources:\n\n"
        
        for i, (content_item, search_item) in enumerate(zip(scraped_content, search_results)):
            context += f"Source {i+1}: {search_item.get('title', 'Unknown Title')}\n"
            context += f"URL: {search_item.get('url', 'Unknown URL')}\n"
            context += f"Content: {content_item.get('content', 'No content available')[:1000]}...\n\n"
        
        return context
    
    def _calculate_confidence(
        self,
        scraped_content: List[Dict[str, Any]],
        query: str
    ) -> float:
        """Calculate confidence in the extracted information"""
        if not scraped_content:
            return 0.0
        
        # Base confidence from number of sources
        source_confidence = min(1.0, len(scraped_content) / 3.0)
        
        # Content quality confidence
        total_content_length = sum(
            len(item.get("content", "")) for item in scraped_content
        )
        content_confidence = min(1.0, total_content_length / 5000.0)
        
        # Relevance confidence (simple keyword matching)
        query_terms = set(query.lower().split())
        relevance_scores = []
        
        for item in scraped_content:
            content = item.get("content", "").lower()
            content_terms = set(content.split())
            overlap = len(query_terms.intersection(content_terms))
            relevance = min(1.0, overlap / len(query_terms)) if query_terms else 0.0
            relevance_scores.append(relevance)
        
        relevance_confidence = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
        
        # Weighted average
        final_confidence = (
            source_confidence * 0.3 +
            content_confidence * 0.3 +
            relevance_confidence * 0.4
        )
        
        return min(1.0, max(0.0, final_confidence))
    
    def _fallback_content_processing(
        self,
        scraped_content: List[Dict[str, Any]],
        search_results: List[Dict[str, Any]]
    ) -> str:
        """Fallback content processing when LLM fails"""
        if not scraped_content:
            return "No content was extracted from the sources."
        
        # Simple content aggregation
        response = "Based on the available sources:\n\n"
        
        for i, (content_item, search_item) in enumerate(scraped_content[:3]):  # Limit to 3 sources
            response += f"**{search_item.get('title', 'Source ' + str(i+1))}**\n"
            content = content_item.get("content", "")
            if len(content) > 500:
                content = content[:500] + "..."
            response += f"{content}\n\n"
        
        response += "*Note: This is a basic summary as advanced processing was unavailable.*"
        return response
    
    async def cleanup(self):
        """Cleanup resources"""
        await super().cleanup()
        
        # Clean up research cache
        current_time = time.time()
        if current_time - self.last_research_cleanup > 3600:  # Every hour
            expired_keys = [
                key for key, value in self.research_cache.items()
                if current_time - value["timestamp"] > 3600  # 1 hour TTL
            ]
            
            for key in expired_keys:
                del self.research_cache[key]
            
            self.last_research_cleanup = current_time

def register():
    """Register the modern Find Learn Remember filter"""
    return ModernFindLearnRemember()