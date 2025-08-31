"""
title: Modern Luna Enhanced Capabilities
version: 1.0.0
description: Modern, secure, and performant enhanced capabilities module with latest Ollama and OpenWebUI integration
author: commstech
tags: [filter, enhanced, emotional intelligence, memory, reasoning, modern, ollama-v0.11.8+, openwebui-v0.6.26+]
"""

import asyncio
import time
import json
from typing import Optional, Dict, List, Any, Callable, Awaitable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass

from modern_openwebui_base import (
    ModernBaseFilter, ModernValves, ModernOllamaClient, 
    ModernWebSearch, ModernContentScraper
)

class EnhancedCapabilitiesValves(ModernValves):
    """Enhanced configuration for advanced capabilities"""
    
    # Emotional Intelligence Configuration
    enable_emotional_intelligence: bool = Field(
        default=True,
        description="Enable emotional understanding and response"
    )
    emotional_sensitivity: float = Field(
        default=0.7,
        description="Sensitivity level for emotional detection"
    )
    enable_empathy: bool = Field(
        default=True,
        description="Enable empathetic responses"
    )
    tone_adaptation: bool = Field(
        default=True,
        description="Adapt tone based on user emotional state"
    )
    
    # Memory Configuration
    enable_memory_retention: bool = Field(
        default=True,
        description="Enable long-term memory retention"
    )
    short_term_memory_size: int = Field(
        default=10,
        description="Size of short-term memory buffer"
    )
    long_term_memory_size: int = Field(
        default=1000,
        description="Maximum long-term memory items"
    )
    memory_consolidation_interval: int = Field(
        default=300,
        description="Memory consolidation interval in seconds"
    )
    
    # Reasoning Configuration
    enable_advanced_reasoning: bool = Field(
        default=True,
        description="Enable advanced reasoning capabilities"
    )
    reasoning_depth: str = Field(
        default="balanced",  # shallow, balanced, deep
        description="Depth of reasoning to apply"
    )
    confidence_threshold: float = Field(
        default=0.7,
        description="Confidence threshold for reasoning"
    )
    
    # Learning Configuration
    enable_continuous_learning: bool = Field(
        default=True,
        description="Enable continuous learning from interactions"
    )
    learning_rate: float = Field(
        default=0.1,
        description="Rate of learning from interactions"
    )
    enable_pattern_recognition: bool = Field(
        default=True,
        description="Enable pattern recognition in conversations"
    )

@dataclass
class EmotionalState:
    """Represents the emotional state of a conversation"""
    primary_emotion: str
    intensity: float
    confidence: float
    valence: float  # Positive/negative
    arousal: float  # High/low energy
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_emotion": self.primary_emotion,
            "intensity": self.intensity,
            "confidence": self.confidence,
            "valence": self.valence,
            "arousal": self.arousal,
            "timestamp": self.timestamp
        }

class ModernEmotionalSystem:
    """Modern emotional intelligence system with advanced capabilities"""
    
    def __init__(self, valves: EnhancedCapabilitiesValves):
        self.valves = valves
        self.emotion_lexicon = self._build_emotion_lexicon()
        self.emotional_context = []
        self.max_context_size = 50
    
    def _build_emotion_lexicon(self) -> Dict[str, Dict[str, Any]]:
        """Build comprehensive emotion lexicon"""
        return {
            "joy": {"valence": 0.9, "arousal": 0.7, "keywords": ["happy", "excited", "great", "wonderful", "amazing"]},
            "sadness": {"valence": -0.8, "arousal": -0.3, "keywords": ["sad", "unfortunate", "sorry", "disappointed", "upset"]},
            "anger": {"valence": -0.6, "arousal": 0.8, "keywords": ["angry", "frustrated", "annoyed", "mad", "irritated"]},
            "fear": {"valence": -0.7, "arousal": 0.6, "keywords": ["worried", "concerned", "afraid", "nervous", "anxious"]},
            "surprise": {"valence": 0.0, "arousal": 0.5, "keywords": ["wow", "unexpected", "surprising", "shocking", "amazing"]},
            "disgust": {"valence": -0.9, "arousal": -0.2, "keywords": ["disgusting", "gross", "awful", "terrible", "horrible"]},
            "trust": {"valence": 0.6, "arousal": 0.2, "keywords": ["trust", "believe", "confident", "sure", "certain"]},
            "anticipation": {"valence": 0.4, "arousal": 0.4, "keywords": ["expect", "hope", "look forward", "anticipate", "await"]}
        }
    
    def analyze_emotional_state(self, content: str, context: Optional[List[str]] = None) -> EmotionalState:
        """Analyze emotional state from content and context"""
        if not self.valves.enable_emotional_intelligence:
            return EmotionalState("neutral", 0.0, 0.0, 0.0, 0.0, time.time())
        
        # Analyze current content
        current_emotion = self._analyze_content_emotion(content)
        
        # Consider context if available
        if context and len(context) > 0:
            contextual_emotion = self._analyze_contextual_emotion(context)
            # Blend current and contextual emotions
            blended_emotion = self._blend_emotions(current_emotion, contextual_emotion)
            return blended_emotion
        
        return current_emotion
    
    def _analyze_content_emotion(self, content: str) -> EmotionalState:
        """Analyze emotion from content text"""
        content_lower = content.lower()
        
        # Calculate emotion scores
        emotion_scores = {}
        for emotion, data in self.emotion_lexicon.items():
            score = 0
            for keyword in data["keywords"]:
                if keyword in content_lower:
                    score += 1
            
            if score > 0:
                emotion_scores[emotion] = score
        
        if not emotion_scores:
            return EmotionalState("neutral", 0.0, 0.5, 0.0, 0.0, time.time())
        
        # Find primary emotion
        primary_emotion = max(emotion_scores, key=emotion_scores.get)
        emotion_data = self.emotion_lexicon[primary_emotion]
        
        # Calculate intensity and confidence
        intensity = min(1.0, emotion_scores[primary_emotion] / 3.0)
        confidence = min(1.0, intensity * self.valves.emotional_sensitivity)
        
        return EmotionalState(
            primary_emotion=primary_emotion,
            intensity=intensity,
            confidence=confidence,
            valence=emotion_data["valence"],
            arousal=emotion_data["arousal"],
            timestamp=time.time()
        )
    
    def _analyze_contextual_emotion(self, context: List[str]) -> EmotionalState:
        """Analyze emotion from conversation context"""
        if not context:
            return EmotionalState("neutral", 0.0, 0.0, 0.0, 0.0, time.time())
        
        # Analyze recent context (last 5 messages)
        recent_context = context[-5:]
        context_emotions = []
        
        for message in recent_context:
            emotion = self._analyze_content_emotion(message)
            context_emotions.append(emotion)
        
        # Calculate average emotional state
        avg_valence = sum(e.valence for e in context_emotions) / len(context_emotions)
        avg_arousal = sum(e.arousal for e in context_emotions) / len(context_emotions)
        avg_intensity = sum(e.intensity for e in context_emotions) / len(context_emotions)
        
        # Determine primary emotion from context
        emotion_counts = {}
        for emotion in context_emotions:
            if emotion.primary_emotion in emotion_counts:
                emotion_counts[emotion.primary_emotion] += 1
            else:
                emotion_counts[emotion.primary_emotion] = 1
        
        primary_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else "neutral"
        
        return EmotionalState(
            primary_emotion=primary_emotion,
            intensity=avg_intensity,
            confidence=min(1.0, avg_intensity * 0.8),
            valence=avg_valence,
            arousal=avg_arousal,
            timestamp=time.time()
        )
    
    def _blend_emotions(self, current: EmotionalState, contextual: EmotionalState) -> EmotionalState:
        """Blend current and contextual emotions"""
        # Weight current emotion more heavily (70% current, 30% contextual)
        blended_valence = current.valence * 0.7 + contextual.valence * 0.3
        blended_arousal = current.arousal * 0.7 + contextual.arousal * 0.3
        blended_intensity = current.intensity * 0.7 + contextual.intensity * 0.3
        
        # Determine primary emotion (prefer current if similar intensity)
        if current.intensity > contextual.intensity * 0.8:
            primary_emotion = current.primary_emotion
        else:
            primary_emotion = contextual.primary_emotion
        
        return EmotionalState(
            primary_emotion=primary_emotion,
            intensity=blended_intensity,
            confidence=min(1.0, (current.confidence + contextual.confidence) / 2),
            valence=blended_valence,
            arousal=blended_arousal,
            timestamp=time.time()
        )
    
    def generate_empathetic_response(self, emotional_state: EmotionalState, base_response: str) -> str:
        """Generate empathetic response based on emotional state"""
        if not self.valves.enable_empathy:
            return base_response
        
        # Add emotional context to response
        if emotional_state.primary_emotion == "sadness" and emotional_state.intensity > 0.5:
            prefix = "I understand this might be difficult to hear, but "
        elif emotional_state.primary_emotion == "anger" and emotional_state.intensity > 0.5:
            prefix = "I can see this is frustrating, and I want to help. "
        elif emotional_state.primary_emotion == "fear" and emotional_state.intensity > 0.5:
            prefix = "I want to reassure you that "
        elif emotional_state.primary_emotion == "joy" and emotional_state.intensity > 0.5:
            prefix = "I'm glad to share this good news with you! "
        else:
            prefix = ""
        
        return prefix + base_response

class ModernMemorySystem:
    """Modern memory system with intelligent retention and retrieval"""
    
    def __init__(self, valves: EnhancedCapabilitiesValves):
        self.valves = valves
        self.short_term_memory: List[Dict[str, Any]] = []
        self.long_term_memory: Dict[str, Dict[str, Any]] = {}
        self.memory_index: Dict[str, List[str]] = {}  # term -> memory_ids
        self.last_consolidation = time.time()
    
    def add_memory(self, content: str, importance: float = 1.0, context: Optional[Dict[str, Any]] = None):
        """Add new memory to the system"""
        timestamp = time.time()
        memory_id = f"mem_{int(timestamp * 1000)}"
        
        memory_item = {
            "id": memory_id,
            "content": content,
            "importance": importance,
            "timestamp": timestamp,
            "context": context or {},
            "access_count": 0,
            "last_accessed": timestamp
        }
        
        # Add to short-term memory
        self.short_term_memory.append(memory_item)
        
        # Maintain short-term memory size
        if len(self.short_term_memory) > self.valves.short_term_memory_size:
            oldest_memory = self.short_term_memory.pop(0)
            # Move to long-term memory if important enough
            if oldest_memory["importance"] > 0.5:
                self._add_to_long_term_memory(oldest_memory)
        
        # Index the memory
        self._index_memory(memory_item)
        
        # Consolidate if needed
        self._consolidate_if_needed()
    
    def _add_to_long_term_memory(self, memory_item: Dict[str, Any]):
        """Add memory to long-term storage"""
        memory_id = memory_item["id"]
        self.long_term_memory[memory_id] = memory_item
        
        # Maintain long-term memory size
        if len(self.long_term_memory) > self.valves.long_term_memory_size:
            # Remove least important memories
            sorted_memories = sorted(
                self.long_term_memory.items(),
                key=lambda x: x[1]["importance"] + x[1]["access_count"] / 100
            )
            
            items_to_remove = len(self.long_term_memory) - self.valves.long_term_memory_size
            for i in range(items_to_remove):
                memory_id = sorted_memories[i][0]
                self._remove_from_index(sorted_memories[i][1])
                del self.long_term_memory[memory_id]
    
    def _index_memory(self, memory_item: Dict[str, Any]):
        """Index memory for search"""
        content = memory_item["content"].lower()
        terms = self._extract_terms(content)
        
        for term in terms:
            if term not in self.memory_index:
                self.memory_index[term] = []
            if memory_item["id"] not in self.memory_index[term]:
                self.memory_index[term].append(memory_item["id"])
    
    def _remove_from_index(self, memory_item: Dict[str, Any]):
        """Remove memory from index"""
        content = memory_item["content"].lower()
        terms = self._extract_terms(content)
        
        for term in terms:
            if term in self.memory_index:
                if memory_item["id"] in self.memory_index[term]:
                    self.memory_index[term].remove(memory_item["id"])
                if not self.memory_index[term]:
                    del self.memory_index[term]
    
    def _extract_terms(self, text: str) -> List[str]:
        """Extract meaningful terms from text"""
        words = text.lower().split()
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        terms = [word for word in words if len(word) > 2 and word not in stop_words]
        return list(set(terms))
    
    def find_relevant_memories(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find memories relevant to a query"""
        query_terms = self._extract_terms(query)
        
        # Find matching memories
        matching_memory_ids = set()
        for term in query_terms:
            if term in self.memory_index:
                matching_memory_ids.update(self.memory_index[term])
        
        # Get memory items and rank by relevance
        relevant_memories = []
        for memory_id in matching_memory_ids:
            # Check short-term memory
            for memory in self.short_term_memory:
                if memory["id"] == memory_id:
                    memory["access_count"] += 1
                    memory["last_accessed"] = time.time()
                    relevant_memories.append(memory)
                    break
            
            # Check long-term memory
            if memory_id in self.long_term_memory:
                memory = self.long_term_memory[memory_id]
                memory["access_count"] += 1
                memory["last_accessed"] = time.time()
                relevant_memories.append(memory)
        
        # Rank by relevance (importance + recency + access count)
        relevant_memories.sort(
            key=lambda x: (
                x["importance"] * 0.5 +
                (time.time() - x["timestamp"]) / 86400 * 0.3 +
                x["access_count"] / 100 * 0.2
            ),
            reverse=True
        )
        
        return relevant_memories[:limit]
    
    def _consolidate_if_needed(self):
        """Consolidate memories if needed"""
        current_time = time.time()
        if current_time - self.last_consolidation > self.valves.memory_consolidation_interval:
            self._consolidate_memories()
            self.last_consolidation = current_time
    
    def _consolidate_memories(self):
        """Consolidate related memories"""
        # This is a simplified consolidation - can be enhanced with more sophisticated algorithms
        # For now, just move old short-term memories to long-term
        current_time = time.time()
        memories_to_move = []
        
        for memory in self.short_term_memory:
            if current_time - memory["timestamp"] > 3600:  # 1 hour old
                memories_to_move.append(memory)
        
        for memory in memories_to_move:
            self.short_term_memory.remove(memory)
            if memory["importance"] > 0.3:  # Only keep moderately important memories
                self._add_to_long_term_memory(memory)

class ModernReasoningSystem:
    """Modern reasoning system with advanced analysis capabilities"""
    
    def __init__(self, valves: EnhancedCapabilitiesValves):
        self.valves = valves
        self.reasoning_cache = {}
        self.last_cache_cleanup = time.time()
    
    def analyze_response(
        self,
        query: str,
        response: str,
        context: List[str],
        emotional_state: Optional[EmotionalState] = None
    ) -> Dict[str, Any]:
        """Analyze response quality and reasoning"""
        if not self.valves.enable_advanced_reasoning:
            return {"confidence": 0.5, "completeness": 0.5, "quality": "basic"}
        
        # Calculate various quality metrics
        confidence = self._calculate_confidence(query, response, context)
        completeness = self._calculate_completeness(query, response, context)
        coherence = self._calculate_coherence(response)
        relevance = self._calculate_relevance(query, response)
        
        # Consider emotional context if available
        if emotional_state:
            emotional_adaptation = self._assess_emotional_adaptation(response, emotional_state)
        else:
            emotional_adaptation = 0.5
        
        # Calculate overall quality score
        quality_score = (
            confidence * 0.3 +
            completeness * 0.25 +
            coherence * 0.2 +
            relevance * 0.15 +
            emotional_adaptation * 0.1
        )
        
        # Determine quality level
        if quality_score >= 0.8:
            quality_level = "excellent"
        elif quality_score >= 0.6:
            quality_level = "good"
        elif quality_score >= 0.4:
            quality_level = "adequate"
        else:
            quality_level = "poor"
        
        return {
            "confidence": confidence,
            "completeness": completeness,
            "coherence": coherence,
            "relevance": relevance,
            "emotional_adaptation": emotional_adaptation,
            "overall_quality": quality_score,
            "quality_level": quality_level,
            "recommendations": self._generate_recommendations(quality_score, confidence, completeness)
        }
    
    def _calculate_confidence(self, query: str, response: str, context: List[str]) -> float:
        """Calculate confidence in the response"""
        # Check if response addresses the query
        query_terms = set(query.lower().split())
        response_terms = set(response.lower().split())
        
        if not query_terms:
            return 0.5
        
        # Calculate term overlap
        term_overlap = len(query_terms.intersection(response_terms)) / len(query_terms)
        
        # Check response length (too short might indicate low confidence)
        length_factor = min(1.0, len(response.split()) / 20.0)
        
        # Check for uncertainty indicators
        uncertainty_words = ["maybe", "perhaps", "might", "could", "unsure", "not sure", "uncertain"]
        uncertainty_count = sum(1 for word in uncertainty_words if word in response.lower())
        uncertainty_factor = max(0.0, 1.0 - uncertainty_count * 0.1)
        
        return min(1.0, (term_overlap * 0.5 + length_factor * 0.3 + uncertainty_factor * 0.2))
    
    def _calculate_completeness(self, query: str, response: str, context: List[str]) -> float:
        """Calculate completeness of the response"""
        # Check if response provides a comprehensive answer
        response_length = len(response.split())
        
        # Basic length threshold
        if response_length < 10:
            return 0.3
        elif response_length < 30:
            return 0.6
        elif response_length < 100:
            return 0.8
        else:
            return 1.0
    
    def _calculate_coherence(self, response: str) -> float:
        """Calculate coherence of the response"""
        # Simple coherence check - can be enhanced with NLP
        sentences = response.split('.')
        if len(sentences) < 2:
            return 0.8  # Single sentence is coherent
        
        # Check for logical connectors
        connectors = ["because", "therefore", "however", "although", "furthermore", "additionally"]
        connector_count = sum(1 for connector in connectors if connector in response.lower())
        
        # Check sentence length consistency
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
        if sentence_lengths:
            length_variance = max(sentence_lengths) - min(sentence_lengths)
            length_factor = max(0.0, 1.0 - length_variance / 50.0)
        else:
            length_factor = 0.5
        
        return min(1.0, (connector_count * 0.1 + length_factor * 0.9))
    
    def _calculate_relevance(self, query: str, response: str) -> float:
        """Calculate relevance of response to query"""
        query_terms = set(query.lower().split())
        response_terms = set(response.lower().split())
        
        if not query_terms:
            return 0.5
        
        # Calculate Jaccard similarity
        intersection = len(query_terms.intersection(response_terms))
        union = len(query_terms.union(response_terms))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _assess_emotional_adaptation(self, response: str, emotional_state: EmotionalState) -> float:
        """Assess how well the response adapts to emotional state"""
        if emotional_state.primary_emotion == "neutral":
            return 0.8  # Neutral responses are generally appropriate
        
        # Check for emotional keywords that match the state
        emotional_keywords = {
            "sadness": ["understand", "sorry", "unfortunate", "difficult"],
            "anger": ["frustrating", "annoying", "help", "resolve"],
            "fear": ["reassure", "safe", "help", "support"],
            "joy": ["great", "excellent", "wonderful", "happy"]
        }
        
        relevant_keywords = emotional_keywords.get(emotional_state.primary_emotion, [])
        if not relevant_keywords:
            return 0.5
        
        keyword_matches = sum(1 for keyword in relevant_keywords if keyword in response.lower())
        return min(1.0, keyword_matches * 0.2)
    
    def _generate_recommendations(self, quality_score: float, confidence: float, completeness: float) -> List[str]:
        """Generate recommendations for improvement"""
        recommendations = []
        
        if quality_score < 0.6:
            if confidence < 0.5:
                recommendations.append("Consider providing more specific and confident information")
            if completeness < 0.5:
                recommendations.append("Expand the response to cover more aspects of the query")
            if confidence < 0.3:
                recommendations.append("Acknowledge any uncertainties in the response")
        
        if quality_score < 0.4:
            recommendations.append("Consider breaking down complex responses into smaller parts")
            recommendations.append("Add examples or context to improve clarity")
        
        return recommendations

class ModernLunaEnhancedCapabilities(ModernBaseFilter):
    """Modern Luna Enhanced Capabilities filter with advanced features"""
    
    def __init__(self, valves: Optional[EnhancedCapabilitiesValves] = None):
        self.valves = valves or EnhancedCapabilitiesValves()
        super().__init__(self.valves)
        
        # Initialize enhanced components
        self.emotional_system = ModernEmotionalSystem(self.valves)
        self.memory_system = ModernMemorySystem(self.valves)
        self.reasoning_system = ModernReasoningSystem(self.valves)
        
        # Context tracking
        self.conversation_context = []
        self.max_context_size = 20
    
    async def _process_user_message(
        self,
        user_message: str,
        messages: List[Dict[str, Any]],
        event_emitter: Optional[Callable[[Any], Awaitable[None]]] = None,
        user: Optional[Dict[str, Any]] = None
    ) -> str:
        """Process user message with enhanced capabilities"""
        try:
            # Update conversation context
            self._update_conversation_context(user_message, messages)
            
            # Analyze emotional state
            emotional_state = self.emotional_system.analyze_emotional_state(
                user_message, self.conversation_context
            )
            
            # Check memory for relevant information
            relevant_memories = self.memory_system.find_relevant_memories(user_message)
            
            # Generate base response
            base_response = await self._generate_base_response(
                user_message, messages, relevant_memories, emotional_state, event_emitter
            )
            
            # Enhance response with emotional intelligence
            if self.valves.enable_emotional_intelligence:
                enhanced_response = self.emotional_system.generate_empathetic_response(
                    emotional_state, base_response
                )
            else:
                enhanced_response = base_response
            
            # Store interaction in memory
            self.memory_system.add_memory(
                content=f"User: {user_message}\nAssistant: {enhanced_response}",
                importance=0.7,
                context={"emotional_state": emotional_state.to_dict()}
            )
            
            # Analyze response quality
            quality_analysis = self.reasoning_system.analyze_response(
                user_message, enhanced_response, self.conversation_context, emotional_state
            )
            
            # Log quality metrics
            self.logger.info(f"Response quality: {quality_analysis['quality_level']} (score: {quality_analysis['overall_quality']:.2f})")
            
            return enhanced_response
            
        except Exception as e:
            self.logger.error(f"Error processing user message: {e}")
            return f"I encountered an error while processing your request: {str(e)}"
    
    def _update_conversation_context(self, user_message: str, messages: List[Dict[str, Any]]):
        """Update conversation context"""
        # Add user message to context
        self.conversation_context.append(user_message)
        
        # Add recent assistant messages
        for message in reversed(messages[-3:]):  # Last 3 messages
            if message.get("role") == "assistant":
                self.conversation_context.append(message.get("content", ""))
                break
        
        # Maintain context size
        if len(self.conversation_context) > self.max_context_size:
            self.conversation_context = self.conversation_context[-self.max_context_size:]
    
    async def _generate_base_response(
        self,
        user_message: str,
        messages: List[Dict[str, Any]],
        relevant_memories: List[Dict[str, Any]],
        emotional_state: EmotionalState,
        event_emitter: Optional[Callable[[Any], Awaitable[None]]] = None
    ) -> str:
        """Generate base response using available information"""
        # Check if we have sufficient context from memory
        if relevant_memories and len(relevant_memories) > 0:
            return self._format_memory_based_response(user_message, relevant_memories)
        
        # Check if we need to perform research
        if self._should_research(user_message):
            return await self._research_and_respond(user_message, event_emitter)
        
        # Generate response using LLM
        return await self._generate_llm_response(user_message, messages, emotional_state)
    
    def _format_memory_based_response(
        self,
        user_message: str,
        relevant_memories: List[Dict[str, Any]]
    ) -> str:
        """Format response based on relevant memories"""
        if not relevant_memories:
            return f"I don't have specific information about '{user_message}' in my memory."
        
        # Get the most relevant memory
        best_memory = relevant_memories[0]
        
        response = f"Based on our previous conversation about '{user_message}':\n\n"
        response += best_memory["content"]
        
        # Add context if available
        if best_memory.get("context"):
            context = best_memory["context"]
            if "emotional_state" in context:
                emotional_context = context["emotional_state"]
                if emotional_context.get("primary_emotion") != "neutral":
                    response += f"\n\n*Note: This was discussed in a {emotional_context['primary_emotion']} context.*"
        
        return response
    
    def _should_research(self, user_message: str) -> bool:
        """Determine if research is needed"""
        # Simple heuristic - can be enhanced
        research_keywords = ["latest", "current", "recent", "news", "update", "today", "now"]
        return any(keyword in user_message.lower() for keyword in research_keywords)
    
    async def _research_and_respond(
        self,
        user_message: str,
        event_emitter: Optional[Callable[[Any], Awaitable[None]]] = None
    ) -> str:
        """Perform research and generate response"""
        try:
            if event_emitter:
                await event_emitter({
                    "type": "status",
                    "data": {
                        "description": "🔍 Researching current information...",
                        "done": False
                    }
                })
            
            # Perform web search
            search_results = await self.web_search.search(user_message)
            
            if not search_results:
                return f"I couldn't find current information about '{user_message}'."
            
            # Scrape content
            urls = [result["url"] for result in search_results if result.get("url")]
            scraped_content = await self.content_scraper.scrape_content(urls, 2000)
            
            if not scraped_content:
                return f"I found search results but couldn't extract meaningful content."
            
            # Process with LLM
            response = await self._process_research_with_llm(user_message, scraped_content, search_results)
            
            # Store research results in memory
            self.memory_system.add_memory(
                content=f"Research on '{user_message}': {response}",
                importance=0.8,
                context={"type": "research", "sources": search_results}
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Research failed: {e}")
            return f"I encountered an error while researching '{user_message}': {str(e)}"
    
    async def _process_research_with_llm(
        self,
        query: str,
        scraped_content: List[Dict[str, Any]],
        search_results: List[Dict[str, Any]]
    ) -> str:
        """Process research results with LLM"""
        try:
            # Prepare context
            context = self._prepare_research_context(scraped_content, search_results)
            
            system_prompt = """You are an AI assistant that provides current, accurate information.

Your task is to:
1. Analyze the provided web content
2. Extract the most relevant and current information
3. Present it in a clear, organized manner
4. Highlight any important updates or changes
5. Provide context and background when helpful

Guidelines:
- Focus on the most recent information
- Be transparent about information sources
- Highlight any uncertainties or limitations
- Provide actionable insights when possible"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {query}\n\nResearch Context:\n{context}"}
            ]
            
            response = await self.ollama_client.chat_completion(
                messages=messages,
                stream=False,
                temperature=0.3
            )
            
            if "message" in response and "content" in response["message"]:
                return response["message"]["content"]
            else:
                return "I processed the research but couldn't generate a proper response."
                
        except Exception as e:
            self.logger.error(f"LLM processing failed: {e}")
            return self._fallback_research_processing(scraped_content, search_results)
    
    def _prepare_research_context(
        self,
        scraped_content: List[Dict[str, Any]],
        search_results: List[Dict[str, Any]]
    ) -> str:
        """Prepare research context for LLM"""
        context = "Research Sources:\n\n"
        
        for i, (content_item, search_item) in enumerate(zip(scraped_content, search_results)):
            context += f"Source {i+1}: {search_item.get('title', 'Unknown')}\n"
            context += f"URL: {search_item.get('url', 'Unknown')}\n"
            content = content_item.get("content", "")
            if len(content) > 800:
                content = content[:800] + "..."
            context += f"Content: {content}\n\n"
        
        return context
    
    def _fallback_research_processing(
        self,
        scraped_content: List[Dict[str, Any]],
        search_results: List[Dict[str, Any]]
    ) -> str:
        """Fallback processing when LLM fails"""
        if not scraped_content:
            return "No content was extracted from the research sources."
        
        response = "Based on my research:\n\n"
        
        for i, (content_item, search_item) in enumerate(scraped_content[:3]):
            response += f"**{search_item.get('title', f'Source {i+1}')}**\n"
            content = content_item.get("content", "")
            if len(content) > 400:
                content = content[:400] + "..."
            response += f"{content}\n\n"
        
        response += "*Note: This is a basic summary as advanced processing was unavailable.*"
        return response
    
    async def _generate_llm_response(
        self,
        user_message: str,
        messages: List[Dict[str, Any]],
        emotional_state: EmotionalState
    ) -> str:
        """Generate response using LLM"""
        try:
            # Prepare context including emotional state
            context_messages = messages[-5:] if len(messages) > 5 else messages
            
            # Add emotional context if enabled
            if self.valves.enable_emotional_intelligence:
                system_prompt = f"""You are a helpful AI assistant. The user appears to be in a {emotional_state.primary_emotion} emotional state.

Please respond appropriately, considering their emotional context while providing helpful information."""
            else:
                system_prompt = "You are a helpful AI assistant."
            
            llm_messages = [{"role": "system", "content": system_prompt}] + context_messages
            
            response = await self.ollama_client.chat_completion(
                messages=llm_messages,
                stream=False,
                temperature=0.7
            )
            
            if "message" in response and "content" in response["message"]:
                return response["message"]["content"]
            else:
                return "I'm sorry, I couldn't generate a proper response at the moment."
                
        except Exception as e:
            self.logger.error(f"LLM response generation failed: {e}")
            return f"I'm having trouble generating a response right now. Could you please rephrase your question?"
    
    async def cleanup(self):
        """Cleanup resources"""
        await super().cleanup()
        
        # Clean up conversation context
        self.conversation_context.clear()

def register():
    """Register the modern Luna Enhanced Capabilities filter"""
    return ModernLunaEnhancedCapabilities()