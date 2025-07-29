"""
Memory-NLP Feedback Loop - Enhances memory retrieval and enrichment with NLP analysis
Creates a bidirectional flow of information between the memory system and NLP understanding
"""

import os
import sys
import logging
import json
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from datetime import datetime
from pathlib import Path
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import NLP system
from nlp.integration import get_instance as get_nlp

# Try to import knowledge graph
try:
    from knowledge.graph import get_instance as get_knowledge_graph
    KNOWLEDGE_GRAPH_AVAILABLE = True
except ImportError:
    logger.warning("Knowledge graph not available for memory feedback")
    KNOWLEDGE_GRAPH_AVAILABLE = False


class MemoryFeedback:
    """
    Manages bidirectional feedback between memory and NLP understanding
    Enriches memories with NLP insights and improves NLP with memory context
    """
    
    def __init__(self):
        """Initialize the memory feedback system"""
        self.nlp = get_nlp()
        if not self.nlp:
            raise ValueError("NLP system must be available for memory feedback")
        
        # Connect to knowledge graph if available
        self.kg = get_knowledge_graph() if KNOWLEDGE_GRAPH_AVAILABLE else None
        
        # Directory for storing memory enrichments
        self.storage_dir = os.path.join(parent_dir, "nlp", "memory_enrichments")
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Memory context window
        self.memory_context = []
        self.max_context_items = 10
        
        # Last analyzed memories cache
        self.memory_analysis_cache = {}
        self.max_cache_size = 20
        
        logger.info("Memory feedback system initialized")
    
    def enrich_memory(self, memory_id: str, memory_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich a memory with NLP analysis
        
        Args:
            memory_id: Unique identifier for the memory
            memory_content: The memory content to enrich
            
        Returns:
            Enriched memory content
        """
        # Skip if already enriched recently
        if memory_id in self.memory_analysis_cache:
            logger.debug(f"Using cached enrichment for memory {memory_id}")
            return self.memory_analysis_cache[memory_id]
        
        # Create a copy to avoid modifying the original
        enriched = dict(memory_content)
        
        # Get the text content to analyze
        text_content = ""
        
        if "content" in memory_content:
            text_content = memory_content["content"]
        elif "message" in memory_content:
            text_content = memory_content["message"]
        elif "text" in memory_content:
            text_content = memory_content["text"]
        elif "summary" in memory_content:
            text_content = memory_content["summary"]
        
        if not text_content:
            logger.warning(f"No text content found in memory {memory_id}")
            return enriched
        
        # Perform NLP analysis
        try:
            analysis = self.nlp.analyze_text(text_content, 
                                           analysis_types=["entities", "sentiment", "topics", "keywords"])
            
            # Add enrichments
            enriched["nlp_enrichment"] = {
                "entities": analysis.get("entities", []),
                "sentiment": analysis.get("sentiment", {}),
                "topics": analysis.get("topics", {}),
                "keywords": analysis.get("keywords", []),
                "timestamp": datetime.now().isoformat()
            }
            
            # Add to knowledge graph if available
            if self.kg:
                entity_ids = self.kg.add_entities_from_analysis(
                    analysis.get("entities", []), 
                    source=f"memory-{memory_id}"
                )
                enriched["nlp_enrichment"]["knowledge_graph_entities"] = entity_ids
            
            # Cache the result
            self.memory_analysis_cache[memory_id] = enriched
            
            # Trim cache if needed
            if len(self.memory_analysis_cache) > self.max_cache_size:
                # Remove oldest item (first key)
                self.memory_analysis_cache.pop(next(iter(self.memory_analysis_cache)))
            
            # Save the enrichment to disk for persistence
            self._save_enrichment(memory_id, enriched["nlp_enrichment"])
            
            logger.info(f"Enriched memory {memory_id} with NLP analysis")
            return enriched
            
        except Exception as e:
            logger.error(f"Error enriching memory {memory_id}: {e}")
            return enriched
    
    def calculate_relevance(self, memory_content: Dict[str, Any], query: str) -> float:
        """
        Calculate the relevance of a memory to a query
        
        Args:
            memory_content: Memory content dict
            query: Query to compare against
            
        Returns:
            Relevance score (0-1)
        """
        if not query.strip():
            return 0.0
        
        # Basic score
        base_score = 0.0
        
        # Check for direct keyword matches
        query_keywords = set(re.findall(r'\w+', query.lower()))
        
        # Check for matches in different fields
        for field in ["content", "message", "text", "summary"]:
            if field in memory_content:
                text = memory_content[field]
                if isinstance(text, str):
                    # Count keyword matches
                    content_words = set(re.findall(r'\w+', text.lower()))
                    matches = query_keywords.intersection(content_words)
                    base_score = max(base_score, len(matches) / max(1, len(query_keywords)))
        
        # Check enriched data if available
        if "nlp_enrichment" in memory_content:
            enrichment = memory_content["nlp_enrichment"]
            
            # Check entity matches
            entity_score = 0.0
            if "entities" in enrichment:
                # Extract entity texts
                entity_texts = [e["text"].lower() for e in enrichment["entities"]]
                
                # Check if query contains any entities
                for entity in entity_texts:
                    if entity in query.lower():
                        entity_score += 0.2  # Boost for each entity match
                
                # Cap at 0.6
                entity_score = min(0.6, entity_score)
            
            # Check topic match
            topic_score = 0.0
            if "topics" in enrichment and query_keywords:
                # Get top topics
                topics = enrichment["topics"]
                if isinstance(topics, dict):
                    topic_words = set()
                    for topic, score in topics.items():
                        topic_words.update(topic.lower().split())
                    
                    # Score based on topic matches
                    matches = query_keywords.intersection(topic_words)
                    topic_score = len(matches) / max(1, len(query_keywords)) * 0.3
            
            # Check keyword match
            keyword_score = 0.0
            if "keywords" in enrichment and isinstance(enrichment["keywords"], list):
                keywords = set(k.lower() for k in enrichment["keywords"] if isinstance(k, str))
                matches = query_keywords.intersection(keywords)
                keyword_score = len(matches) / max(1, len(query_keywords)) * 0.4
            
            # Combine scores
            nlp_score = entity_score + topic_score + keyword_score
            
            # Take the best of base or NLP score, with a slight preference for NLP
            final_score = max(base_score, nlp_score * 1.1)
            
            # Cap at 1.0
            return min(1.0, final_score)
        
        return base_score
    
    def rank_memories(self, memories: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        Rank memories by relevance to a query
        
        Args:
            memories: List of memory content dicts
            query: Query to rank against
            
        Returns:
            List of memories with added relevance scores, sorted by relevance
        """
        scored_memories = []
        
        for memory in memories:
            # Calculate relevance
            relevance = self.calculate_relevance(memory, query)
            
            # Add relevance score
            memory_with_score = dict(memory)
            memory_with_score["relevance_score"] = relevance
            
            scored_memories.append(memory_with_score)
        
        # Sort by relevance (highest first)
        scored_memories.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        return scored_memories
    
    def provide_context_to_nlp(self, user_input: str, recent_memories: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Provide memory context to NLP analysis
        
        Args:
            user_input: User input to analyze
            recent_memories: Recent memory items to provide context
            
        Returns:
            Enhanced NLP analysis
        """
        if not user_input.strip():
            return {}
        
        # Start with basic analysis
        analysis = self.nlp.analyze_text(user_input)
        
        # If no memories provided, use internal context window
        if recent_memories is None:
            recent_memories = self.memory_context
        
        if not recent_memories:
            return analysis
        
        # Extract relevant context from memories
        context_texts = []
        
        for memory in recent_memories:
            # Try to get the most information-rich text
            if "content" in memory:
                context_texts.append(memory["content"])
            elif "message" in memory:
                context_texts.append(memory["message"])
            elif "summary" in memory:
                context_texts.append(memory["summary"])
        
        # If we have context, enhance the analysis
        if context_texts:
            # Create a combined context
            combined_context = " ".join(context_texts)
            
            # Get entities from the context
            try:
                context_analysis = self.nlp.analyze_text(combined_context, 
                                                      analysis_types=["entities"])
                context_entities = context_analysis.get("entities", [])
                
                # Add a 'from_context' flag to context entities
                for entity in context_entities:
                    entity["from_context"] = True
                
                # Add context entities to the analysis if not already there
                existing_texts = {e["text"].lower() for e in analysis.get("entities", [])}
                for entity in context_entities:
                    if entity["text"].lower() not in existing_texts:
                        analysis.setdefault("entities", []).append(entity)
                
                # Mark that the analysis includes context
                analysis["includes_memory_context"] = True
                
            except Exception as e:
                logger.error(f"Error analyzing memory context: {e}")
        
        return analysis
    
    def update_context_window(self, memory_item: Dict[str, Any]) -> None:
        """
        Update the memory context window with a new item
        
        Args:
            memory_item: Memory item to add to context
        """
        # Add to context window
        self.memory_context.append(memory_item)
        
        # Trim if needed
        while len(self.memory_context) > self.max_context_items:
            self.memory_context.pop(0)
    
    def clear_context_window(self) -> None:
        """Clear the memory context window"""
        self.memory_context = []
    
    def _save_enrichment(self, memory_id: str, enrichment: Dict[str, Any]) -> None:
        """
        Save memory enrichment to disk
        
        Args:
            memory_id: Memory ID
            enrichment: Enrichment data to save
        """
        try:
            # Create a safe filename
            safe_id = re.sub(r'[^\w]', '_', memory_id)
            file_path = os.path.join(self.storage_dir, f"{safe_id}.json")
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(enrichment, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving memory enrichment: {e}")
    
    def _load_enrichment(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Load memory enrichment from disk
        
        Args:
            memory_id: Memory ID
            
        Returns:
            Loaded enrichment or None if not found
        """
        try:
            # Create a safe filename
            safe_id = re.sub(r'[^\w]', '_', memory_id)
            file_path = os.path.join(self.storage_dir, f"{safe_id}.json")
            
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
                    
        except Exception as e:
            logger.error(f"Error loading memory enrichment: {e}")
            
        return None


# Create singleton instance
_memory_feedback = None

def initialize() -> MemoryFeedback:
    """Initialize the memory feedback singleton"""
    global _memory_feedback
    if _memory_feedback is None:
        logger.info("Initializing memory feedback system...")
        try:
            _memory_feedback = MemoryFeedback()
            logger.info("Memory feedback system initialized")
        except Exception as e:
            logger.error(f"Error initializing memory feedback system: {e}")
            return None
    return _memory_feedback

def get_instance() -> Optional[MemoryFeedback]:
    """Get the memory feedback singleton instance"""
    global _memory_feedback
    # Auto-initialize if not already done
    if _memory_feedback is None:
        return initialize()
    return _memory_feedback

# Helper functions for easy access
def enrich_memory(memory_id: str, memory_content: Dict[str, Any]) -> Dict[str, Any]:
    """Enrich a memory with NLP analysis"""
    mf = get_instance()
    if mf:
        return mf.enrich_memory(memory_id, memory_content)
    return memory_content

def rank_memories_by_relevance(memories: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """Rank memories by relevance to a query"""
    mf = get_instance()
    if mf:
        return mf.rank_memories(memories, query)
    return memories

def analyze_with_memory_context(user_input: str, memories: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Analyze text with memory context"""
    mf = get_instance()
    if mf:
        return mf.provide_context_to_nlp(user_input, memories)
    # Fall back to basic NLP
    nlp = get_nlp()
    return nlp.analyze_text(user_input) if nlp else {}


# Auto-initialize
try:
    initialize()
except Exception as e:
    logger.error(f"Error during auto-initialization of memory feedback: {e}")


if __name__ == "__main__":
    # Simple test
    feedback = get_instance()
    if feedback:
        # Test with a memory
        test_memory = {
            "id": "test-123",
            "content": "I went to Paris last summer and visited the Eiffel Tower. The weather was beautiful."
        }
        
        # Enrich the memory
        enriched = feedback.enrich_memory("test-123", test_memory)
        
        # Print the enriched memory
        print("Enriched memory:")
        print(json.dumps(enriched.get("nlp_enrichment", {}), indent=2))
        
        # Test relevance calculation
        print("\nRelevance to 'paris vacation':", feedback.calculate_relevance(enriched, "paris vacation"))
        print("Relevance to 'eiffel tower':", feedback.calculate_relevance(enriched, "eiffel tower"))
        print("Relevance to 'new york':", feedback.calculate_relevance(enriched, "new york"))
