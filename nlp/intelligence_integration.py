"""
Intelligence Integration - Connects Phase 2 intelligence enhancements to Anima
Provides unified interface for memory-NLP feedback loop and custom entity training
"""

import os
import sys
import logging
from typing import Dict, List, Any, Optional, Tuple
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import NLP system
from nlp.integration import get_instance as get_nlp

# Import memory feedback
try:
    from nlp.memory_feedback import get_instance as get_memory_feedback
    MEMORY_FEEDBACK_AVAILABLE = True
except ImportError:
    logger.warning("Memory feedback module not available")
    MEMORY_FEEDBACK_AVAILABLE = False

# Import custom entity training
try:
    from nlp.custom_entities import get_instance as get_entity_trainer
    from nlp.custom_entities import initialize as initialize_entity_trainer
    CUSTOM_ENTITIES_AVAILABLE = True
except ImportError:
    logger.warning("Custom entity training module not available")
    CUSTOM_ENTITIES_AVAILABLE = False

# Import knowledge graph
try:
    from knowledge.graph import get_instance as get_knowledge_graph
    KNOWLEDGE_GRAPH_AVAILABLE = True
except ImportError:
    logger.warning("Knowledge graph module not available")
    KNOWLEDGE_GRAPH_AVAILABLE = False


class IntelligenceManager:
    """
    Manages enhanced intelligence features for Anima
    Coordinates memory feedback loop and custom entity training
    """
    
    def __init__(self):
        """Initialize the intelligence manager"""
        # Get NLP system
        self.nlp = get_nlp()
        if not self.nlp:
            raise ValueError("NLP system must be available for intelligence manager")
        
        # Get memory feedback
        self.memory_feedback = get_memory_feedback() if MEMORY_FEEDBACK_AVAILABLE else None
        if not self.memory_feedback:
            logger.warning("Memory feedback not available")
        
        # Get entity trainer (don't initialize automatically as it's resource-intensive)
        self.entity_trainer = get_entity_trainer()
        
        # Get knowledge graph
        self.kg = get_knowledge_graph() if KNOWLEDGE_GRAPH_AVAILABLE else None
        if not self.kg:
            logger.warning("Knowledge graph not available")
        
        # Statistics
        self.stats = {
            "enriched_memories": 0,
            "custom_entities_detected": 0,
            "context_enhanced_analyses": 0
        }
        
        logger.info("Intelligence manager initialized")
    
    def analyze_with_intelligence(self, user_input: str, 
                                conversation_context: Dict[str, Any] = None,
                                memories: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform enhanced analysis with memory context and custom entities
        
        Args:
            user_input: User's message
            conversation_context: Current conversation context
            memories: Recent memories for context
            
        Returns:
            Enhanced analysis results
        """
        results = {
            "basic_analysis": {},
            "enhanced_analysis": {},
            "custom_entities": [],
            "memory_relevance": {},
            "knowledge_connections": []
        }
        
        # Step 1: Get basic NLP analysis
        if self.nlp:
            results["basic_analysis"] = self.nlp.analyze_text(user_input)
        
        # Step 2: Enhance with memory context if available
        if self.memory_feedback and memories:
            try:
                results["enhanced_analysis"] = self.memory_feedback.provide_context_to_nlp(user_input, memories)
                self.stats["context_enhanced_analyses"] += 1
            except Exception as e:
                logger.error(f"Error enhancing with memory context: {e}")
                results["enhanced_analysis"] = results["basic_analysis"]
        else:
            results["enhanced_analysis"] = results["basic_analysis"]
        
        # Step 3: Detect custom entities if available
        if self.entity_trainer:
            try:
                custom_entities = self.entity_trainer.identify_custom_entities(user_input)
                results["custom_entities"] = custom_entities
                self.stats["custom_entities_detected"] += len(custom_entities)
            except Exception as e:
                logger.error(f"Error detecting custom entities: {e}")
        
        # Step 4: Get knowledge connections if available
        if self.kg and "entities" in results["enhanced_analysis"]:
            try:
                # Add entities to knowledge graph
                entity_ids = self.kg.add_entities_from_analysis(
                    results["enhanced_analysis"]["entities"],
                    source="user_input"
                )
                
                # Get connections for each entity
                connections = []
                for entity_id in entity_ids:
                    entity_graph = self.kg.get_entity_graph(entity_id, depth=1)
                    if entity_graph and entity_graph["nodes"]:
                        connections.append({
                            "entity_id": entity_id,
                            "entity_text": next((n["text"] for n in entity_graph["nodes"] if n["id"] == entity_id), ""),
                            "connections": entity_graph
                        })
                
                results["knowledge_connections"] = connections
            except Exception as e:
                logger.error(f"Error getting knowledge connections: {e}")
        
        # Step 5: Rank memories by relevance if available
        if self.memory_feedback and memories and len(memories) > 0:
            try:
                ranked_memories = self.memory_feedback.rank_memories(memories, user_input)
                # Keep only the relevance scores
                memory_relevance = {m["id"]: m["relevance_score"] for m in ranked_memories if "id" in m}
                results["memory_relevance"] = memory_relevance
            except Exception as e:
                logger.error(f"Error ranking memories: {e}")
        
        return results
    
    def enrich_memory(self, memory_id: str, memory_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich a memory with NLP analysis and intelligence
        
        Args:
            memory_id: Memory ID
            memory_content: Memory content
            
        Returns:
            Enriched memory
        """
        # Skip if memory feedback not available
        if not self.memory_feedback:
            logger.warning("Memory feedback not available for enrichment")
            return memory_content
        
        try:
            enriched = self.memory_feedback.enrich_memory(memory_id, memory_content)
            self.stats["enriched_memories"] += 1
            return enriched
        except Exception as e:
            logger.error(f"Error enriching memory: {e}")
            return memory_content
    
    def learn_from_conversation(self, user_input: str, ai_response: str) -> Dict[str, Any]:
        """
        Extract learnings from conversation
        
        Args:
            user_input: User's message
            ai_response: AI's response
            
        Returns:
            Learning results
        """
        results = {
            "learned_entities": [],
            "updated_knowledge": False
        }
        
        # Extract custom entity examples if available
        if self.entity_trainer:
            try:
                example_ids = self.entity_trainer.extract_training_examples_from_conversation(
                    user_input, ai_response
                )
                if example_ids:
                    results["learned_entities"] = example_ids
                    logger.info(f"Extracted {len(example_ids)} entity examples from conversation")
            except Exception as e:
                logger.error(f"Error extracting entity examples: {e}")
        
        # Update knowledge graph if available
        if self.kg and results["learned_entities"]:
            try:
                # Add user input to knowledge graph
                self.kg.add_entities_from_text(user_input, source="conversation_learning")
                results["updated_knowledge"] = True
            except Exception as e:
                logger.error(f"Error updating knowledge graph: {e}")
        
        return results
    
    def train_custom_entities(self) -> bool:
        """
        Train the custom entity recognition model
        
        Returns:
            Success status
        """
        if not self.entity_trainer:
            # Try to initialize it if needed for training
            if CUSTOM_ENTITIES_AVAILABLE:
                logger.info("Initializing entity trainer for training")
                self.entity_trainer = initialize_entity_trainer()
            
            if not self.entity_trainer:
                logger.warning("Custom entity trainer not available for training")
                return False
        
        try:
            success = self.entity_trainer.train()
            if success:
                logger.info("Successfully trained custom entity model")
            else:
                logger.warning("Custom entity training completed but may not have been successful")
            return success
        except Exception as e:
            logger.error(f"Error training custom entities: {e}")
            return False
    
    def enhance_prompt(self, base_prompt: str, user_input: str, 
                      conversation_context: Dict[str, Any] = None,
                      memories: List[Dict[str, Any]] = None) -> str:
        """
        Enhance an LLM prompt with intelligence features
        
        Args:
            base_prompt: Base prompt text
            user_input: User's message
            conversation_context: Current conversation context
            memories: Recent memories for context
            
        Returns:
            Enhanced prompt
        """
        # Start with base prompt
        enhanced_prompt = base_prompt
        
        # Get full intelligence analysis
        analysis = self.analyze_with_intelligence(user_input, conversation_context, memories)
        
        # Add custom entities if detected
        custom_entities = analysis.get("custom_entities", [])
        if custom_entities:
            entity_info = "\n\nDetected custom entities:\n"
            for i, entity in enumerate(custom_entities[:5]):  # Limit to top 5
                entity_info += f"- {entity['text']} ({entity['type']})\n"
            enhanced_prompt += entity_info
        
        # Add knowledge connections if available
        connections = analysis.get("knowledge_connections", [])
        if connections:
            connection_info = "\n\nKnowledge connections:\n"
            for i, conn in enumerate(connections[:3]):  # Limit to top 3
                entity_text = conn.get("entity_text", "")
                conn_graph = conn.get("connections", {})
                related = [n.get("text", "") for n in conn_graph.get("nodes", []) 
                          if n.get("id") != conn.get("entity_id")]
                related = related[:3]  # Limit related items
                
                if entity_text and related:
                    connection_info += f"- {entity_text} is related to: {', '.join(related)}\n"
            
            enhanced_prompt += connection_info
        
        # Add memory relevance if available
        memory_relevance = analysis.get("memory_relevance", {})
        if memory_relevance:
            # Get top memories by relevance
            top_memories = sorted(memory_relevance.items(), key=lambda x: x[1], reverse=True)[:3]
            
            if top_memories and top_memories[0][1] > 0.5:  # Only add if relevance > 0.5
                memory_info = "\n\nRelevant memories detected:\n"
                
                # Find the memory content for each memory ID
                for memory_id, relevance in top_memories:
                    if relevance < 0.3:  # Skip low relevance
                        continue
                        
                    # Find this memory in the provided memories
                    memory_content = next((m for m in memories if m.get("id") == memory_id), None)
                    if memory_content:
                        # Get a summary from the memory
                        summary = ""
                        for field in ["summary", "content", "message", "text"]:
                            if field in memory_content and memory_content[field]:
                                summary = memory_content[field]
                                if len(summary) > 100:
                                    summary = summary[:100] + "..."
                                break
                        
                        if summary:
                            memory_info += f"- Memory ({relevance:.2f}): {summary}\n"
                
                enhanced_prompt += memory_info
        
        return enhanced_prompt
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about intelligence features
        
        Returns:
            Statistics dictionary
        """
        stats = dict(self.stats)
        
        # Add knowledge graph stats if available
        if self.kg:
            kg_stats = self.kg.get_statistics()
            stats["knowledge_graph"] = kg_stats
        
        # Add custom entity stats if available
        if self.entity_trainer:
            stats["custom_entity_types"] = self.entity_trainer.get_custom_entity_types()
            stats["training_examples"] = len(self.entity_trainer.training_examples)
        
        return stats


# Create singleton instance
_intelligence_manager = None

def initialize() -> IntelligenceManager:
    """Initialize the intelligence manager singleton"""
    global _intelligence_manager
    if _intelligence_manager is None:
        logger.info("Initializing intelligence manager...")
        try:
            _intelligence_manager = IntelligenceManager()
            logger.info("Intelligence manager initialized")
        except Exception as e:
            logger.error(f"Error initializing intelligence manager: {e}")
            return None
    return _intelligence_manager

def get_instance() -> Optional[IntelligenceManager]:
    """Get the intelligence manager singleton instance"""
    global _intelligence_manager
    # Auto-initialize if not already done
    if _intelligence_manager is None:
        return initialize()
    return _intelligence_manager

# Helper functions for easy access

def analyze_with_intelligence(user_input: str, context=None, memories=None) -> Dict[str, Any]:
    """Analyze user input with all intelligence features"""
    im = get_instance()
    if im:
        return im.analyze_with_intelligence(user_input, context, memories)
    return {}

def enhance_memory(memory_id: str, memory_content: Dict[str, Any]) -> Dict[str, Any]:
    """Enrich a memory with intelligence features"""
    im = get_instance()
    if im:
        return im.enrich_memory(memory_id, memory_content)
    return memory_content

def learn_from_conversation(user_input: str, ai_response: str) -> Dict[str, Any]:
    """Extract learnings from a conversation"""
    im = get_instance()
    if im:
        return im.learn_from_conversation(user_input, ai_response)
    return {}

def enhance_prompt_with_intelligence(base_prompt: str, user_input: str, context=None, memories=None) -> str:
    """Enhance an LLM prompt with intelligence features"""
    im = get_instance()
    if im:
        return im.enhance_prompt(base_prompt, user_input, context, memories)
    return base_prompt

def get_intelligence_help() -> str:
    """Get help text for intelligence features"""
    return """
# Enhanced Intelligence Features

Anima now has advanced intelligence capabilities:

## Memory-NLP Feedback Loop
- **Memory Enrichment**: Memories are automatically analyzed and enhanced with entities, sentiment, and topics
- **Contextual Analysis**: Your messages are analyzed in the context of relevant memories
- **Relevance Ranking**: Memories are ranked by relevance to your current message
- **Bidirectional Learning**: NLP insights improve memory retrieval, and memory context improves NLP understanding

## Custom Entity Recognition
- **Domain-Specific Entities**: Anima learns to recognize entities specific to your world
- **Incremental Learning**: Continuously improves understanding from conversations
- **Custom Categories**: Define your own entity types beyond standard categories
- **Entity Suggestions**: Anima can suggest potential custom entities in text

## Knowledge Integration
- **Connected Understanding**: Links NLP insights with the knowledge graph
- **Entity Tracking**: Recognizes and remembers important entities over time
- **Relationship Inference**: Automatically infers relationships between entities
- **Memory Enhancement**: Uses knowledge connections to enrich memory retrieval

All processing happens locally on your machine.
"""

# Auto-initialize the intelligence manager
try:
    initialize()
except Exception as e:
    logger.error(f"Error during auto-initialization of intelligence manager: {e}")


if __name__ == "__main__":
    # Simple test
    im = get_instance()
    if im:
        # Test analysis
        analysis = im.analyze_with_intelligence(
            "I'm going to visit Paris next month for a tech conference at the Eiffel Tower."
        )
        
        print("Intelligence analysis:")
        print(json.dumps(analysis, indent=2))
