"""
Knowledge Graph - Graph-based knowledge representation for Anima
Manages entities and relationships to build a structured understanding of the world
"""

import os
import sys
import logging
import json
import uuid
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from datetime import datetime
from pathlib import Path
import networkx as nx

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import NLP if available
try:
    from nlp.integration import get_instance as get_nlp
    NLP_AVAILABLE = True
except ImportError:
    logger.warning("NLP system not available for knowledge graph")
    NLP_AVAILABLE = False

class KnowledgeGraph:
    """
    Graph-based knowledge representation system
    Maintains a network of entities and their relationships
    """
    
    def __init__(self):
        """Initialize the knowledge graph"""
        # Create graph storage directory
        self.storage_dir = os.path.join(parent_dir, "knowledge", "storage")
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Initialize graph
        self.graph = nx.DiGraph()
        
        # Load existing graph if available
        self.graph_file = os.path.join(self.storage_dir, "knowledge_graph.json")
        self._load_graph()
        
        # NLP system for entity extraction
        self.nlp = get_nlp() if NLP_AVAILABLE else None
    
    def add_entity(self, entity: Dict[str, Any], source: str = None) -> str:
        """
        Add an entity to the knowledge graph
        
        Args:
            entity: Entity dictionary with text, type and attributes
            source: Source of this entity information
            
        Returns:
            Entity ID
        """
        # Generate a unique ID if not provided
        entity_id = entity.get("id", str(uuid.uuid4()))
        entity["id"] = entity_id
        
        # Normalize the entity text
        text = entity.get("text", "").strip()
        if not text:
            logger.warning("Attempted to add entity with no text")
            return None
        
        # Add source and timestamp
        if source:
            entity["source"] = source
            
        entity["last_updated"] = datetime.now().isoformat()
        
        # Add/update the entity in the graph
        self.graph.add_node(entity_id, **entity)
        logger.debug(f"Added entity: {text} ({entity.get('type', 'UNKNOWN')})")
        
        # Save the updated graph
        self._save_graph()
        
        return entity_id
    
    def add_relationship(self, source_id: str, target_id: str, relationship_type: str, 
                         confidence: float = 1.0, attributes: Dict[str, Any] = None) -> bool:
        """
        Add a relationship between two entities
        
        Args:
            source_id: ID of the source entity
            target_id: ID of the target entity
            relationship_type: Type of relationship
            confidence: Confidence score (0-1)
            attributes: Additional attributes for this relationship
            
        Returns:
            Success status
        """
        # Check if entities exist
        if not (self.graph.has_node(source_id) and self.graph.has_node(target_id)):
            logger.warning(f"Cannot add relationship - entity not found: {source_id} or {target_id}")
            return False
        
        # Create relationship attributes
        rel_attrs = attributes or {}
        rel_attrs.update({
            "type": relationship_type,
            "confidence": confidence,
            "last_updated": datetime.now().isoformat()
        })
        
        # Add edge to the graph
        self.graph.add_edge(source_id, target_id, **rel_attrs)
        
        # Get entity names for logging
        source_name = self.graph.nodes[source_id].get("text", source_id)
        target_name = self.graph.nodes[target_id].get("text", target_id)
        logger.debug(f"Added relationship: {source_name} --[{relationship_type}]--> {target_name}")
        
        # Save the updated graph
        self._save_graph()
        
        return True
    
    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Get entity by ID
        
        Args:
            entity_id: ID of the entity
            
        Returns:
            Entity dictionary or None if not found
        """
        if not self.graph.has_node(entity_id):
            return None
        
        return dict(self.graph.nodes[entity_id])
    
    def find_entities(self, text: str = None, entity_type: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Find entities by text or type
        
        Args:
            text: Text to search for
            entity_type: Type of entity to find
            limit: Maximum number of results
            
        Returns:
            List of matching entities
        """
        results = []
        
        for node_id, data in self.graph.nodes(data=True):
            match = True
            
            if text and text.lower() not in data.get("text", "").lower():
                match = False
                
            if entity_type and data.get("type") != entity_type:
                match = False
                
            if match:
                results.append(dict(data))
                
            if len(results) >= limit:
                break
                
        return results
    
    def get_relationships(self, entity_id: str, direction: str = "both") -> List[Dict[str, Any]]:
        """
        Get relationships for an entity
        
        Args:
            entity_id: ID of the entity
            direction: 'outgoing', 'incoming', or 'both'
            
        Returns:
            List of relationship dictionaries
        """
        if not self.graph.has_node(entity_id):
            return []
        
        relationships = []
        
        # Get outgoing relationships
        if direction in ["outgoing", "both"]:
            for _, target, data in self.graph.out_edges(entity_id, data=True):
                rel = dict(data)
                rel["source_id"] = entity_id
                rel["target_id"] = target
                rel["direction"] = "outgoing"
                relationships.append(rel)
        
        # Get incoming relationships
        if direction in ["incoming", "both"]:
            for source, _, data in self.graph.in_edges(entity_id, data=True):
                rel = dict(data)
                rel["source_id"] = source
                rel["target_id"] = entity_id
                rel["direction"] = "incoming"
                relationships.append(rel)
        
        return relationships
    
    def add_entities_from_text(self, text: str, source: str = None) -> List[str]:
        """
        Extract and add entities from text
        
        Args:
            text: Text to extract entities from
            source: Source of this information
            
        Returns:
            List of entity IDs
        """
        if not self.nlp:
            logger.warning("NLP system not available for entity extraction")
            return []
        
        entity_ids = []
        
        try:
            # Analyze text to extract entities
            analysis = self.nlp.analyze_text(text, analysis_types=["entities"])
            
            # Add each entity to the graph
            for entity in analysis.get("entities", []):
                entity_id = self.add_entity(entity, source=source)
                if entity_id:
                    entity_ids.append(entity_id)
            
            # Try to find relationships between entities
            self._infer_relationships(entity_ids, text)
            
        except Exception as e:
            logger.error(f"Error extracting entities from text: {e}")
        
        return entity_ids
    
    def add_entities_from_analysis(self, entities: List[Dict[str, Any]], source: str = None) -> List[str]:
        """
        Add entities from an existing analysis
        
        Args:
            entities: List of entity dictionaries
            source: Source of this information
            
        Returns:
            List of entity IDs
        """
        entity_ids = []
        
        for entity in entities:
            entity_id = self.add_entity(entity, source=source)
            if entity_id:
                entity_ids.append(entity_id)
        
        # Try to find relationships between these entities
        if len(entity_ids) > 1:
            self._infer_relationships(entity_ids)
        
        return entity_ids
    
    def get_entity_graph(self, entity_id: str, depth: int = 1) -> Dict[str, Any]:
        """
        Get a subgraph centered on an entity
        
        Args:
            entity_id: ID of the central entity
            depth: How many steps out to include
            
        Returns:
            Dict representation of the subgraph
        """
        if not self.graph.has_node(entity_id):
            return {"nodes": [], "edges": []}
        
        # Get nodes within n steps
        nodes = {entity_id}
        current_nodes = {entity_id}
        
        for _ in range(depth):
            next_nodes = set()
            for node in current_nodes:
                # Add outgoing connections
                next_nodes.update(self.graph.successors(node))
                # Add incoming connections
                next_nodes.update(self.graph.predecessors(node))
            
            nodes.update(next_nodes)
            current_nodes = next_nodes
        
        # Create subgraph
        subgraph = self.graph.subgraph(nodes)
        
        # Convert to dict representation
        result = {
            "nodes": [],
            "edges": []
        }
        
        for node in subgraph.nodes():
            node_data = dict(subgraph.nodes[node])
            node_data["id"] = node  # Ensure ID is included
            result["nodes"].append(node_data)
        
        for u, v, data in subgraph.edges(data=True):
            edge = dict(data)
            edge["source"] = u
            edge["target"] = v
            result["edges"].append(edge)
        
        return result
    
    def search_graph(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """
        Search the knowledge graph
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            Search results
        """
        results = {
            "entities": [],
            "relationships": []
        }
        
        # Find matching entities
        for node, data in self.graph.nodes(data=True):
            node_text = data.get("text", "").lower()
            if query.lower() in node_text:
                results["entities"].append({
                    "id": node,
                    **data
                })
                
            if len(results["entities"]) >= limit:
                break
        
        # Find relationships mentioning the query
        for u, v, data in self.graph.edges(data=True):
            rel_type = data.get("type", "").lower()
            if query.lower() in rel_type:
                results["relationships"].append({
                    "source": u,
                    "source_text": self.graph.nodes[u].get("text", ""),
                    "target": v,
                    "target_text": self.graph.nodes[v].get("text", ""),
                    **data
                })
                
            if len(results["relationships"]) >= limit:
                break
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge graph
        
        Returns:
            Statistics dictionary
        """
        entity_types = {}
        relationship_types = {}
        
        # Count entity types
        for _, data in self.graph.nodes(data=True):
            entity_type = data.get("type", "UNKNOWN")
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        
        # Count relationship types
        for _, _, data in self.graph.edges(data=True):
            rel_type = data.get("type", "UNKNOWN")
            relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
        
        return {
            "entity_count": self.graph.number_of_nodes(),
            "relationship_count": self.graph.number_of_edges(),
            "entity_types": entity_types,
            "relationship_types": relationship_types
        }
    
    def clear(self) -> None:
        """Clear the knowledge graph"""
        self.graph = nx.DiGraph()
        self._save_graph()
        logger.info("Knowledge graph cleared")
    
    def _infer_relationships(self, entity_ids: List[str], context_text: str = None) -> None:
        """
        Try to infer relationships between entities
        
        Args:
            entity_ids: List of entity IDs to check
            context_text: Optional context text for better inference
        """
        if len(entity_ids) < 2:
            return
        
        # Get entity objects
        entities = {eid: self.get_entity(eid) for eid in entity_ids}
        
        # Simple co-occurrence relationship
        for i, id1 in enumerate(entity_ids):
            for id2 in entity_ids[i+1:]:
                # Skip if either entity doesn't exist
                if not (entities[id1] and entities[id2]):
                    continue
                
                # Create co-occurrence relationship
                self.add_relationship(
                    id1, id2, 
                    "MENTIONED_WITH", 
                    confidence=0.7,
                    attributes={"context": context_text[:100] if context_text else "Co-occurrence"}
                )
                
                # Try to infer more specific relationships
                if self.nlp and context_text:
                    try:
                        entity1 = entities[id1].get("text", "")
                        entity2 = entities[id2].get("text", "")
                        
                        # Check for specific patterns or NLP inference here
                        # This would be expanded with more sophisticated inference logic
                    except Exception as e:
                        logger.error(f"Error inferring relationships: {e}")
    
    def _load_graph(self) -> None:
        """Load the knowledge graph from disk"""
        try:
            if os.path.exists(self.graph_file):
                with open(self.graph_file, 'r', encoding='utf-8') as f:
                    graph_data = json.load(f)
                
                # Create a new graph
                self.graph = nx.DiGraph()
                
                # Add nodes
                for node in graph_data.get("nodes", []):
                    node_id = node.pop("id")
                    self.graph.add_node(node_id, **node)
                
                # Add edges
                for edge in graph_data.get("edges", []):
                    source = edge.pop("source")
                    target = edge.pop("target")
                    self.graph.add_edge(source, target, **edge)
                
                logger.info(f"Loaded knowledge graph with {self.graph.number_of_nodes()} entities and {self.graph.number_of_edges()} relationships")
            else:
                logger.info("No existing knowledge graph found, creating a new one")
        except Exception as e:
            logger.error(f"Error loading knowledge graph: {e}")
            self.graph = nx.DiGraph()
    
    def _save_graph(self) -> None:
        """Save the knowledge graph to disk"""
        try:
            # Convert graph to serializable format
            graph_data = {
                "nodes": [],
                "edges": []
            }
            
            # Add nodes
            for node in self.graph.nodes():
                node_data = dict(self.graph.nodes[node])
                node_data["id"] = node  # Ensure ID is included
                graph_data["nodes"].append(node_data)
            
            # Add edges
            for u, v, data in self.graph.edges(data=True):
                edge = dict(data)
                edge["source"] = u
                edge["target"] = v
                graph_data["edges"].append(edge)
            
            # Save to file
            with open(self.graph_file, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, indent=2)
            
            logger.debug("Knowledge graph saved")
        except Exception as e:
            logger.error(f"Error saving knowledge graph: {e}")


# Create singleton instance
_knowledge_graph = None

def initialize() -> KnowledgeGraph:
    """Initialize the knowledge graph singleton"""
    global _knowledge_graph
    if _knowledge_graph is None:
        logger.info("Initializing knowledge graph...")
        try:
            _knowledge_graph = KnowledgeGraph()
            logger.info("Knowledge graph initialized")
        except Exception as e:
            logger.error(f"Error initializing knowledge graph: {e}")
            return None
    return _knowledge_graph

def get_instance() -> Optional[KnowledgeGraph]:
    """Get the knowledge graph singleton instance"""
    global _knowledge_graph
    # Auto-initialize if not already done
    if _knowledge_graph is None:
        return initialize()
    return _knowledge_graph

# Auto-initialize the knowledge graph
try:
    initialize()
except Exception as e:
    logger.error(f"Error during auto-initialization of knowledge graph: {e}")


if __name__ == "__main__":
    # Simple test
    kg = get_instance()
    if kg:
        # Add some test entities
        person = kg.add_entity({"text": "John Doe", "type": "PERSON"}, source="test")
        org = kg.add_entity({"text": "Acme Corp", "type": "ORG"}, source="test")
        
        # Add a relationship
        kg.add_relationship(person, org, "WORKS_FOR", confidence=0.9)
        
        # Print statistics
        print(json.dumps(kg.get_statistics(), indent=2))
