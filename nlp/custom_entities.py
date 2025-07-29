"""
Custom Entity Training - Allows training of custom entity types specific to the user's domain
Provides incremental learning capabilities for the NLP system
"""

import os
import sys
import logging
import json
import uuid
import random
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from pathlib import Path
from datetime import datetime
import re
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import spaCy conditionally
try:
    import spacy
    from spacy.tokens import DocBin
    from spacy.training import Example
    from spacy.util import minibatch, compounding
    SPACY_AVAILABLE = True
except ImportError:
    logger.warning("spaCy not available for custom entity training")
    SPACY_AVAILABLE = False


class EntityTrainer:
    """
    Trains custom entity types for the NLP system
    Provides incremental learning from conversation
    """
    
    def __init__(self):
        """Initialize the entity trainer"""
        if not SPACY_AVAILABLE:
            raise ImportError("spaCy must be available for entity training")
        
        # Create directories for training data and models
        self.training_dir = os.path.join(parent_dir, "nlp", "custom_entities")
        self.models_dir = os.path.join(self.training_dir, "models")
        self.datasets_dir = os.path.join(self.training_dir, "datasets")
        
        for directory in [self.training_dir, self.models_dir, self.datasets_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Load base model (medium by default)
        try:
            self.base_model = "en_core_web_md"
            self.nlp = spacy.load(self.base_model)
            logger.info(f"Loaded base model: {self.base_model}")
        except Exception as e:
            logger.error(f"Error loading base model: {e}")
            raise
        
        # Custom entity types
        self.custom_types = set()
        self._load_custom_types()
        
        # Load custom model if available
        self.custom_model_path = os.path.join(self.models_dir, "custom_ner_model")
        if os.path.exists(self.custom_model_path):
            try:
                self.nlp = spacy.load(self.custom_model_path)
                logger.info("Loaded custom NER model")
            except Exception as e:
                logger.error(f"Error loading custom model: {e}")
        
        # Training examples
        self.training_examples = []
        self.load_training_examples()
        
        logger.info("Entity trainer initialized")
    
    def add_custom_entity_type(self, entity_type: str, description: str = None) -> bool:
        """
        Add a new custom entity type
        
        Args:
            entity_type: Entity type name (should be uppercase)
            description: Optional description
            
        Returns:
            Success status
        """
        # Validate entity type
        entity_type = entity_type.upper()
        if not re.match(r'^[A-Z_]+$', entity_type):
            logger.warning(f"Invalid entity type: {entity_type}. Use uppercase letters and underscores only.")
            return False
        
        # Add to custom types
        self.custom_types.add(entity_type)
        
        # Save custom type info
        self._save_custom_types()
        
        logger.info(f"Added custom entity type: {entity_type}")
        return True
    
    def add_training_example(self, text: str, entities: List[Dict[str, Any]]) -> str:
        """
        Add a training example
        
        Args:
            text: The text containing entities
            entities: List of entity dicts with start, end, label
            
        Returns:
            ID of the added example
        """
        # Validate entities
        valid_entities = []
        for entity in entities:
            if "start" not in entity or "end" not in entity or "label" not in entity:
                logger.warning(f"Invalid entity format: {entity}")
                continue
            
            # Check that the entity is within text bounds
            if entity["start"] < 0 or entity["end"] > len(text) or entity["start"] >= entity["end"]:
                logger.warning(f"Entity boundaries out of range: {entity}")
                continue
            
            # Add to valid entities
            valid_entities.append(entity)
        
        # Create example ID
        example_id = str(uuid.uuid4())
        
        # Create example object
        example = {
            "id": example_id,
            "text": text,
            "entities": valid_entities,
            "added": datetime.now().isoformat()
        }
        
        # Add to training examples
        self.training_examples.append(example)
        
        # Save training examples
        self._save_training_examples()
        
        logger.info(f"Added training example with {len(valid_entities)} entities")
        return example_id
    
    def extract_training_examples_from_conversation(self, user_input: str, ai_response: str) -> List[str]:
        """
        Extract potential training examples from conversation
        
        Args:
            user_input: User's message
            ai_response: AI's response
            
        Returns:
            List of extracted example IDs
        """
        extracted_ids = []
        
        # Check for explicit annotation requests
        annotation_pattern = r'remember\s+(?:that\s+)?([A-Za-z\s]+)\s+(?:is|are)\s+(?:a|an)\s+([A-Z_]+)'
        matches = re.findall(annotation_pattern, user_input, re.IGNORECASE)
        
        for match in matches:
            entity_text, entity_type = match
            entity_text = entity_text.strip()
            entity_type = entity_type.upper()
            
            # Check if this is a known entity type
            if entity_type not in self.custom_types:
                # Add as new type
                self.add_custom_entity_type(entity_type)
            
            # Find the entity in the text
            start = user_input.lower().find(entity_text.lower())
            if start >= 0:
                end = start + len(entity_text)
                
                # Add as training example
                entities = [{"start": start, "end": end, "label": entity_type}]
                example_id = self.add_training_example(user_input, entities)
                extracted_ids.append(example_id)
        
        # Look for confirmation of entity types in AI response
        confirmation_pattern = r'I\'ll remember that ([A-Za-z\s]+) (?:is|are) (?:a|an) ([A-Z_]+)'
        matches = re.findall(confirmation_pattern, ai_response)
        
        for match in matches:
            entity_text, entity_type = match
            entity_text = entity_text.strip()
            entity_type = entity_type.upper()
            
            # Check if already added from user input
            if extracted_ids:
                continue
                
            # Check if this is a known entity type
            if entity_type not in self.custom_types:
                # Add as new type
                self.add_custom_entity_type(entity_type)
            
            # Find the entity in the user input
            start = user_input.lower().find(entity_text.lower())
            if start >= 0:
                end = start + len(entity_text)
                
                # Add as training example
                entities = [{"start": start, "end": end, "label": entity_type}]
                example_id = self.add_training_example(user_input, entities)
                extracted_ids.append(example_id)
        
        return extracted_ids
    
    def train(self, iterations: int = 10) -> bool:
        """
        Train the custom NER model
        
        Args:
            iterations: Number of training iterations
            
        Returns:
            Success status
        """
        if not self.training_examples:
            logger.warning("No training examples available")
            return False
        
        try:
            # Prepare training data
            train_data = self._prepare_training_data()
            if not train_data:
                logger.warning("No valid training data could be prepared")
                return False
            
            # Create a new pipeline
            logger.info("Creating NER training pipeline")
            if "ner" not in self.nlp.pipe_names:
                ner = self.nlp.add_pipe("ner")
            else:
                ner = self.nlp.get_pipe("ner")
            
            # Add custom entity labels
            for entity_type in self.custom_types:
                ner.add_label(entity_type)
            
            # Train the model
            logger.info(f"Starting training with {len(train_data)} examples")
            
            # Get the other pipeline components
            other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != "ner"]
            
            with self.nlp.disable_pipes(*other_pipes):
                optimizer = self.nlp.create_optimizer()
                
                # Training loop
                for i in range(iterations):
                    random.shuffle(train_data)
                    losses = {}
                    
                    # Batch the examples
                    batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
                    
                    for batch in batches:
                        self.nlp.update(
                            batch,
                            drop=0.5,
                            losses=losses,
                        )
                    
                    logger.info(f"Iteration {i+1}/{iterations}: Loss: {losses.get('ner', 0):.4f}")
            
            # Save the model
            self.nlp.to_disk(self.custom_model_path)
            logger.info(f"Saved custom model to {self.custom_model_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training custom model: {e}")
            return False
    
    def identify_custom_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Identify custom entities in a text
        
        Args:
            text: Text to analyze
            
        Returns:
            List of entity dictionaries
        """
        if not text.strip():
            return []
            
        # Process with current model
        doc = self.nlp(text)
        
        # Extract entities
        entities = []
        for ent in doc.ents:
            # Only include custom types
            if ent.label_ in self.custom_types:
                entities.append({
                    "text": ent.text,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "type": ent.label_
                })
        
        return entities
    
    def suggest_custom_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Suggest potential custom entities in text
        
        Args:
            text: Text to analyze
            
        Returns:
            List of potential entity suggestions
        """
        if not text.strip():
            return []
        
        # Process with current model
        doc = self.nlp(text)
        
        # Extract noun phrases as potential entities
        suggestions = []
        for chunk in doc.noun_chunks:
            # Skip very short chunks
            if len(chunk.text) < 3:
                continue
                
            # Skip if already an entity
            if any(chunk.start_char >= ent.start_char and chunk.end_char <= ent.end_char for ent in doc.ents):
                continue
            
            # Suggest entity type based on characteristics
            suggested_type = self._suggest_entity_type(chunk)
            
            suggestions.append({
                "text": chunk.text,
                "start": chunk.start_char,
                "end": chunk.end_char,
                "suggested_type": suggested_type,
                "confidence": 0.7  # Placeholder confidence
            })
        
        return suggestions
    
    def load_training_examples(self) -> None:
        """Load training examples from disk"""
        examples_file = os.path.join(self.datasets_dir, "training_examples.json")
        
        if os.path.exists(examples_file):
            try:
                with open(examples_file, 'r', encoding='utf-8') as f:
                    self.training_examples = json.load(f)
                logger.info(f"Loaded {len(self.training_examples)} training examples")
            except Exception as e:
                logger.error(f"Error loading training examples: {e}")
                self.training_examples = []
        else:
            logger.info("No training examples file found")
            self.training_examples = []
    
    def export_training_data(self, output_file: str = None) -> str:
        """
        Export training data to a file
        
        Args:
            output_file: Optional output file path
            
        Returns:
            Path to the exported file
        """
        if not self.training_examples:
            return None
            
        # Default output file
        if not output_file:
            output_file = os.path.join(self.datasets_dir, f"training_data_{datetime.now().strftime('%Y%m%d')}.json")
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.training_examples, f, indent=2)
            
            logger.info(f"Exported {len(self.training_examples)} training examples to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error exporting training data: {e}")
            return None
    
    def import_training_data(self, input_file: str) -> int:
        """
        Import training data from a file
        
        Args:
            input_file: Path to the input file
            
        Returns:
            Number of examples imported
        """
        if not os.path.exists(input_file):
            logger.warning(f"Input file not found: {input_file}")
            return 0
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                imported_examples = json.load(f)
            
            # Validate examples
            valid_examples = []
            for example in imported_examples:
                if "text" in example and "entities" in example:
                    # Generate new ID
                    example["id"] = str(uuid.uuid4())
                    valid_examples.append(example)
            
            # Add to existing examples
            self.training_examples.extend(valid_examples)
            
            # Save training examples
            self._save_training_examples()
            
            logger.info(f"Imported {len(valid_examples)} training examples")
            return len(valid_examples)
            
        except Exception as e:
            logger.error(f"Error importing training data: {e}")
            return 0
    
    def get_custom_entity_types(self) -> List[str]:
        """
        Get list of custom entity types
        
        Returns:
            List of custom entity types
        """
        return sorted(list(self.custom_types))
    
    def _prepare_training_data(self) -> List[Example]:
        """
        Prepare training data for spaCy
        
        Returns:
            List of spaCy training examples
        """
        train_data = []
        
        for example in self.training_examples:
            text = example.get("text", "")
            entities = example.get("entities", [])
            
            if not text or not entities:
                continue
            
            # Convert to spaCy's format
            ents = []
            for entity in entities:
                # Skip invalid entities
                if not all(k in entity for k in ["start", "end", "label"]):
                    continue
                    
                ents.append((entity["start"], entity["end"], entity["label"]))
            
            # Skip examples with no valid entities
            if not ents:
                continue
                
            # Create the Example object
            doc = self.nlp.make_doc(text)
            example = Example.from_dict(doc, {"entities": ents})
            train_data.append(example)
        
        return train_data
    
    def _save_training_examples(self) -> None:
        """Save training examples to disk"""
        examples_file = os.path.join(self.datasets_dir, "training_examples.json")
        
        try:
            with open(examples_file, 'w', encoding='utf-8') as f:
                json.dump(self.training_examples, f, indent=2)
            logger.debug(f"Saved {len(self.training_examples)} training examples")
        except Exception as e:
            logger.error(f"Error saving training examples: {e}")
    
    def _load_custom_types(self) -> None:
        """Load custom entity types from disk"""
        types_file = os.path.join(self.training_dir, "custom_types.json")
        
        if os.path.exists(types_file):
            try:
                with open(types_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.custom_types = set(data.get("types", []))
                logger.info(f"Loaded {len(self.custom_types)} custom entity types")
            except Exception as e:
                logger.error(f"Error loading custom types: {e}")
    
    def _save_custom_types(self) -> None:
        """Save custom entity types to disk"""
        types_file = os.path.join(self.training_dir, "custom_types.json")
        
        try:
            with open(types_file, 'w', encoding='utf-8') as f:
                json.dump({"types": list(self.custom_types)}, f, indent=2)
            logger.debug(f"Saved {len(self.custom_types)} custom entity types")
        except Exception as e:
            logger.error(f"Error saving custom types: {e}")
    
    def _suggest_entity_type(self, chunk) -> str:
        """
        Suggest entity type for a chunk
        
        Args:
            chunk: spaCy chunk
            
        Returns:
            Suggested entity type
        """
        text = chunk.text.lower()
        
        # Try to infer type from context
        if any(word in text for word in ["system", "app", "software", "platform"]):
            return "SOFTWARE"
        elif any(word in text for word in ["company", "corporation", "inc", "llc"]):
            return "ORGANIZATION"
        elif any(word in text for word in ["project", "task", "initiative"]):
            return "PROJECT"
        elif any(word in text for word in ["feature", "functionality", "capability"]):
            return "FEATURE"
        else:
            return "CUSTOM_ENTITY"


# Create singleton instance
_entity_trainer = None

def initialize() -> Optional[EntityTrainer]:
    """Initialize the entity trainer singleton"""
    global _entity_trainer
    if _entity_trainer is None:
        logger.info("Initializing entity trainer...")
        try:
            if SPACY_AVAILABLE:
                _entity_trainer = EntityTrainer()
                logger.info("Entity trainer initialized")
            else:
                logger.warning("spaCy not available, entity trainer not initialized")
                return None
        except Exception as e:
            logger.error(f"Error initializing entity trainer: {e}")
            return None
    return _entity_trainer

def get_instance() -> Optional[EntityTrainer]:
    """Get the entity trainer singleton instance"""
    global _entity_trainer
    # Don't auto-initialize as this is a heavy component
    return _entity_trainer

# Don't auto-initialize this module as it's resource intensive

if __name__ == "__main__":
    # Simple test
    trainer = initialize()
    if trainer:
        # Add a custom entity type
        trainer.add_custom_entity_type("PRODUCT")
        
        # Add a training example
        text = "The new MacBook Pro is really impressive."
        entities = [{"start": 4, "end": 15, "label": "PRODUCT"}]
        trainer.add_training_example(text, entities)
        
        # Print the available entity types
        print("Custom entity types:", trainer.get_custom_entity_types())
        
        # Train the model
        print("Training model...")
        trainer.train(iterations=5)
        
        # Test the model
        test_text = "I really love my MacBook Air and my iPhone."
        entities = trainer.identify_custom_entities(test_text)
        print(f"Entities in '{test_text}':", entities)
        
        # Suggest entities
        suggestions = trainer.suggest_custom_entities("The Tesla Model 3 is a great electric car.")
        print("Entity suggestions:", suggestions)
