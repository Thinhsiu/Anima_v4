"""
Annotation Tool - Command-line utility for creating custom entity training data
Allows user to annotate text for custom entity recognition training
"""

import os
import sys
import json
import argparse
import logging
import uuid
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import entity trainer
try:
    from nlp.custom_entities import initialize as initialize_entity_trainer
    from nlp.custom_entities import get_instance as get_entity_trainer
    ENTITY_TRAINER_AVAILABLE = True
except ImportError:
    logger.error("Custom entity trainer not available")
    ENTITY_TRAINER_AVAILABLE = False

# Import NLP system
try:
    from nlp.integration import get_instance as get_nlp
    NLP_AVAILABLE = True
except ImportError:
    logger.warning("NLP system not available")
    NLP_AVAILABLE = False


class AnnotationTool:
    """
    Command-line tool for annotating text with custom entity tags
    """
    
    def __init__(self):
        """Initialize the annotation tool"""
        if not ENTITY_TRAINER_AVAILABLE:
            raise ImportError("Entity trainer must be available for annotation tool")
        
        # Initialize entity trainer
        self.entity_trainer = get_entity_trainer()
        if not self.entity_trainer:
            logger.info("Initializing entity trainer")
            self.entity_trainer = initialize_entity_trainer()
        
        if not self.entity_trainer:
            raise ValueError("Failed to initialize entity trainer")
        
        # Get NLP system
        self.nlp = get_nlp() if NLP_AVAILABLE else None
        
        # Load custom entity types
        self.custom_types = self.entity_trainer.get_custom_entity_types()
        
        logger.info(f"Annotation tool initialized with {len(self.custom_types)} custom entity types")
    
    def suggest_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Suggest possible entity annotations
        
        Args:
            text: Input text
            
        Returns:
            List of suggested entities
        """
        suggestions = []
        
        # Use existing custom entities if available
        if self.entity_trainer:
            custom_entities = self.entity_trainer.identify_custom_entities(text)
            for entity in custom_entities:
                suggestions.append({
                    "start": entity["start"],
                    "end": entity["end"],
                    "text": entity["text"],
                    "label": entity["type"],
                    "source": "custom"
                })
        
        # Use spaCy NER for additional suggestions
        if self.nlp:
            nlp_entities = self.nlp.analyze_text(text, analysis_types=["entities"])
            if "entities" in nlp_entities:
                for entity in nlp_entities["entities"]:
                    # Skip duplicates
                    if any(s["start"] == entity["start"] and s["end"] == entity["end"] for s in suggestions):
                        continue
                    
                    suggestions.append({
                        "start": entity["start"],
                        "end": entity["end"],
                        "text": entity["text"],
                        "label": entity["type"],
                        "source": "spacy"
                    })
        
        # Generate entity type suggestions
        if self.entity_trainer:
            additional = self.entity_trainer.suggest_custom_entities(text)
            for entity in additional:
                # Skip duplicates
                if any(s["start"] == entity["start"] and s["end"] == entity["end"] for s in suggestions):
                    continue
                
                suggestions.append({
                    "start": entity["start"],
                    "end": entity["end"],
                    "text": entity["text"],
                    "label": entity["suggested_type"],
                    "source": "suggested"
                })
        
        return suggestions
    
    def highlight_text_with_entities(self, text: str, entities: List[Dict[str, Any]], colored: bool = True) -> str:
        """
        Generate text with highlighted entities
        
        Args:
            text: The text to highlight
            entities: List of entity dictionaries
            colored: Use ANSI color codes
            
        Returns:
            Highlighted text
        """
        # Sort entities by start position in reverse order
        sorted_entities = sorted(entities, key=lambda e: e["start"], reverse=True)
        
        # Create a copy of the text
        highlighted = text
        
        # Add highlights for each entity
        for entity in sorted_entities:
            start = entity["start"]
            end = entity["end"]
            label = entity["label"]
            
            # Skip invalid entities
            if start < 0 or end > len(text) or start >= end:
                continue
            
            # Color codes
            if colored:
                colors = {
                    "custom": "\033[1;32m",  # Bold green
                    "spacy": "\033[1;34m",   # Bold blue
                    "suggested": "\033[1;33m", # Bold yellow
                    "reset": "\033[0m"
                }
                
                source = entity.get("source", "custom")
                color = colors.get(source, colors["custom"])
                reset = colors["reset"]
                
                # Insert highlighting
                entity_text = highlighted[start:end]
                replacement = f"{color}{entity_text} [{label}]{reset}"
                highlighted = highlighted[:start] + replacement + highlighted[end:]
            else:
                # Plain text format
                entity_text = highlighted[start:end]
                replacement = f"[{entity_text}]({label})"
                highlighted = highlighted[:start] + replacement + highlighted[end:]
        
        return highlighted
    
    def annotate_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Interactive annotation of text
        
        Args:
            text: Text to annotate
            
        Returns:
            List of entity annotations
        """
        print("\n=== Text Annotation ===\n")
        print(text)
        print("\n" + "=" * 40)
        
        # Get suggestions
        suggestions = self.suggest_entities(text)
        
        if suggestions:
            print("\nSuggested entities:")
            highlighted = self.highlight_text_with_entities(text, suggestions)
            print(highlighted)
        
        # Custom entity types
        print("\nAvailable entity types:")
        for i, entity_type in enumerate(sorted(self.custom_types)):
            print(f"{i+1}. {entity_type}")
        
        # Collect annotations
        annotations = []
        print("\nAnnotate entities (enter 'done' when finished):")
        
        while True:
            # Get entity text
            entity_text = input("\nEnter entity text (or 'done', 'add', 'list'): ").strip()
            
            if entity_text.lower() == 'done':
                break
            
            elif entity_text.lower() == 'add':
                # Add new entity type
                new_type = input("Enter new entity type (UPPERCASE): ").strip().upper()
                if new_type and self.entity_trainer.add_custom_entity_type(new_type):
                    self.custom_types = self.entity_trainer.get_custom_entity_types()
                    print(f"Added new entity type: {new_type}")
                continue
            
            elif entity_text.lower() == 'list':
                # List entity types
                print("\nAvailable entity types:")
                for i, entity_type in enumerate(sorted(self.custom_types)):
                    print(f"{i+1}. {entity_type}")
                continue
            
            # Find entity in text
            start = text.find(entity_text)
            if start < 0:
                print(f"Entity text not found: '{entity_text}'")
                continue
            
            # Get entity type
            type_input = input(f"Entity type for '{entity_text}' (number or name): ").strip()
            
            try:
                # Check if input is a number
                type_idx = int(type_input) - 1
                if 0 <= type_idx < len(self.custom_types):
                    entity_type = sorted(self.custom_types)[type_idx]
                else:
                    print("Invalid type number")
                    continue
            except ValueError:
                # Input is a name
                entity_type = type_input.upper()
                if entity_type not in self.custom_types:
                    add_type = input(f"Type {entity_type} not found. Add it? (y/n): ").strip().lower()
                    if add_type == 'y':
                        if self.entity_trainer.add_custom_entity_type(entity_type):
                            self.custom_types = self.entity_trainer.get_custom_entity_types()
                            print(f"Added new entity type: {entity_type}")
                        else:
                            print("Failed to add entity type")
                            continue
                    else:
                        continue
            
            # Add annotation
            end = start + len(entity_text)
            annotation = {
                "start": start,
                "end": end,
                "label": entity_type
            }
            annotations.append(annotation)
            
            # Show updated annotations
            print("\nCurrent annotations:")
            highlighted = self.highlight_text_with_entities(text, [
                {"start": a["start"], "end": a["end"], "label": a["label"], "source": "custom"}
                for a in annotations
            ])
            print(highlighted)
        
        return annotations
    
    def annotate_file(self, file_path: str) -> bool:
        """
        Annotate text from a file
        
        Args:
            file_path: Path to the input file
            
        Returns:
            Success status
        """
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return False
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            # Check if text is too long
            if len(text) > 10000:
                print(f"Text is too long ({len(text)} chars). Please use a shorter text.")
                return False
            
            # Annotate text
            annotations = self.annotate_text(text)
            
            if not annotations:
                print("No annotations created")
                return False
            
            # Add training example
            example_id = self.entity_trainer.add_training_example(text, annotations)
            
            print(f"\nAdded training example with {len(annotations)} annotations")
            print(f"Example ID: {example_id}")
            
            # Ask if user wants to train the model
            train_model = input("\nTrain the model now? (y/n): ").strip().lower()
            if train_model == 'y':
                print("Training model...")
                success = self.entity_trainer.train()
                if success:
                    print("Model trained successfully")
                else:
                    print("Model training may not have been successful")
            
            return True
            
        except Exception as e:
            print(f"Error annotating file: {e}")
            logger.error(f"Error annotating file: {e}")
            return False
    
    def annotate_text_input(self) -> bool:
        """
        Annotate user-provided text input
        
        Returns:
            Success status
        """
        print("Enter text to annotate (Ctrl+D or Ctrl+Z+Enter to finish):")
        
        try:
            lines = []
            while True:
                try:
                    line = input()
                    lines.append(line)
                except EOFError:
                    break
            
            text = "\n".join(lines).strip()
            if not text:
                print("No text provided")
                return False
            
            # Annotate text
            annotations = self.annotate_text(text)
            
            if not annotations:
                print("No annotations created")
                return False
            
            # Add training example
            example_id = self.entity_trainer.add_training_example(text, annotations)
            
            print(f"\nAdded training example with {len(annotations)} annotations")
            print(f"Example ID: {example_id}")
            
            # Ask if user wants to train the model
            train_model = input("\nTrain the model now? (y/n): ").strip().lower()
            if train_model == 'y':
                print("Training model...")
                success = self.entity_trainer.train()
                if success:
                    print("Model trained successfully")
                else:
                    print("Model training may not have been successful")
            
            return True
            
        except Exception as e:
            print(f"Error processing text input: {e}")
            logger.error(f"Error processing text input: {e}")
            return False
    
    def list_examples(self) -> None:
        """List training examples"""
        examples = self.entity_trainer.training_examples
        
        if not examples:
            print("No training examples found")
            return
        
        print(f"\n=== Training Examples ({len(examples)}) ===\n")
        
        for i, example in enumerate(examples):
            # Get creation date
            created = example.get("added", "Unknown")
            if created != "Unknown":
                try:
                    created = datetime.fromisoformat(created).strftime("%Y-%m-%d %H:%M")
                except:
                    pass
            
            # Get entity count
            entity_count = len(example.get("entities", []))
            
            # Get snippet of text
            text = example.get("text", "")
            if len(text) > 60:
                text = text[:57] + "..."
            
            # Print summary
            print(f"{i+1}. [{created}] {text}")
            print(f"   ID: {example.get('id', 'Unknown')} | Entities: {entity_count}")
        
        # Allow viewing full examples
        while True:
            view_input = input("\nEnter example number to view details (or 'q' to quit): ").strip()
            
            if view_input.lower() == 'q':
                break
            
            try:
                idx = int(view_input) - 1
                if 0 <= idx < len(examples):
                    example = examples[idx]
                    
                    print("\n=== Example Details ===\n")
                    print(f"ID: {example.get('id', 'Unknown')}")
                    print(f"Added: {example.get('added', 'Unknown')}")
                    print(f"Entities: {len(example.get('entities', []))}")
                    
                    # Show text with highlights
                    print("\nAnnotated Text:")
                    highlighted = self.highlight_text_with_entities(
                        example.get("text", ""),
                        [{"start": e["start"], "end": e["end"], "label": e["label"], "source": "custom"}
                         for e in example.get("entities", [])]
                    )
                    print(highlighted)
                else:
                    print("Invalid example number")
            except ValueError:
                print("Invalid input")
    
    def train_model(self) -> None:
        """Train the custom entity model"""
        examples = self.entity_trainer.training_examples
        
        if not examples:
            print("No training examples found")
            return
        
        print(f"\nTraining model with {len(examples)} examples...")
        
        # Get training iterations
        iterations = 10
        try:
            iterations_input = input("Enter number of training iterations (default: 10): ").strip()
            if iterations_input:
                iterations = int(iterations_input)
        except ValueError:
            print(f"Invalid input, using default {iterations} iterations")
        
        # Train the model
        success = self.entity_trainer.train(iterations=iterations)
        
        if success:
            print("\nModel trained successfully")
        else:
            print("\nModel training may not have been successful")


def main():
    """Main function for the annotation tool"""
    parser = argparse.ArgumentParser(description="Custom Entity Annotation Tool")
    parser.add_argument("--file", "-f", help="Input file to annotate")
    parser.add_argument("--list", "-l", action="store_true", help="List training examples")
    parser.add_argument("--train", "-t", action="store_true", help="Train the model")
    
    args = parser.parse_args()
    
    # Check if entity trainer is available
    if not ENTITY_TRAINER_AVAILABLE:
        print("Error: Custom entity trainer not available")
        return 1
    
    try:
        # Initialize the annotation tool
        tool = AnnotationTool()
        
        if args.list:
            # List training examples
            tool.list_examples()
        elif args.train:
            # Train the model
            tool.train_model()
        elif args.file:
            # Annotate file
            tool.annotate_file(args.file)
        else:
            # Interactive text annotation
            tool.annotate_text_input()
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        logger.error(f"Error in annotation tool: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
