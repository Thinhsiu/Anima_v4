"""
Multimodal Core - Integrates vision and NLP capabilities
This module enables Anima to understand both visual and textual information together
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import vision system
from vision.vision_processor import analyze_image, get_image_description
from vision_integration import vision_integration

# Import NLP system
from nlp.integration import get_instance as get_nlp


class MultimodalProcessor:
    """Core processor for multimodal understanding (text + vision)"""
    
    def __init__(self):
        """Initialize the multimodal processor"""
        self.nlp = get_nlp()  # Get the NLP singleton instance
        if not self.nlp:
            logger.warning("NLP system not available for multimodal integration")
        
        # Create a directory for temporary outputs
        self.output_dir = os.path.join(parent_dir, "multimodal", "output")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def process_image_with_text(self, image_path: str, text_context: str) -> Dict[str, Any]:
        """
        Process an image with accompanying text to create a unified understanding
        
        Args:
            image_path: Path to the image
            text_context: Accompanying text context
            
        Returns:
            Dict containing combined analysis
        """
        results = {
            "image_path": image_path,
            "text_context": text_context,
            "timestamp": self._get_timestamp(),
            "visual_analysis": {},
            "textual_analysis": {},
            "combined_analysis": {}
        }
        
        # Step 1: Get image description and analysis
        try:
            image_analysis = analyze_image(image_path)
            results["visual_analysis"] = image_analysis
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            results["visual_analysis"] = {"error": str(e)}
        
        # Step 2: Analyze text with NLP if available
        if self.nlp:
            try:
                text_analysis = self.nlp.analyze_text(text_context)
                results["textual_analysis"] = text_analysis
            except Exception as e:
                logger.error(f"Error analyzing text: {e}")
                results["textual_analysis"] = {"error": str(e)}
        
        # Step 3: Combine analyses
        self._combine_analyses(results)
        
        return results
    
    def enhance_prompt_with_vision(self, image_path: str, base_prompt: str) -> str:
        """
        Enhance a text prompt with visual information from an image
        
        Args:
            image_path: Path to the image
            base_prompt: Original text prompt
            
        Returns:
            Enhanced prompt combining visual and textual elements
        """
        # Get a brief description of the image
        try:
            image_desc = get_image_description(image_path, brief=True)
        except Exception as e:
            logger.error(f"Error getting image description: {e}")
            return base_prompt
        
        # Extract entities from the image description
        entities = []
        if self.nlp:
            try:
                analysis = self.nlp.analyze_text(image_desc)
                entities = [e["text"] for e in analysis.get("entities", [])]
            except Exception as e:
                logger.error(f"Error extracting entities from image description: {e}")
        
        # Enhance the prompt with image information
        enhanced_prompt = f"{base_prompt}\n\nImage context: {image_desc}"
        if entities:
            enhanced_prompt += f"\nKey elements in image: {', '.join(entities)}"
        
        return enhanced_prompt
    
    def extract_visual_memory_elements(self, image_path: str, text_context: str = None) -> Dict[str, Any]:
        """
        Extract memory-worthy elements from an image and optional text
        
        Args:
            image_path: Path to the image
            text_context: Optional accompanying text
            
        Returns:
            Dict of memory elements
        """
        memory_elements = {
            "image_path": image_path,
            "entities": [],
            "summary": "",
            "keywords": []
        }
        
        # Get image analysis
        try:
            # Get a detailed description of the image
            image_desc = get_image_description(image_path)
            memory_elements["summary"] = image_desc
            
            # Use NLP to extract entities from the description
            if self.nlp:
                analysis = self.nlp.analyze_text(image_desc)
                memory_elements["entities"] = [
                    ent for ent in analysis.get("entities", [])
                    if ent["type"] in ["PERSON", "ORG", "GPE", "DATE", "EVENT", "PRODUCT"]
                ]
                memory_elements["keywords"] = analysis.get("keywords", [])
        except Exception as e:
            logger.error(f"Error extracting visual memory elements: {e}")
        
        # Add text context if available
        if text_context and self.nlp:
            try:
                text_elements = self.nlp.extract_memory_elements(text_context)
                
                # Merge entities, avoiding duplicates
                existing_texts = {e["text"].lower() for e in memory_elements["entities"]}
                for ent in text_elements.get("entities", []):
                    if ent["text"].lower() not in existing_texts:
                        memory_elements["entities"].append(ent)
                        existing_texts.add(ent["text"].lower())
                
                # Merge keywords
                memory_elements["keywords"].extend(text_elements.get("keywords", []))
                memory_elements["keywords"] = list(set(memory_elements["keywords"]))
            except Exception as e:
                logger.error(f"Error processing text context: {e}")
        
        return memory_elements
    
    def generate_image_caption(self, image_path: str) -> str:
        """
        Generate a descriptive caption for an image
        
        Args:
            image_path: Path to the image
            
        Returns:
            Descriptive caption
        """
        try:
            return get_image_description(image_path, brief=True)
        except Exception as e:
            logger.error(f"Error generating image caption: {e}")
            return "Unable to generate caption for this image."
    
    def _combine_analyses(self, results: Dict[str, Any]) -> None:
        """
        Combine visual and textual analyses into a unified understanding
        
        Args:
            results: Dict containing visual_analysis and textual_analysis
        """
        combined = {
            "entities": [],
            "keywords": [],
            "sentiment": "neutral",
            "summary": ""
        }
        
        # Extract entities from visual analysis
        visual_desc = results.get("visual_analysis", {}).get("analysis", "")
        if visual_desc and self.nlp:
            try:
                visual_nlp = self.nlp.analyze_text(visual_desc)
                combined["entities"].extend(visual_nlp.get("entities", []))
                
                # Extract keywords from visual description
                visual_keywords = visual_nlp.get("keywords", [])
                combined["keywords"].extend(visual_keywords)
            except Exception as e:
                logger.error(f"Error analyzing visual description: {e}")
        
        # Add entities from textual analysis
        text_entities = results.get("textual_analysis", {}).get("entities", [])
        
        # Merge entities, avoiding duplicates
        existing_texts = {e["text"].lower() for e in combined["entities"]}
        for entity in text_entities:
            if entity["text"].lower() not in existing_texts:
                combined["entities"].append(entity)
                existing_texts.add(entity["text"].lower())
        
        # Add keywords from textual analysis
        text_keywords = results.get("textual_analysis", {}).get("keywords", [])
        combined["keywords"].extend(text_keywords)
        combined["keywords"] = list(set(combined["keywords"]))
        
        # Set sentiment from text analysis (prioritize text sentiment over image)
        sentiment = results.get("textual_analysis", {}).get("sentiment", {}).get("overall_sentiment", "neutral")
        combined["sentiment"] = sentiment
        
        # Generate a combined summary
        visual_summary = visual_desc[:100] + "..." if len(visual_desc) > 100 else visual_desc
        text_summary = results.get("text_context", "")[:100] + "..." if len(results.get("text_context", "")) > 100 else results.get("text_context", "")
        
        if visual_summary and text_summary:
            combined["summary"] = f"Visual: {visual_summary}\nText: {text_summary}"
        elif visual_summary:
            combined["summary"] = f"Visual: {visual_summary}"
        else:
            combined["summary"] = f"Text: {text_summary}"
        
        # Store the combined analysis
        results["combined_analysis"] = combined
    
    def _get_timestamp(self) -> str:
        """Get the current timestamp in ISO format"""
        from datetime import datetime
        return datetime.now().isoformat()


# Create singleton instance
_multimodal_processor = None

def initialize() -> MultimodalProcessor:
    """Initialize the multimodal processor singleton"""
    global _multimodal_processor
    if _multimodal_processor is None:
        logger.info("Initializing multimodal processor...")
        _multimodal_processor = MultimodalProcessor()
        logger.info("Multimodal processor initialized")
    return _multimodal_processor

def get_instance() -> Optional[MultimodalProcessor]:
    """Get the multimodal processor singleton instance"""
    global _multimodal_processor
    # Auto-initialize if not already done
    if _multimodal_processor is None:
        return initialize()
    return _multimodal_processor

# Auto-initialize the multimodal processor
try:
    initialize()
except Exception as e:
    logger.error(f"Error initializing multimodal processor: {e}")
