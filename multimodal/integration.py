"""
Multimodal Integration - Connects multimodal understanding to Anima's core
"""

import os
import sys
import logging
from typing import Dict, List, Any, Optional, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import multimodal core
from multimodal.multimodal_core import get_instance as get_multimodal
from vision_integration import vision_integration

# Import knowledge graph
try:
    from knowledge.graph import get_instance as get_knowledge_graph
    KNOWLEDGE_GRAPH_AVAILABLE = True
except ImportError:
    logger.warning("Knowledge graph module not available")
    KNOWLEDGE_GRAPH_AVAILABLE = False

def integrate_multimodal_with_anima(user_input: str, full_prompt: str, ai_response: str = None) -> Tuple[str, str, Optional[str]]:
    """
    Integrate multimodal processing into Anima's conversation flow
    
    Args:
        user_input: Original user input
        full_prompt: Full prompt being sent to the LLM
        ai_response: Anima's response (for context tracking)
        
    Returns:
        tuple: (modified_input, modified_prompt, response_override)
    """
    # Default return values
    modified_input = user_input
    modified_prompt = full_prompt
    response_override = None
    
    # Check for multimodal commands
    if any(cmd in user_input.lower() for cmd in ["analyze image and text", "multimodal", "combined analysis"]):
        # First, let the vision integration handle image extraction/processing
        modified_input, vision_context, vision_override = vision_integration.process_message(user_input)
        
        # If vision found an image but didn't fully override the response
        if vision_context and not vision_override:
            # Get the multimodal processor
            mm = get_multimodal()
            if not mm:
                logger.warning("Multimodal processor not available")
                return modified_input, modified_prompt, None
            
            # Extract image path from vision context
            image_path = vision_context.get("image_path")
            if image_path:
                # Process image with text context
                results = mm.process_image_with_text(image_path, modified_input)
                
                # Update the prompt with multimodal insights
                enhanced_prompt = mm.enhance_prompt_with_vision(image_path, full_prompt)
                
                # Add combined analysis to the knowledge graph if available
                if KNOWLEDGE_GRAPH_AVAILABLE:
                    kg = get_knowledge_graph()
                    if kg and 'combined_analysis' in results:
                        entities = results['combined_analysis'].get('entities', [])
                        kg.add_entities_from_analysis(entities, source=f"multimodal-{os.path.basename(image_path)}")
                
                # Return enhanced prompt
                return modified_input, enhanced_prompt, None
        
        # If vision already handled everything, just pass through its results
        return modified_input, full_prompt, vision_override
    
    # Check for multimodal help command
    if "multimodal help" in user_input.lower():
        help_text = get_help()
        return user_input, full_prompt, help_text
    
    return modified_input, modified_prompt, response_override

def extract_multimodal_memory(user_input: str, ai_response: str) -> Optional[Dict[str, Any]]:
    """
    Extract memory elements from multimodal interaction
    
    Args:
        user_input: User's message
        ai_response: Anima's response
        
    Returns:
        Optional memory elements dict
    """
    # First check if there's an image in the context via vision system
    _, vision_context, _ = vision_integration.process_message(user_input)
    
    if not vision_context or 'image_path' not in vision_context:
        return None  # No image in the context
    
    image_path = vision_context.get('image_path')
    
    # Get multimodal processor
    mm = get_multimodal()
    if not mm:
        logger.warning("Multimodal processor not available for memory extraction")
        return None
    
    # Extract memory elements using both image and text
    memory_elements = mm.extract_visual_memory_elements(image_path, user_input)
    
    # Add response analysis if available
    if ai_response:
        memory_elements['response_analysis'] = mm.nlp.analyze_text(ai_response) if mm.nlp else {}
    
    return memory_elements

def process_image_text_query(image_path: str, query: str) -> str:
    """
    Process a specific query about an image
    
    Args:
        image_path: Path to the image
        query: Specific query about the image
        
    Returns:
        Analysis result
    """
    # Get the multimodal processor
    mm = get_multimodal()
    if not mm:
        return "Multimodal processing is not available."
    
    # Process image with specific query
    try:
        results = mm.process_image_with_text(image_path, query)
        
        # Return the combined analysis as a readable string
        combined = results.get('combined_analysis', {})
        entities = combined.get('entities', [])
        keywords = combined.get('keywords', [])
        sentiment = combined.get('sentiment', 'neutral')
        
        response = [
            "# Image & Text Analysis Results",
            f"\n## Query: {query}",
            f"\n## Summary:\n{combined.get('summary', 'No summary available.')}",
            f"\n## Sentiment: {sentiment}",
            "\n## Key Entities:"
        ]
        
        if entities:
            for ent in entities[:10]:  # Limit to top 10 entities
                response.append(f"- {ent['text']} ({ent['type']})")
        else:
            response.append("- No significant entities found")
            
        response.append("\n## Keywords:")
        if keywords:
            response.append(", ".join(keywords[:15]))  # Limit to top 15 keywords
        else:
            response.append("No significant keywords found")
        
        return "\n".join(response)
    except Exception as e:
        logger.error(f"Error in multimodal processing: {e}")
        return f"An error occurred during multimodal processing: {e}"

def get_help() -> str:
    """Get help text for multimodal capabilities"""
    return """
# Multimodal Understanding System

Anima can now analyze images and text together to gain a deeper understanding:

## Commands
- `analyze image and text` - Analyze both an image and text together
- `multimodal help` - Display this help message

## Features
- **Visual + Textual Analysis**: Understand images in context of your messages
- **Enhanced Memory**: Remember key elements from both visual and textual information
- **Cross-Modal Understanding**: Relate concepts across different modes of communication
- **Knowledge Graph Integration**: Build connections between visual and textual entities

## Examples
- "Analyze image and text for this picture of my garden plan"
- "What can you tell me about this image [image] in relation to what we discussed earlier?"
- "Remember this image for future reference"

All processing happens locally on your machine.
"""

# Create easy access functions
def get_multimodal_help():
    """Get help text for multimodal capabilities"""
    return get_help()

def process_multimodal(image_path: str, text: str) -> Dict[str, Any]:
    """
    Process an image and text together using multimodal understanding
    
    Args:
        image_path: Path to the image
        text: Text to analyze with the image
        
    Returns:
        Analysis results
    """
    mm = get_multimodal()
    if mm:
        return mm.process_image_with_text(image_path, text)
    return {"error": "Multimodal processor not available"}
