"""
Memory Bridge for Anima AI

Connects the emotion system with the awareness/memory system
to ensure emotional data is properly stored and retrieved.
"""

import os
import sys
import time
import logging
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("memory_bridge")

# Global flag to track availability
MEMORY_BRIDGE_AVAILABLE = False

class MemoryBridge:
    """Bridge between emotion system and memory/awareness system."""
    
    def __init__(self):
        """Initialize the memory bridge."""
        self.memory_integration = None
        self.available = False
        
        # Try to import memory integration
        try:
            # First try importing from core.memory_integration
            try:
                from core.memory_integration import memory_integration
                self.memory_integration = memory_integration
                self.available = True
                logger.info("Memory integration loaded via core.memory_integration")
                
                # Register ourselves with the enhanced awareness
                try:
                    # Import and inspect the function to verify it's the correct version
                    import inspect
                    from core.enhanced_awareness_integration import register_with_core_awareness
                    
                    # Log function signature for debugging
                    func_sig = str(inspect.signature(register_with_core_awareness))
                    logger.info(f"register_with_core_awareness signature: {func_sig}")
                    
                    # Manual fallback approach using our own registration method
                    try:
                        # First try with direct access to the awareness instance
                        from core.enhanced_awareness_integration import awareness
                        if awareness and hasattr(awareness, 'register_subsystem'):
                            awareness.register_subsystem("memory_integration", self)
                            logger.info("Directly registered with awareness system")
                            
                            # Set flag directly to ensure it's registered
                            from core.enhanced_awareness_integration import available_modules
                            available_modules["memory_integration"] = True
                            
                            # Also set the global variable
                            import core.enhanced_awareness_integration
                            core.enhanced_awareness_integration.MEMORY_INTEGRATION_AVAILABLE = True
                            
                        else:
                            # Call through the function as a fallback
                            register_with_core_awareness("memory_integration", self)
                            
                        logger.info("Successfully registered memory bridge with enhanced awareness")
                    except Exception as reg_error:
                        logger.error(f"Error during registration: {reg_error}")
                        # Final fallback - just set the flag directly
                        try:
                            import core.enhanced_awareness_integration
                            core.enhanced_awareness_integration.MEMORY_INTEGRATION_AVAILABLE = True
                            logger.info("Manually set MEMORY_INTEGRATION_AVAILABLE flag")
                        except Exception as flag_error:
                            logger.error(f"Error setting availability flag: {flag_error}")
                except ImportError as e:
                    logger.warning(f"Could not register with enhanced awareness integration: {e}")
                    
            except ImportError:
                # Try direct import
                try:
                    import core.memory_integration as memory_module
                    self.memory_integration = memory_module.memory_integration
                    self.available = True
                    logger.info("Memory integration loaded via direct import")
                    
                    # Register ourselves with the enhanced awareness
                    try:
                        # Import and inspect the function to verify it's the correct version
                        import inspect
                        from core.enhanced_awareness_integration import register_with_core_awareness
                        
                        # Manual fallback approach using our own registration method
                        try:
                            # First try with direct access to the awareness instance
                            from core.enhanced_awareness_integration import awareness
                            if awareness and hasattr(awareness, 'register_subsystem'):
                                awareness.register_subsystem("memory_integration", self)
                                logger.info("Directly registered with awareness system")
                                
                                # Set flag directly to ensure it's registered
                                from core.enhanced_awareness_integration import available_modules
                                available_modules["memory_integration"] = True
                                
                                # Also set the global variable
                                import core.enhanced_awareness_integration
                                core.enhanced_awareness_integration.MEMORY_INTEGRATION_AVAILABLE = True
                                
                            else:
                                # Call through the function as a fallback
                                register_with_core_awareness("memory_integration", self)
                                
                            logger.info("Successfully registered memory bridge with enhanced awareness")
                        except Exception as reg_error:
                            logger.error(f"Error during registration: {reg_error}")
                            # Final fallback - just set the flag directly
                            try:
                                import core.enhanced_awareness_integration
                                core.enhanced_awareness_integration.MEMORY_INTEGRATION_AVAILABLE = True
                                logger.info("Manually set MEMORY_INTEGRATION_AVAILABLE flag")
                            except Exception as flag_error:
                                logger.error(f"Error setting availability flag: {flag_error}")
                    except ImportError as e:
                        logger.warning(f"Could not register with enhanced awareness integration: {e}")
                        
                except (ImportError, AttributeError):
                    logger.warning("Memory integration not available")
                    self.memory_integration = None
                    self.available = False
        except Exception as e:
            logger.error(f"Error initializing memory bridge: {e}")
            
    def add_emotional_data(self, emotional_data: Dict[str, Any]) -> bool:
        """Add emotional data to the memory system.
        
        Args:
            emotional_data: Dictionary containing emotional data
            
        Returns:
            True if successful, False otherwise
        """
        if not self.available or not self.memory_integration:
            return False
            
        try:
            # Format the emotional data as memory elements
            memory_elements = self._convert_to_memory_elements(emotional_data)
            
            # Try multiple methods to store emotional data
            # First, try add_memory_elements method
            if hasattr(self.memory_integration, 'add_memory_elements'):
                self.memory_integration.add_memory_elements(memory_elements)
                return True
                
            # Second, try add_memory method
            elif hasattr(self.memory_integration, 'add_memory'):
                # Check if add_memory accepts tags parameter
                try:
                    import inspect
                    sig = inspect.signature(self.memory_integration.add_memory)
                    has_tags_param = 'tags' in sig.parameters
                except Exception:
                    # If we can't inspect, assume no tags parameter
                    has_tags_param = False
                    
                # Store each memory element
                for element in memory_elements:
                    try:
                            # Extract required fields
                            content = element.get('content', '')
                            if not content:
                                logger.warning("Memory element missing content field")
                                content = "Voice emotion analysis" # Fallback content
                                
                            # Get memory type from element type or default to "emotion"
                            memory_type = element.get('type', 'emotion')
                            
                            # Prepare metadata including tags
                            metadata = element.get('metadata', {})
                            if 'tags' in element and not 'tags' in metadata:
                                metadata['tags'] = element.get('tags', [])
                            
                            # Call with correct parameter order: memory_type, content, metadata
                            self.memory_integration.add_memory(memory_type, content, metadata)
                    except TypeError as e:
                        # Log the actual error for debugging
                        logger.error(f"Error in add_memory call: {str(e)}")
                        
                        # Try simplified parameter passing as last resort
                        try:
                            # Extract minimal required fields
                            content = element.get('content', '')
                            if not content and isinstance(element, dict) and 'type' in element:
                                # Generate a fallback content
                                content = f"Memory record: {element['type']}"
                            
                            # Get memory_type (required parameter)
                            memory_type = element.get('type', 'emotion')
                            
                            # Simple add_memory call with required parameters only
                            self.memory_integration.add_memory(memory_type, content)
                            logger.debug(f"Successfully added memory with minimal parameters: {memory_type}, {content}")
                        except Exception as final_e:
                            logger.error(f"All add_memory attempts failed: {final_e}")
                            raise
                return True
                
            # Third, try direct attribute access if it's a class-based structure
            elif hasattr(self.memory_integration, 'memories') and isinstance(self.memory_integration.memories, list):
                self.memory_integration.memories.extend(memory_elements)
                return True
                
            else:
                logger.warning("Memory integration has no compatible storage methods")
                return False
                
        except Exception as e:
            logger.error(f"Error adding emotional data to memory: {e}")
            return False
            
    def _convert_to_memory_elements(self, emotional_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert emotional data to memory elements format.
        
        Args:
            emotional_data: Dictionary containing emotional data
            
        Returns:
            List of memory elements
        """
        # Check if this is direct emotion data from voice analysis
        if "emotion" in emotional_data:
            # Direct format from voice-emotion bridge
            dominant_emotion = emotional_data.get("emotion", "neutral")
            confidence = emotional_data.get("confidence", 0.5)
            timestamp = emotional_data.get("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S"))
            source = emotional_data.get("source", "unknown")
            metrics = emotional_data.get("metrics", {})
            
            # Extract intensity from metrics if available
            intensity = "medium"
            if metrics and "energy" in metrics:
                energy = metrics["energy"]
                if energy > 0.7:
                    intensity = "high"
                elif energy < 0.3:
                    intensity = "low"
            
            # Create content string
            content = f"Voice analysis detected {dominant_emotion} emotion with {intensity} intensity"
            
            # Create memory elements with proper content
            return [{
                "type": "emotion_record",
                "content": content,  # This is crucial - content must be a string
                "metadata": {
                    "emotion": dominant_emotion,
                    "intensity": intensity,
                    "confidence": confidence,
                    "source": source,
                    "timestamp": timestamp,
                    "metrics": metrics,
                    "emotion_scores": emotional_data.get("emotion_scores", {})
                },
                "tags": ["emotion", dominant_emotion, intensity, source]
            }]
        else:
            # Legacy/standard format
            emotions = emotional_data.get("emotions", {})
            dominant_emotion = emotions.get("dominant_emotion", "neutral")
            intensity = emotions.get("dominant_intensity", "medium")
            
            # Create content string - THIS IS THE KEY FIX
            content = f"User expressed {dominant_emotion} emotion with {intensity} intensity"
            
            # Create memory elements with proper content
            return [{
                "type": "emotion_record",
                "content": content,  # This is crucial - content must be a string
                "metadata": {
                    "emotion": dominant_emotion,
                    "intensity": intensity,
                    "confidence": emotions.get("confidence", 0.7),
                    "shift_detected": emotions.get("shift_detected", False),
                    "shift_magnitude": emotions.get("shift_magnitude", 0.0),
                    "related_entities": emotional_data.get("related_entities", []),
                    "timestamp": emotional_data.get("timestamp")
                },
                "tags": ["emotion", dominant_emotion, intensity]
            }]
        
# Global instance for easy access
_memory_bridge = None

def get_memory_bridge() -> MemoryBridge:
    """Get or create the global memory bridge instance.
    
    Returns:
        MemoryBridge instance
    """
    global _memory_bridge
    if _memory_bridge is None:
        _memory_bridge = MemoryBridge()
    return _memory_bridge

# Initialize the singleton instance
_memory_bridge = MemoryBridge()

# Export the instance for direct import
memory_bridge = _memory_bridge

# Export for other modules
__all__ = ["MemoryBridge", "memory_bridge", "get_memory_bridge"]
