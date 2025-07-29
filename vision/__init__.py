"""
Vision module for Anima AI companion.

This module allows Anima to analyze and discuss images with users.
"""

from .vision_processor import (
    analyze_image,
    save_image_memory,
    get_memory,
    get_image_description
)

__all__ = [
    'analyze_image',
    'save_image_memory',
    'get_memory',
    'get_image_description'
]
