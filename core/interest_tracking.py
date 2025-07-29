"""
Interest Tracking System for Anima

This module tracks topics of interest to the user over time and builds a model
of the user's interests for more personalized interactions.
"""

import os
import sys
import json
import time
import threading
import datetime
from pathlib import Path
import hashlib
from collections import Counter

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Try to import the main awareness system
try:
    from core.awareness import awareness
except ImportError:
    print("Warning: Core awareness module not found. Some features may be limited.")


class InterestTracking:
    """
    Interest tracking system that monitors topics the user discusses
    and builds a model of their interests over time.
    """
    
    def __init__(self):
        """Initialize the interest tracking system"""
        self.memory_path = Path(parent_dir) / "memories" / "user_interests"
        self.memory_path.mkdir(exist_ok=True, parents=True)
        
        # Interest model structure
        self.interest_model = self._load_interest_model()
        
        # Interest categories
        self.categories = {
            "technology": ["programming", "software", "hardware", "ai", "computer", "tech", "code", "development"],
            "science": ["physics", "chemistry", "biology", "astronomy", "research", "experiment"],
            "arts": ["music", "painting", "drawing", "film", "movies", "photography", "design", "creative"],
            "literature": ["books", "reading", "writing", "poetry", "novel", "author"],
            "philosophy": ["philosophy", "ethics", "meaning", "existence", "consciousness"],
            "politics": ["politics", "government", "election", "democracy", "policy"],
            "health": ["health", "medical", "fitness", "diet", "exercise", "wellness"],
            "travel": ["travel", "vacation", "trip", "country", "city", "destination"],
            "sports": ["sports", "game", "football", "soccer", "basketball", "baseball", "tennis"],
            "food": ["food", "cooking", "recipe", "meal", "restaurant", "ingredient"],
            "finance": ["money", "finance", "investment", "stock", "economy", "business"],
            "education": ["education", "learning", "school", "university", "study", "knowledge"]
        }
        
        # Start tracking thread
        self.tracking_thread = threading.Thread(target=self._periodic_save, daemon=True)
        self.tracking_thread.start()
        
        # Time tracking
        self.last_save_time = time.time()
    
    def _load_interest_model(self):
        """Load the existing interest model"""
        model_path = self.memory_path / "interest_model.json"
        if model_path.exists():
            try:
                with open(model_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return self._create_default_model()
        else:
            return self._create_default_model()
    
    def _create_default_model(self):
        """Create a default interest model"""
        return {
            "topics": {},  # Topic -> frequency, recency, engagement score
            "categories": {},  # Category -> score
            "recent_topics": [],  # List of recent topics discussed (max 50)
            "favorites": [],  # User's favorite topics (high engagement)
            "last_updated": datetime.datetime.now().isoformat()
        }
    
    def _save_interest_model(self):
        """Save the interest model to disk"""
        model_path = self.memory_path / "interest_model.json"
        with open(model_path, 'w', encoding='utf-8') as f:
            json.dump(self.interest_model, f, indent=2)
    
    def _periodic_save(self):
        """Periodically save data to disk"""
        while True:
            time.sleep(300)  # Every 5 minutes
            try:
                if time.time() - self.last_save_time > 300:  # Only if changes likely occurred
                    self._save_interest_model()
                    self.last_save_time = time.time()
            except Exception as e:
                print(f"Error saving interest model: {e}")
    
    def _get_category_for_topic(self, topic):
        """Determine which category a topic belongs to"""
        topic_lower = topic.lower()
        
        for category, keywords in self.categories.items():
            for keyword in keywords:
                if keyword in topic_lower:
                    return category
        
        return "other"  # Default category
    
    def _update_topic_frequency(self, topic):
        """Update frequency for a topic"""
        topics = self.interest_model["topics"]
        
        if topic in topics:
            topics[topic]["frequency"] += 1
            topics[topic]["last_mentioned"] = datetime.datetime.now().isoformat()
        else:
            # New topic
            category = self._get_category_for_topic(topic)
            topics[topic] = {
                "frequency": 1,
                "first_mentioned": datetime.datetime.now().isoformat(),
                "last_mentioned": datetime.datetime.now().isoformat(),
                "engagement_score": 1.0,  # Initial score
                "category": category
            }
    
    def _update_categories(self):
        """Update category scores based on topics"""
        category_counts = Counter()
        
        for topic, data in self.interest_model["topics"].items():
            category = data.get("category", "other")
            # Weight by frequency and recency
            score = data["frequency"]
            
            # Add recency factor
            try:
                last_mentioned = datetime.datetime.fromisoformat(data["last_mentioned"])
                days_since = (datetime.datetime.now() - last_mentioned).days
                recency_factor = max(1.0, 10.0 - (days_since * 0.1))  # Higher for recent topics
                score *= recency_factor
            except (ValueError, KeyError):
                pass
                
            category_counts[category] += score
        
        # Normalize scores
        total = sum(category_counts.values())
        if total > 0:
            for category in category_counts:
                category_counts[category] /= total
                
        # Update model
        self.interest_model["categories"] = {k: v for k, v in category_counts.items()}
    
    def _update_recent_topics(self, topic):
        """Update the list of recent topics"""
        recent = self.interest_model["recent_topics"]
        
        # Remove if already in list
        if topic in recent:
            recent.remove(topic)
            
        # Add to beginning
        recent.insert(0, topic)
        
        # Trim to max 50
        self.interest_model["recent_topics"] = recent[:50]
    
    def _update_favorites(self):
        """Update favorite topics based on engagement scores"""
        topics = self.interest_model["topics"]
        
        # Sort by engagement score
        sorted_topics = sorted(
            [(topic, data["engagement_score"]) for topic, data in topics.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Top 10 become favorites
        self.interest_model["favorites"] = [topic for topic, _ in sorted_topics[:10]]
    
    def update_interest_model(self, text, topics=None):
        """
        Update the interest model based on text and/or extracted topics
        Returns True if the model was updated
        """
        if not text and not topics:
            return False
            
        # If topics not provided, try to get from awareness
        if not topics:
            try:
                if hasattr(awareness, "context_memory"):
                    last_exchange = awareness.context_memory.get_recent_exchanges(1)
                    if last_exchange and "topics" in last_exchange[0]:
                        topics = last_exchange[0]["topics"]
            except Exception:
                topics = []
        
        if not topics:
            # Simple topic extraction fallback
            # This is very basic; real implementation would use NLP
            potential_topics = []
            words = text.lower().split()
            for category, keywords in self.categories.items():
                for keyword in keywords:
                    if keyword in words:
                        potential_topics.append(keyword)
            
            topics = list(set(potential_topics))
        
        if not topics:
            return False
            
        # Update each topic
        for topic in topics:
            self._update_topic_frequency(topic)
            self._update_recent_topics(topic)
            
        # Update categories and favorites
        self._update_categories()
        self._update_favorites()
        
        # Mark as updated
        self.interest_model["last_updated"] = datetime.datetime.now().isoformat()
        self.last_save_time = time.time()
        
        return True
    
    def get_top_interests(self, count=5):
        """Get the user's top interests"""
        topics = self.interest_model["topics"]
        
        # Sort by engagement score
        sorted_topics = sorted(
            [(topic, data["engagement_score"]) for topic, data in topics.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        return [topic for topic, _ in sorted_topics[:count]]
    
    def get_interest_summary(self):
        """Get a summary of the user's interests"""
        if not self.interest_model["topics"]:
            return "No interests tracked yet."
            
        favorites = self.interest_model.get("favorites", [])
        categories = self.interest_model.get("categories", {})
        recent = self.interest_model.get("recent_topics", [])[:5]  # Top 5 recent
        
        summary = "Interest profile:\n"
        
        if favorites:
            summary += "- Favorite topics: " + ", ".join(favorites[:5]) + "\n"
            
        if categories:
            top_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:3]
            summary += "- Top categories: " + ", ".join([f"{cat} ({score:.2f})" for cat, score in top_categories]) + "\n"
            
        if recent:
            summary += "- Recently discussed: " + ", ".join(recent) + "\n"
            
        return summary
    
    def enhance_prompt_with_interests(self, prompt, context=None):
        """Enhance a prompt with user interest information"""
        if not self.interest_model["topics"]:
            return prompt
            
        # Select relevant interests if context provided
        relevant_interests = []
        if context:
            context_lower = context.lower()
            for topic, data in self.interest_model["topics"].items():
                if topic.lower() in context_lower:
                    relevant_interests.append((topic, data))
                    
            # Sort by relevance (frequency)
            relevant_interests.sort(key=lambda x: x[1]["frequency"], reverse=True)
            relevant_interests = relevant_interests[:3]  # Top 3
            
        if not relevant_interests:
            # Fall back to favorites
            favorites = self.interest_model.get("favorites", [])
            for fav in favorites[:3]:
                if fav in self.interest_model["topics"]:
                    relevant_interests.append((fav, self.interest_model["topics"][fav]))
        
        # Format interest information
        if relevant_interests:
            interests_text = "\n\nUser interest context:\n"
            for topic, data in relevant_interests:
                interests_text += f"- {topic}: mentioned {data['frequency']} times"
                
                # Add category if available
                if "category" in data and data["category"] != "other":
                    interests_text += f" (category: {data['category']})"
                    
                interests_text += "\n"
                
            # Add to prompt
            split_prompt = prompt.split("\n\n")
            if len(split_prompt) > 1:
                return "\n\n".join(split_prompt[:-1]) + interests_text + "\n\n" + split_prompt[-1]
            else:
                return prompt + interests_text
                
        return prompt


# Create global instance
interest_tracking = InterestTracking()

# Exposed functions
def update_interest_model(text, topics=None):
    """Update the interest model with new information"""
    return interest_tracking.update_interest_model(text, topics)

def get_top_interests(count=5):
    """Get the user's top interests"""
    return interest_tracking.get_top_interests(count)

def get_interest_summary():
    """Get a summary of the user's interests"""
    return interest_tracking.get_interest_summary()

def enhance_prompt_with_interests(prompt, context=None):
    """Enhance prompt with user interest information"""
    return interest_tracking.enhance_prompt_with_interests(prompt, context)
