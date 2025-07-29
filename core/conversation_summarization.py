"""
Conversation Summarization System for Anima

This module automatically generates summaries of conversations to provide
better long-term memory and context for future interactions.
"""

import os
import sys
import json
import time
import threading
import datetime
from pathlib import Path
import hashlib
import re

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Try to import the main awareness system
try:
    from core.awareness import awareness
except ImportError:
    print("Warning: Core awareness module not found. Some features may be limited.")


class ConversationSummarization:
    """
    Conversation summarization system that creates concise summaries of
    conversations to preserve context while reducing token usage.
    """
    
    def __init__(self):
        """Initialize the conversation summarization system"""
        self.memory_path = Path(parent_dir) / "memories" / "conversation_summaries"
        self.memory_path.mkdir(exist_ok=True, parents=True)
        
        self.summaries = self._load_summaries()
        self.pending_conversations = {}  # Conversations waiting to be summarized
        
        # Minimum conversation length that triggers summarization
        self.min_exchanges_for_summary = 10
        
        # Track when conversation was last active
        self.last_activity = {}
        
        # Start summarization thread
        self.summarization_thread = threading.Thread(target=self._periodic_summarize, daemon=True)
        self.summarization_thread.start()
    
    def _load_summaries(self):
        """Load existing summaries"""
        summaries = {}
        
        for file in self.memory_path.glob("*.json"):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    summary_data = json.load(f)
                    conversation_id = summary_data.get("conversation_id")
                    if conversation_id:
                        summaries[conversation_id] = summary_data
            except Exception as e:
                print(f"Error loading summary file {file}: {e}")
                
        return summaries
    
    def _save_summary(self, summary_data):
        """Save a summary to disk"""
        conversation_id = summary_data["conversation_id"]
        file_path = self.memory_path / f"{conversation_id}.json"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2)
    
    def _periodic_summarize(self):
        """Periodically check for conversations to summarize"""
        while True:
            time.sleep(60)  # Check every minute
            
            try:
                current_time = time.time()
                to_summarize = []
                
                # Find conversations that have been inactive for at least 10 minutes
                for conv_id, timestamp in list(self.last_activity.items()):
                    if current_time - timestamp > 600:  # 10 minutes
                        if conv_id in self.pending_conversations and len(self.pending_conversations[conv_id]) >= self.min_exchanges_for_summary:
                            to_summarize.append(conv_id)
                
                # Process each conversation
                for conv_id in to_summarize:
                    exchanges = self.pending_conversations.pop(conv_id, [])
                    if exchanges:
                        self._create_summary(conv_id, exchanges)
                        # Also remove from activity tracker
                        self.last_activity.pop(conv_id, None)
                        
            except Exception as e:
                print(f"Error in summarization thread: {e}")
    
    def _extract_key_points(self, exchanges):
        """Extract key points from conversation exchanges"""
        key_points = []
        
        # Look for questions and answers
        for i, exchange in enumerate(exchanges):
            user_msg = exchange.get("user", "")
            ai_msg = exchange.get("assistant", "")
            
            # Check for questions from user
            if "?" in user_msg:
                questions = [s.strip() for s in re.findall(r'[^.!?]+\?', user_msg)]
                for q in questions:
                    # Only add if substantial question
                    if len(q) > 10:
                        key_points.append(f"User asked: {q}")
            
            # Check for key information in AI responses
            if ai_msg:
                # Look for explanations
                if any(marker in ai_msg for marker in ["explain", "means", "definition", "refers to"]):
                    # Extract the sentence with the explanation
                    explanations = re.findall(r'[^.!?]+(?:explain|means|definition|refers to)[^.!?]+[.!?]', ai_msg)
                    for expl in explanations[:2]:  # Limit to 2 explanations per message
                        key_points.append(f"Anima explained: {expl.strip()}")
                
                # Look for recommendations
                if any(marker in ai_msg for marker in ["recommend", "suggest", "should try", "could use"]):
                    recommendations = re.findall(r'[^.!?]+(?:recommend|suggest|should try|could use)[^.!?]+[.!?]', ai_msg)
                    for rec in recommendations[:2]:
                        key_points.append(f"Anima recommended: {rec.strip()}")
                        
        return key_points
    
    def _extract_main_topics(self, exchanges):
        """Extract main topics from conversation"""
        all_topics = []
        
        for exchange in exchanges:
            # Try to get topics from exchange data
            if "topics" in exchange:
                all_topics.extend(exchange["topics"])
            
            # Fallback: Look for common topic indicators
            for field in ["user", "assistant"]:
                msg = exchange.get(field, "")
                if any(marker in msg.lower() for marker in ["talking about", "discussing", "topic", "subject"]):
                    # Simple extraction - a real system would use NLP
                    potential_topics = re.findall(r'(?:talking about|discussing|topic|subject)[:\s]+([a-zA-Z0-9\s]+)', msg.lower())
                    all_topics.extend([t.strip() for t in potential_topics])
        
        # Count frequency
        topic_counts = {}
        for topic in all_topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
            
        # Get top topics
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, _ in sorted_topics[:5]]  # Top 5 topics
    
    def _create_summary(self, conversation_id, exchanges):
        """Create a summary of the conversation"""
        if not exchanges:
            return
            
        # Extract basic metadata
        start_time = exchanges[0].get("timestamp", datetime.datetime.now().isoformat())
        end_time = exchanges[-1].get("timestamp", datetime.datetime.now().isoformat())
        
        # Convert to datetime objects if they're strings
        if isinstance(start_time, str):
            try:
                start_time = datetime.datetime.fromisoformat(start_time)
            except ValueError:
                start_time = datetime.datetime.now()
                
        if isinstance(end_time, str):
            try:
                end_time = datetime.datetime.fromisoformat(end_time)
            except ValueError:
                end_time = datetime.datetime.now()
        
        # Extract key points and topics
        key_points = self._extract_key_points(exchanges)
        topics = self._extract_main_topics(exchanges)
        
        # Count messages
        user_messages = sum(1 for e in exchanges if "user" in e)
        assistant_messages = sum(1 for e in exchanges if "assistant" in e)
        
        # Create summary
        summary_text = f"Conversation with {user_messages} user messages and {assistant_messages} responses.\n\n"
        
        if topics:
            summary_text += "Main topics discussed:\n"
            for topic in topics:
                summary_text += f"- {topic}\n"
            summary_text += "\n"
            
        if key_points:
            summary_text += "Key points:\n"
            for i, point in enumerate(key_points, 1):
                summary_text += f"{i}. {point}\n"
                
        # Create summary data
        summary_data = {
            "conversation_id": conversation_id,
            "summary": summary_text,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "message_count": len(exchanges),
            "topics": topics,
            "created_at": datetime.datetime.now().isoformat()
        }
        
        # Save the summary
        self.summaries[conversation_id] = summary_data
        self._save_summary(summary_data)
        
        print(f"Created summary for conversation {conversation_id}")
        return summary_data
    
    def add_exchange(self, user_message, assistant_message, conversation_id=None, metadata=None):
        """Add a conversation exchange for future summarization"""
        if not conversation_id:
            # Generate one based on time if not provided
            conversation_id = f"conv_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        # Create exchange data
        exchange = {
            "user": user_message,
            "assistant": assistant_message,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Add any metadata if provided
        if metadata:
            exchange.update(metadata)
            
        # Add to pending conversations
        if conversation_id not in self.pending_conversations:
            self.pending_conversations[conversation_id] = []
            
        self.pending_conversations[conversation_id].append(exchange)
        
        # Update activity timestamp
        self.last_activity[conversation_id] = time.time()
        
        # Check if we should create a summary now
        if len(self.pending_conversations[conversation_id]) >= self.min_exchanges_for_summary * 2:
            # If conversation is getting long, summarize it now
            exchanges = self.pending_conversations.pop(conversation_id, [])
            summary = self._create_summary(conversation_id, exchanges)
            
            # Start fresh for future exchanges
            self.pending_conversations[conversation_id] = []
            
            return summary
            
        return None
    
    def get_summary(self, conversation_id):
        """Get a summary for a specific conversation"""
        return self.summaries.get(conversation_id)
    
    def get_recent_summaries(self, count=3):
        """Get the most recent conversation summaries"""
        # Sort by end_time (most recent first)
        sorted_summaries = sorted(
            self.summaries.values(),
            key=lambda x: x.get("end_time", ""),
            reverse=True
        )
        
        return sorted_summaries[:count]
    
    def format_summary_for_prompt(self, summary):
        """Format a summary for inclusion in a prompt"""
        if not summary:
            return ""
            
        formatted = "\n\nPrevious conversation summary:\n"
        formatted += summary["summary"]
        return formatted
    
    def enhance_prompt_with_summaries(self, prompt, conversation_id=None, count=1):
        """Enhance a prompt with relevant conversation summaries"""
        if conversation_id and conversation_id in self.summaries:
            # Use the specific conversation summary if available
            summary = self.summaries[conversation_id]
            summary_text = self.format_summary_for_prompt(summary)
        else:
            # Otherwise use recent summaries
            recent_summaries = self.get_recent_summaries(count)
            if not recent_summaries:
                return prompt
                
            summary_text = ""
            for summary in recent_summaries:
                summary_text += self.format_summary_for_prompt(summary)
        
        # Add to prompt in a location that won't interfere with system instructions
        if summary_text:
            split_prompt = prompt.split("\n\n")
            if len(split_prompt) > 1:
                # Insert after system instructions but before user input
                return "\n\n".join(split_prompt[:-1]) + summary_text + "\n\n" + split_prompt[-1]
            else:
                return prompt + summary_text
        
        return prompt


# Create global instance
conversation_summarization = ConversationSummarization()

# Exposed functions
def add_exchange(user_message, assistant_message, conversation_id=None, metadata=None):
    """Add a conversation exchange"""
    return conversation_summarization.add_exchange(user_message, assistant_message, conversation_id, metadata)

def get_summary(conversation_id):
    """Get a summary for a specific conversation"""
    return conversation_summarization.get_summary(conversation_id)

def get_recent_summaries(count=3):
    """Get recent conversation summaries"""
    return conversation_summarization.get_recent_summaries(count)

def enhance_prompt_with_summaries(prompt, conversation_id=None, count=1):
    """Enhance prompt with conversation summaries"""
    return conversation_summarization.enhance_prompt_with_summaries(prompt, conversation_id, count)
