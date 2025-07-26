"""
Segment Analysis using LangGraph/LangChain for intelligent highlight detection.

This module combines chat activity, chat messages, and audio transcriptions
to generate intelligent descriptions of highlight-worthy moments using LLMs.
"""

import json
import os
import csv
from typing import Dict, List, Any, Optional
from datetime import datetime
import re

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage
import os

# Try to import different LLM providers
try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from langchain_community.llms import Ollama
    from langchain_community.chat_models import ChatOllama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Local imports
from .db import get_video_paths, update_file_path

# Pydantic models for structured output
class HighlightDescription(BaseModel):
    """Model for highlight description output."""
    time_interval: str = Field(description="Time interval in MM:SS-MM:SS format")
    highlight_type: str = Field(description="Type of highlight (funny, exciting, dramatic, etc.)")
    chat_summary: str = Field(description="Summary of chat messages during this interval")
    audio_summary: str = Field(description="Summary of what was said in audio during this interval")
    combined_description: str = Field(description="Combined description of why this moment is highlight-worthy")
    confidence_score: float = Field(description="Confidence score from 0.0 to 1.0 for highlight worthiness")
    keywords: List[str] = Field(description="Key words/phrases that make this moment interesting")

class SegmentAnalysisState(BaseModel):
    """State for the LangGraph workflow."""
    video_id: str
    time_interval: str
    chat_messages: List[Dict[str, Any]] = []
    audio_transcription: Optional[str] = None
    chat_summary: Optional[str] = None
    audio_summary: Optional[str] = None
    highlight_description: Optional[HighlightDescription] = None
    errors: List[str] = []

def initialize_llm():
    """Initialize the best available LLM in order of preference."""
    
    # Try to load configuration
    try:
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from config import OLLAMA_MODEL, OLLAMA_BASE_URL
        configured_model = OLLAMA_MODEL
        ollama_base_url = OLLAMA_BASE_URL
    except ImportError:
        configured_model = "llama3.2:3b"
        ollama_base_url = "http://localhost:11434"
    except AttributeError:
        configured_model = "llama3.2:3b"
        ollama_base_url = "http://localhost:11434"
    
    # Try OpenAI first (if API key is available)
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        try:
            from config import OPENAI_API_KEY
            openai_key = OPENAI_API_KEY
        except (ImportError, AttributeError):
            pass
    
    if OPENAI_AVAILABLE and openai_key:
        try:
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, api_key=openai_key)
            print("✅ Using OpenAI GPT-3.5-turbo")
            return llm, True
        except Exception as e:
            print(f"⚠️ OpenAI failed: {e}")
    
    # Try Ollama as fallback
    if OLLAMA_AVAILABLE:
        # List of models to try in order of preference
        models_to_try = [
            configured_model,  # User's configured model
            "llama3.2:3b",     # Good balance of speed and quality
            "llama3.2:1b",     # Smaller, faster model
            "llama3.1:8b",     # Alternative model
            "mistral:7b",      # Another good option
        ]
        
        # Remove duplicates while preserving order
        seen = set()
        models_to_try = [x for x in models_to_try if not (x in seen or seen.add(x))]
        
        for model in models_to_try:
            try:
                llm = ChatOllama(
                    model=model,
                    temperature=0.3,
                    timeout=60,
                    base_url=ollama_base_url
                )
                # Test the connection with a simple message
                test_response = llm.invoke([HumanMessage(content="Hello")])
                print(f"✅ Using Ollama with {model}")
                return llm, True
            except Exception as e:
                print(f"⚠️ Ollama with {model} failed: {e}")
                continue
    
    print("❌ No LLM available. Install Ollama or set OPENAI_API_KEY")
    print("\n🔧 Setup Instructions:")
    print("Option 1 - Ollama (Recommended for local use):")
    print("  1. Install Ollama: https://ollama.ai/download")
    print("  2. Run: ollama pull llama3.2:3b")
    print("  3. Start Ollama service: ollama serve")
    print("  4. Or run setup script: python setup_local_llm.py")
    print("\nOption 2 - OpenAI:")
    print("  1. Get API key from https://platform.openai.com/")
    print("  2. Set in config.py: OPENAI_API_KEY='your-key-here'")
    print("  3. Or set environment: export OPENAI_API_KEY='your-key-here'")
    
    return None, False

# Initialize LLM
llm, LLM_AVAILABLE = initialize_llm()

def load_chat_data(video_id: str) -> tuple[Dict, Dict, str]:
    """
    Load chat activity, chat messages, and transcriptions for a video.
    
    Returns:
        tuple: (chat_activity_data, chat_csv_data, chat_transcriptions_data)
    """
    paths = get_video_paths(video_id)
    if not paths:
        raise ValueError(f"Video {video_id} not found in database")
    
    # Load chat activity JSON
    chat_activity = {}
    if paths['chat_json_path'] and os.path.exists(paths['chat_json_path']):
        with open(paths['chat_json_path'], 'r', encoding='utf-8') as f:
            chat_activity = json.load(f)
    
    # Load chat CSV
    chat_messages = []
    if paths['chat_csv_path'] and os.path.exists(paths['chat_csv_path']):
        with open(paths['chat_csv_path'], 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            chat_messages = list(reader)
    
    # Load chat transcriptions
    chat_transcriptions = {}
    if paths['chat_transcript_path'] and os.path.exists(paths['chat_transcript_path']):
        with open(paths['chat_transcript_path'], 'r', encoding='utf-8') as f:
            chat_transcriptions = json.load(f)
    
    return chat_activity, chat_messages, chat_transcriptions

def find_top_chat_intervals(chat_activity: Dict, top_n: int = 5) -> List[tuple]:
    """
    Find the top N intervals with highest chat activity.
    
    Returns:
        List of (interval, message_count) tuples sorted by message count
    """
    # Convert to list and sort by message count
    intervals = [(interval, count) for interval, count in chat_activity.items()]
    intervals.sort(key=lambda x: x[1], reverse=True)
    return intervals[:top_n]

def parse_time_to_seconds(time_str):
    """Parse time string to seconds, handling both MM:SS and H:MM:SS formats"""
    parts = time_str.split(':')
    if len(parts) == 2:
        # MM:SS format
        minutes, seconds = map(int, parts)
        return minutes * 60 + seconds
    elif len(parts) == 3:
        # H:MM:SS format
        hours, minutes, seconds = map(int, parts)
        return hours * 3600 + minutes * 60 + seconds
    else:
        raise ValueError(f'Invalid time format: {time_str}')
    
def parse_time_interval(interval: str) -> tuple[int, int]:
    """
    Parse time interval string to start and end seconds.
    Handles intervals that might be in different formats.
    """
    start_str, end_str = interval.split('-')
    
    # Handle the case where interval is in MM:SS format but should be H:MM:SS
    # Convert MM:SS to H:MM:SS if minutes > 59
    def convert_if_needed(time_str):
        parts = time_str.split(':')
        if len(parts) == 2:
            minutes, seconds = map(int, parts)
            if minutes >= 60:
                # Convert to H:MM:SS format
                hours = minutes // 60
                mins = minutes % 60
                return f"{hours}:{mins:02d}:{seconds:02d}"
        return time_str
    
    start_str = convert_if_needed(start_str)
    end_str = convert_if_needed(end_str)
    
    return parse_time_to_seconds(start_str), parse_time_to_seconds(end_str)

def get_messages_in_interval(chat_messages: List[Dict], interval: str) -> List[Dict]:
    """
    Get all chat messages within a specific time interval.
    """
    start_sec, end_sec = parse_time_interval(interval)
    interval_messages = []
    
    for message in chat_messages:
        try:
            # Parse timestamp from chat CSV
            timestamp_str = message.get('Timestamp', '')
            if timestamp_str:
                try:
                    message_seconds = parse_time_to_seconds(timestamp_str)
                    if start_sec <= message_seconds <= end_sec:
                        interval_messages.append(message)
                except ValueError:
                    # Skip messages with invalid timestamps
                    continue
        except Exception:
            continue
    
    return interval_messages

# LangGraph node functions
def extract_interval_data(state: SegmentAnalysisState) -> SegmentAnalysisState:
    """Extract chat messages and transcription for the specified interval."""
    try:
        chat_activity, chat_messages, chat_transcriptions = load_chat_data(state.video_id)
        
        # Get messages in the interval
        interval_messages = get_messages_in_interval(chat_messages, state.time_interval)
        state.chat_messages = interval_messages
        
        # Get audio transcription for this interval
        if state.time_interval in chat_transcriptions:
            transcription_data = chat_transcriptions[state.time_interval]
            if isinstance(transcription_data, dict):
                state.audio_transcription = transcription_data.get('text', '')
            else:
                state.audio_transcription = str(transcription_data)
        
        print(f"✓ Extracted {len(interval_messages)} chat messages and audio transcription")
        
    except Exception as e:
        error_msg = f"Error extracting interval data: {e}"
        state.errors.append(error_msg)
        print(f"❌ {error_msg}")
    
    return state

def summarize_chat_messages(state: SegmentAnalysisState) -> SegmentAnalysisState:
    """Summarize chat messages using LLM."""
    if not LLM_AVAILABLE or not state.chat_messages:
        state.chat_summary = f"Found {len(state.chat_messages)} chat messages during this interval"
        return state
    
    try:
        # Prepare chat messages for analysis
        messages_text = "\n".join([
            f"[{msg.get('Author', 'Unknown')}]: {msg.get('Message', '')}"
            for msg in state.chat_messages[:100]  # Limit to first 100 messages
        ])
        
        chat_prompt = ChatPromptTemplate.from_template("""
        Analyze the following chat messages from a gaming/streaming context and provide a concise summary:
        
        Chat Messages:
        {messages}
        
        Provide a summary that captures:
        1. The general mood/sentiment of the chat
        2. Key topics or reactions being discussed
        3. Any notable patterns or spam/emotes
        4. Overall excitement level
        
        Keep the summary under 100 words and focus on what makes this moment interesting.
        """)
        
        chain = chat_prompt | llm
        response = chain.invoke({"messages": messages_text})
        state.chat_summary = response.content
        
        print("✓ Generated chat summary")
        
    except Exception as e:
        error_msg = f"Error summarizing chat: {e}"
        state.errors.append(error_msg)
        state.chat_summary = f"Analysis of {len(state.chat_messages)} chat messages (summary generation failed)"
        print(f"❌ {error_msg}")
    
    return state

def summarize_audio_transcription(state: SegmentAnalysisState) -> SegmentAnalysisState:
    """Summarize audio transcription using LLM."""
    if not LLM_AVAILABLE or not state.audio_transcription:
        state.audio_summary = "No audio transcription available for this interval"
        return state
    
    try:
        audio_prompt = ChatPromptTemplate.from_template("""
        Analyze the following audio transcription from a gaming/streaming context:
        
        Audio Transcription:
        {transcription}
        
        Provide a concise summary that captures:
        1. What the streamer was saying or reacting to
        2. The tone/emotion in their speech
        3. Any significant events or reactions
        4. Context that would make this moment highlight-worthy
        
        Keep the summary under 80 words and focus on the key moments.
        """)
        
        chain = audio_prompt | llm
        response = chain.invoke({"transcription": state.audio_transcription})
        state.audio_summary = response.content
        
        print("✓ Generated audio summary")
        
    except Exception as e:
        error_msg = f"Error summarizing audio: {e}"
        state.errors.append(error_msg)
        state.audio_summary = "Audio transcription analysis failed"
        print(f"❌ {error_msg}")
    
    return state

def generate_highlight_description(state: SegmentAnalysisState) -> SegmentAnalysisState:
    """Generate final highlight description combining chat and audio analysis."""
    if not LLM_AVAILABLE:
        # Fallback description without LLM
        state.highlight_description = HighlightDescription(
            time_interval=state.time_interval,
            highlight_type="chat_activity",
            chat_summary=state.chat_summary or "High chat activity",
            audio_summary=state.audio_summary or "No audio analysis",
            combined_description=f"High activity segment with {len(state.chat_messages)} chat messages",
            confidence_score=0.7,
            keywords=["high_activity", "chat_spike"]
        )
        return state
    
    try:
        # Create structured output parser
        parser = JsonOutputParser(pydantic_object=HighlightDescription)
        
        highlight_prompt = ChatPromptTemplate.from_template("""
        Based on the chat and audio analysis below, generate a highlight description for this streaming moment:

        Time Interval: {time_interval}
        Chat Summary: {chat_summary}
        Audio Summary: {audio_summary}
        Number of Chat Messages: {message_count}

        Create a comprehensive highlight description that combines both the chat activity and audio content.
        Consider what type of moment this represents (funny, exciting, dramatic, skillful, etc.).

        IMPORTANT: Return ONLY the JSON object with actual values, not a schema definition.

        {format_instructions}

        Example of expected output:
        {{
        "time_interval": "267:00-267:05",
        "highlight_type": "funny",
        "chat_summary": "Chat was laughing and spamming emotes",
        "audio_summary": "Streamer made a joke",
        "combined_description": "Funny moment where streamer's joke caused chat to explode with laughter",
        "confidence_score": 0.8,
        "keywords": ["funny", "joke", "laughter"]
        }}
        """)
        
        chain = highlight_prompt | llm | parser
        
        result = chain.invoke({
            "time_interval": state.time_interval,
            "chat_summary": state.chat_summary or "No chat summary available",
            "audio_summary": state.audio_summary or "No audio summary available", 
            "message_count": len(state.chat_messages),
            "format_instructions": parser.get_format_instructions()
        })
        
        state.highlight_description = HighlightDescription(**result)
        print("✓ Generated highlight description")
        
    except Exception as e:
        error_msg = f"Error generating highlight description: {e}"
        state.errors.append(error_msg)
        print(f"❌ {error_msg}")
        
        # Fallback description
        state.highlight_description = HighlightDescription(
            time_interval=state.time_interval,
            highlight_type="unknown",
            chat_summary=state.chat_summary or "Analysis failed",
            audio_summary=state.audio_summary or "Analysis failed",
            combined_description="Error occurred during analysis",
            confidence_score=0.1,
            keywords=["error"]
        )
    
    return state

# Create LangGraph workflow
def create_segment_analysis_graph():
    """Create the LangGraph workflow for segment analysis."""
    from langgraph.graph import StateGraph, END
    
    # Create StateGraph with our state type
    workflow = StateGraph(SegmentAnalysisState)
    
    # Add nodes
    workflow.add_node("extract_data", extract_interval_data)
    workflow.add_node("summarize_chat", summarize_chat_messages)
    workflow.add_node("summarize_audio", summarize_audio_transcription)
    workflow.add_node("generate_highlight", generate_highlight_description)
    
    # Add edges
    workflow.add_edge("extract_data", "summarize_chat")
    workflow.add_edge("summarize_chat", "summarize_audio")
    workflow.add_edge("summarize_audio", "generate_highlight")
    workflow.add_edge("generate_highlight", END)
    
    # Set entry point
    workflow.set_entry_point("extract_data")
    
    return workflow.compile()

def analyze_top_segments(video_id: str, top_n: int = 5) -> Dict[str, Any]:
    """
    Analyze the top N segments with highest chat activity.
    
    Args:
        video_id: Video ID to analyze
        top_n: Number of top segments to analyze
    
    Returns:
        Dictionary with analysis results and highlight descriptions
    """
    print(f"🔍 Analyzing top {top_n} segments for video {video_id}")
    
    try:
        # Load data
        chat_activity, _, _ = load_chat_data(video_id)
        
        # Find top intervals
        top_intervals = find_top_chat_intervals(chat_activity, top_n)
        print(f"Found top intervals: {[(i, c) for i, c in top_intervals]}")
        
        # Create workflow
        workflow = create_segment_analysis_graph()
        
        # Analyze each interval
        highlights = {}
        
        for interval, message_count in top_intervals:
            print(f"\n--- Analyzing interval {interval} ({message_count} messages) ---")
            
            # Create initial state
            state = SegmentAnalysisState(
                video_id=video_id,
                time_interval=interval
            )
            
            # Run workflow
            final_state = workflow.invoke(state)
            
            # Handle both dict and SegmentAnalysisState returns
            if isinstance(final_state, dict):
                # LangGraph returned a dict
                highlight_desc = final_state.get('highlight_description')
                errors = final_state.get('errors', [])
            else:
                # LangGraph returned a SegmentAnalysisState object
                highlight_desc = final_state.highlight_description
                errors = final_state.errors
            
            # Store result
            if highlight_desc:
                if hasattr(highlight_desc, 'dict'):
                    highlights[interval] = highlight_desc.dict()
                else:
                    highlights[interval] = highlight_desc
            
            if errors:
                print(f"Errors during analysis: {errors}")
        
        # Save results
        paths = get_video_paths(video_id)
        if paths:
            video_dir = os.path.dirname(paths['video_path'])
            output_path = os.path.join(video_dir, "highlight_descriptions.json")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(highlights, f, ensure_ascii=False, indent=2)
            
            # Update database
            update_file_path(video_id, highlight_descriptions_path=output_path)
            
            print(f"\n✅ Analysis complete! Results saved to: {output_path}")
            
            return {
                "video_id": video_id,
                "highlights": highlights,
                "output_file": output_path,
                "total_segments_analyzed": len(highlights),
                "top_intervals_found": len(top_intervals)
            }
        
    except Exception as e:
        print(f"❌ Error during segment analysis: {e}")
        return {
            "video_id": video_id,
            "error": str(e),
            "highlights": {},
            "total_segments_analyzed": 0
        }

def analyze_segment_by_interval(video_id: str, time_interval: str) -> Dict[str, Any]:
    """
    Analyze a specific time interval for highlight worthiness.
    
    Args:
        video_id: Video ID to analyze
        time_interval: Specific interval to analyze (e.g., "267:00-267:05")
    
    Returns:
        Analysis result for the specific interval
    """
    print(f"🔍 Analyzing specific segment {time_interval} for video {video_id}")
    
    try:
        # Create workflow
        workflow = create_segment_analysis_graph()
        
        # Create initial state
        state = SegmentAnalysisState(
            video_id=video_id,
            time_interval=time_interval
        )
        
        # Run workflow
        final_state = workflow.invoke(state)
        
        # Handle both dict and SegmentAnalysisState returns
        if isinstance(final_state, dict):
            highlight_desc = final_state.get('highlight_description')
            errors = final_state.get('errors', [])
        else:
            highlight_desc = final_state.highlight_description
            errors = final_state.errors
        
        if highlight_desc:
            if hasattr(highlight_desc, 'dict'):
                result = highlight_desc.dict()
            else:
                result = highlight_desc
            print(f"✅ Analysis complete for interval {time_interval}")
            return {
                "video_id": video_id,
                "interval": time_interval,
                "highlight_description": result,
                "errors": errors
            }
        else:
            return {
                "video_id": video_id,
                "interval": time_interval,
                "error": "Failed to generate highlight description",
                "errors": errors
            }
        
    except Exception as e:
        print(f"❌ Error analyzing interval {time_interval}: {e}")
        return {
            "video_id": video_id,
            "interval": time_interval,
            "error": str(e)
        } 