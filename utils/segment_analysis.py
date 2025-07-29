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

try:
    from langchain_community.chat_models import QianfanChatEndpoint
    # For Qwen models, we can also use OpenAI-compatible endpoints
    QWEN_AVAILABLE = True
except ImportError:
    QWEN_AVAILABLE = False

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
    """State for segment analysis workflow."""
    video_id: str
    time_interval: str
    chat_messages: List[Dict[str, Any]] = []  # Made optional with default
    audio_transcription: Optional[Dict[str, Any]] = None
    chat_summary: Optional[str] = None
    audio_summary: Optional[str] = None
    highlight_description: Optional[HighlightDescription] = None
    errors: List[str] = []

def initialize_llm():
    """Initialize the best available LLM in order of preference: OpenAI -> Qwen -> Ollama."""
    
    # Try to load configuration
    try:
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        import config
        configured_ollama_model = getattr(config, 'OLLAMA_MODEL', 'llama3.2:3b')
        ollama_base_url = getattr(config, 'OLLAMA_BASE_URL', 'http://localhost:11434')
        qwen_api_key = getattr(config, 'QWEN_API_KEY', None)
        qwen_base_url = getattr(config, 'QWEN_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
        qwen_model = getattr(config, 'QWEN_MODEL', 'qwen-plus')
    except ImportError:
        configured_ollama_model = "llama3.2:3b"
        ollama_base_url = "http://localhost:11434"
        qwen_api_key = None
        qwen_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        qwen_model = "qwen-plus"
    except AttributeError:
        configured_ollama_model = "llama3.2:3b"
        ollama_base_url = "http://localhost:11434"
        qwen_api_key = None
        qwen_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        qwen_model = "qwen-plus"
    
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
    
    # Try Qwen next (if API key is available)
    if not qwen_api_key:
        qwen_api_key = os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    
    if QWEN_AVAILABLE and qwen_api_key:
        try:
            # Use OpenAI-compatible endpoint for Qwen
            llm = ChatOpenAI(
                model=qwen_model,
                temperature=0.3,
                api_key=qwen_api_key,
                base_url=qwen_base_url
            )
            # Test the connection
            test_response = llm.invoke([HumanMessage(content="Hello")])
            print(f"✅ Using Qwen model: {qwen_model}")
            return llm, True
        except Exception as e:
            print(f"⚠️ Qwen failed: {e}")
    
    # Try Ollama as final fallback
    if OLLAMA_AVAILABLE:
        # List of models to try in order of preference
        models_to_try = [
            configured_ollama_model,  # User's configured model
            "qwen2.5:3b",      # Qwen 3B - Good balance for highlights
            "qwen2.5:1.5b",    # Qwen 1.5B - Faster option
            "qwen2.5:7b",      # Qwen 7B - Higher quality
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
    
    print("❌ No LLM available. Install Ollama, set OPENAI_API_KEY, or configure Qwen")
    print("\n🔧 Setup Instructions:")
    print("Option 1 - OpenAI:")
    print("  Set OPENAI_API_KEY environment variable")
    print("\nOption 2 - Qwen (Alibaba Cloud):")
    print("  1. Get API key from: https://dashscope.console.aliyun.com/")
    print("  2. Set QWEN_API_KEY or DASHSCOPE_API_KEY environment variable")
    print("  3. Available models: qwen-plus, qwen-turbo, qwen-max, etc.")
    print("\nOption 3 - Ollama (Local/Free):")
    print("  1. Install Ollama: https://ollama.ai/download")
    
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


# ================================================================================================
# MULTI-MODAL HIGHLIGHT SCORING & RANKING SYSTEM
# ================================================================================================

class MultiModalScore(BaseModel):
    """Model for multi-modal scoring output."""
    time_interval: str = Field(description="Time interval")
    chat_score: float = Field(description="Chat activity score (0.0-1.0)")
    emotion_score: float = Field(description="Video emotion score (0.0-1.0)")
    laugh_score: float = Field(description="Audio laugh score (0.0-1.0)")
    transcription_score: float = Field(description="Audio transcription quality score (0.0-1.0)")
    combined_score: float = Field(description="Weighted combined score (0.0-1.0)")
    highlight_type: str = Field(description="Type of highlight based on dominant signals")
    reasoning: str = Field(description="Explanation of why this moment scored highly")

class MultiModalAnalysisState(BaseModel):
    """State for the multi-modal analysis workflow."""
    video_id: str
    time_interval: str
    interval_data_dir: str  # Directory containing all interval-specific JSON files
    
    # Raw data
    chat_activity_count: int = 0
    chat_messages: List[Dict[str, Any]] = []
    video_emotions: List[Dict[str, Any]] = []
    audio_laughs: List[Dict[str, Any]] = []
    audio_transcription: Optional[str] = None
    
    # Individual scores
    chat_score: float = 0.0
    emotion_score: float = 0.0
    laugh_score: float = 0.0
    transcription_score: float = 0.0
    
    # Final result
    multi_modal_score: Optional[MultiModalScore] = None
    errors: List[str] = []

def load_multi_modal_data(video_id: str, time_interval: str, interval_data_dir: str) -> Dict[str, Any]:
    """Load all multi-modal data for a specific interval."""
    data = {
        'chat_activity_count': 0,
        'chat_messages': [],
        'video_emotions': [],
        'audio_laughs': [],
        'audio_transcription': None
    }
    
    try:
        # Load chat activity
        chat_activity_path = os.path.join(interval_data_dir, 'chat_activity.json')
        if os.path.exists(chat_activity_path):
            with open(chat_activity_path, 'r', encoding='utf-8') as f:
                chat_activity = json.load(f)
                data['chat_activity_count'] = chat_activity.get(time_interval, 0)
        
        # Load chat messages (from CSV)
        paths = get_video_paths(video_id)
        if paths and paths['chat_csv_path'] and os.path.exists(paths['chat_csv_path']):
            with open(paths['chat_csv_path'], 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                all_messages = list(reader)
                data['chat_messages'] = get_messages_in_interval(all_messages, time_interval)
        
        # Load video emotions
        emotions_path = os.path.join(interval_data_dir, 'video_emotions.json')
        if os.path.exists(emotions_path):
            with open(emotions_path, 'r', encoding='utf-8') as f:
                emotions_data = json.load(f)
                emotion_value = emotions_data.get(time_interval, [])
                
                # Convert string emotion to list format
                if isinstance(emotion_value, str):
                    if emotion_value == "no_face":
                        data['video_emotions'] = []
                    else:
                        data['video_emotions'] = [{'emotion': emotion_value, 'confidence': 0.8}]
                elif isinstance(emotion_value, list):
                    data['video_emotions'] = emotion_value
                else:
                    data['video_emotions'] = []
        
        # Load audio laughs
        laughs_path = os.path.join(interval_data_dir, 'audio_laughs.json')
        if os.path.exists(laughs_path):
            with open(laughs_path, 'r', encoding='utf-8') as f:
                laughs_data = json.load(f)
                laugh_value = laughs_data.get(time_interval, [])
                
                # Convert boolean laugh to list format
                if isinstance(laugh_value, bool):
                    if laugh_value:
                        data['audio_laughs'] = [{'confidence': 0.7, 'detected': True}]
                    else:
                        data['audio_laughs'] = []
                elif isinstance(laugh_value, list):
                    data['audio_laughs'] = laugh_value
                else:
                    data['audio_laughs'] = []
        
        # Load audio transcription
        transcription_path = os.path.join(interval_data_dir, 'chat_transcriptions.json')
        if os.path.exists(transcription_path):
            with open(transcription_path, 'r', encoding='utf-8') as f:
                transcription_data = json.load(f)
                interval_transcription = transcription_data.get(time_interval)
                if isinstance(interval_transcription, dict):
                    data['audio_transcription'] = interval_transcription.get('text', '')
                else:
                    data['audio_transcription'] = str(interval_transcription) if interval_transcription else None
    
    except Exception as e:
        print(f"Error loading multi-modal data: {e}")
    
    return data

# LangGraph workflow nodes for multi-modal analysis
def extract_multi_modal_data(state: MultiModalAnalysisState) -> MultiModalAnalysisState:
    """Extract all multi-modal data for the interval."""
    try:
        data = load_multi_modal_data(state.video_id, state.time_interval, state.interval_data_dir)
        
        state.chat_activity_count = data['chat_activity_count']
        state.chat_messages = data['chat_messages']
        state.video_emotions = data['video_emotions']
        state.audio_laughs = data['audio_laughs']
        state.audio_transcription = data['audio_transcription']
        
        print(f"✓ Loaded multi-modal data: {len(state.chat_messages)} chat, {len(state.video_emotions)} emotions, {len(state.audio_laughs)} laughs")
        
    except Exception as e:
        error_msg = f"Error extracting multi-modal data: {e}"
        state.errors.append(error_msg)
        print(f"❌ {error_msg}")
    
    return state

def score_chat_activity(state: MultiModalAnalysisState) -> MultiModalAnalysisState:
    """Score chat activity based on message count and content."""
    try:
        # Base score from message count (normalize to 0-1)
        message_count = len(state.chat_messages)
        count_score = min(message_count / 50.0, 1.0)  # Cap at 50 messages = 1.0
        
        # Content quality score (emotes, excitement indicators)
        if message_count > 0:
            excitement_keywords = ['lol', 'lmao', 'kekw', 'omg', 'wow', 'pog', 'hype', '!!!']
            emote_patterns = ['lul', 'kappa', 'pepega', 'monka', 'pog']
            
            excitement_count = 0
            for msg in state.chat_messages:
                content = msg.get('Message', '').lower()
                if any(keyword in content for keyword in excitement_keywords):
                    excitement_count += 1
                if any(emote in content for emote in emote_patterns):
                    excitement_count += 1
            
            content_score = min(excitement_count / message_count, 1.0) if message_count > 0 else 0.0
        else:
            content_score = 0.0
        
        # Combined chat score (weighted)
        state.chat_score = (count_score * 0.7) + (content_score * 0.3)
        print(f"✓ Chat score: {state.chat_score:.3f} (count: {count_score:.3f}, content: {content_score:.3f})")
        
    except Exception as e:
        error_msg = f"Error scoring chat activity: {e}"
        state.errors.append(error_msg)
        print(f"❌ {error_msg}")
        state.chat_score = 0.0
    
    return state

def score_video_emotions(state: MultiModalAnalysisState) -> MultiModalAnalysisState:
    """Score video emotions based on detected emotions."""
    try:
        if not state.video_emotions:
            state.emotion_score = 0.0
            print("✓ Emotion score: 0.000 (no emotions detected)")
            return state
        
        # Score based on emotion intensity and variety
        positive_emotions = ['happy', 'surprise', 'joy', 'excited']
        negative_emotions = ['angry', 'fear', 'disgust', 'sad']
        
        total_intensity = 0.0
        emotion_variety = set()
        
        for emotion_data in state.video_emotions:
            if isinstance(emotion_data, dict):
                emotion = emotion_data.get('emotion', '').lower()
                confidence = emotion_data.get('confidence', 0.0)
                
                emotion_variety.add(emotion)
                
                # Weight positive emotions higher for highlights
                if emotion in positive_emotions:
                    total_intensity += confidence * 1.2
                elif emotion in negative_emotions:
                    total_intensity += confidence * 0.8
                else:
                    total_intensity += confidence
        
        # Normalize and boost for variety
        avg_intensity = total_intensity / len(state.video_emotions) if state.video_emotions else 0.0
        variety_boost = min(len(emotion_variety) / 5.0, 1.0) * 0.2  # Up to 20% boost for variety
        
        state.emotion_score = min(avg_intensity + variety_boost, 1.0)
        print(f"✓ Emotion score: {state.emotion_score:.3f} (intensity: {avg_intensity:.3f}, variety: {len(emotion_variety)})")
        
    except Exception as e:
        error_msg = f"Error scoring video emotions: {e}"
        state.errors.append(error_msg)
        print(f"❌ {error_msg}")
        state.emotion_score = 0.0
    
    return state

def score_audio_laughs(state: MultiModalAnalysisState) -> MultiModalAnalysisState:
    """Score audio laughs based on detected laughter."""
    try:
        if not state.audio_laughs:
            state.laugh_score = 0.0
            print("✓ Laugh score: 0.000 (no laughs detected)")
            return state
        
        # Score based on laugh count and confidence
        total_confidence = 0.0
        laugh_count = len(state.audio_laughs)
        
        for laugh_data in state.audio_laughs:
            if isinstance(laugh_data, dict):
                confidence = laugh_data.get('confidence', 0.0)
                total_confidence += confidence
        
        # Normalize and apply count bonus
        avg_confidence = total_confidence / laugh_count if laugh_count > 0 else 0.0
        count_bonus = min(laugh_count / 10.0, 1.0) * 0.3  # Up to 30% bonus for multiple laughs
        
        state.laugh_score = min(avg_confidence + count_bonus, 1.0)
        print(f"✓ Laugh score: {state.laugh_score:.3f} (confidence: {avg_confidence:.3f}, count: {laugh_count})")
        
    except Exception as e:
        error_msg = f"Error scoring audio laughs: {e}"
        state.errors.append(error_msg)
        print(f"❌ {error_msg}")
        state.laugh_score = 0.0
    
    return state

def score_audio_transcription(state: MultiModalAnalysisState) -> MultiModalAnalysisState:
    """Score audio transcription based on content quality and excitement."""
    try:
        if not state.audio_transcription:
            state.transcription_score = 0.0
            print("✓ Transcription score: 0.000 (no transcription)")
            return state
        
        text = state.audio_transcription.lower()
        
        # Score based on excitement keywords and patterns
        excitement_words = ['amazing', 'incredible', 'unbelievable', 'wow', 'omg', 'insane', 'clutch', 'epic']
        reaction_words = ['yes!', 'no way', 'what!', 'how!', 'damn', 'sick', 'poggers']
        
        excitement_score = sum(1 for word in excitement_words if word in text) / 10.0
        reaction_score = sum(1 for word in reaction_words if word in text) / 5.0
        
        # Length bonus (longer transcriptions often mean more content)
        length_score = min(len(text.split()) / 50.0, 1.0) * 0.3
        
        state.transcription_score = min(excitement_score + reaction_score + length_score, 1.0)
        print(f"✓ Transcription score: {state.transcription_score:.3f} (excitement: {excitement_score:.3f}, reactions: {reaction_score:.3f})")
        
    except Exception as e:
        error_msg = f"Error scoring audio transcription: {e}"
        state.errors.append(error_msg)
        print(f"❌ {error_msg}")
        state.transcription_score = 0.0
    
    return state

def combine_multi_modal_scores(state: MultiModalAnalysisState) -> MultiModalAnalysisState:
    """Combine all scores into a final multi-modal score."""
    try:
        # Weighted combination of scores
        weights = {
            'chat': 0.3,      # Chat activity is important for engagement
            'emotion': 0.25,  # Visual emotions show reaction
            'laugh': 0.3,     # Laughter is a strong highlight indicator
            'transcription': 0.15  # Audio content provides context
        }
        
        combined_score = (
            state.chat_score * weights['chat'] +
            state.emotion_score * weights['emotion'] +
            state.laugh_score * weights['laugh'] +
            state.transcription_score * weights['transcription']
        )
        
        # Determine highlight type based on dominant signals
        scores = {
            'chat': state.chat_score,
            'emotion': state.emotion_score,
            'laugh': state.laugh_score,
            'transcription': state.transcription_score
        }
        
        dominant_signal = max(scores, key=scores.get)
        highlight_types = {
            'chat': 'high_engagement',
            'emotion': 'emotional_moment',
            'laugh': 'funny_moment',
            'transcription': 'content_rich'
        }
        
        highlight_type = highlight_types.get(dominant_signal, 'mixed_moment')
        
        # Generate reasoning
        reasoning_parts = []
        if state.chat_score > 0.6:
            reasoning_parts.append(f"High chat activity ({len(state.chat_messages)} messages)")
        if state.emotion_score > 0.6:
            reasoning_parts.append(f"Strong emotional reactions detected")
        if state.laugh_score > 0.6:
            reasoning_parts.append(f"Multiple laugh instances ({len(state.audio_laughs)} laughs)")
        if state.transcription_score > 0.6:
            reasoning_parts.append(f"Exciting audio content detected")
        
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Moderate activity across multiple signals"
        
        # Create final score object
        state.multi_modal_score = MultiModalScore(
            time_interval=state.time_interval,
            chat_score=state.chat_score,
            emotion_score=state.emotion_score,
            laugh_score=state.laugh_score,
            transcription_score=state.transcription_score,
            combined_score=combined_score,
            highlight_type=highlight_type,
            reasoning=reasoning
        )
        
        print(f"✓ Combined score: {combined_score:.3f} ({highlight_type})")
        
    except Exception as e:
        error_msg = f"Error combining multi-modal scores: {e}"
        state.errors.append(error_msg)
        print(f"❌ {error_msg}")
    
    return state

def create_multi_modal_analysis_graph():
    """Create the LangGraph workflow for multi-modal analysis."""
    workflow = StateGraph(MultiModalAnalysisState)
    
    # Add nodes
    workflow.add_node("extract_data", extract_multi_modal_data)
    workflow.add_node("score_chat", score_chat_activity)
    workflow.add_node("score_emotions", score_video_emotions)
    workflow.add_node("score_laughs", score_audio_laughs)
    workflow.add_node("score_transcription", score_audio_transcription)
    workflow.add_node("combine_scores", combine_multi_modal_scores)
    
    # Add edges
    workflow.add_edge("extract_data", "score_chat")
    workflow.add_edge("score_chat", "score_emotions")
    workflow.add_edge("score_emotions", "score_laughs")
    workflow.add_edge("score_laughs", "score_transcription")
    workflow.add_edge("score_transcription", "combine_scores")
    workflow.add_edge("combine_scores", END)
    
    # Set entry point
    workflow.set_entry_point("extract_data")
    
    return workflow.compile()

def analyze_multi_modal_highlights(video_id: str, interval_data_dir: str, top_n: int = 10) -> Dict[str, Any]:
    """
    Analyze intervals using smart filtering based on chat engagement with fallbacks for low-chat streams.
    
    Args:
        video_id: Video ID to analyze
        interval_data_dir: Directory containing interval-specific JSON files
        top_n: Number of top highlights to return
    
    Returns:
        Dictionary with ranked highlights and scores
    """
    print(f"🎯 Smart Multi-Modal Analysis for video {video_id}")
    print(f"📁 Using interval data from: {interval_data_dir}")
    
    try:
        # Load chat activity to get all intervals
        chat_activity_path = os.path.join(interval_data_dir, 'chat_activity.json')
        if not os.path.exists(chat_activity_path):
            raise ValueError(f"Chat activity file not found: {chat_activity_path}")
        
        with open(chat_activity_path, 'r', encoding='utf-8') as f:
            chat_activity = json.load(f)
        
        # Smart interval selection strategy
        selected_intervals = smart_interval_selection(chat_activity, interval_data_dir)
        
        print(f"🧠 Smart selection: Analyzing {len(selected_intervals)} intervals (from {len(chat_activity)} total)")
        
        # Create workflow
        workflow = create_multi_modal_analysis_graph()
        
        # Analyze selected intervals
        all_scores = []
        
        for i, interval in enumerate(selected_intervals, 1):
            print(f"\n--- Analyzing interval {interval} ({i}/{len(selected_intervals)}) ---")
            
            # Create initial state
            state = MultiModalAnalysisState(
                video_id=video_id,
                time_interval=interval,
                interval_data_dir=interval_data_dir
            )
            
            # Run workflow
            final_state = workflow.invoke(state)
            
            # Handle both dict and state object returns
            if isinstance(final_state, dict):
                multi_modal_score = final_state.get('multi_modal_score')
                errors = final_state.get('errors', [])
            else:
                multi_modal_score = final_state.multi_modal_score
                errors = final_state.errors
            
            # Store result
            if multi_modal_score:
                if hasattr(multi_modal_score, 'dict'):
                    score_data = multi_modal_score.dict()
                else:
                    score_data = multi_modal_score
                all_scores.append(score_data)
            
            if errors:
                print(f"Errors during analysis: {errors}")
        
        # Sort by combined score
        all_scores.sort(key=lambda x: x['combined_score'], reverse=True)
        top_highlights = all_scores[:top_n]
        
        # Save results
        output_path = os.path.join(interval_data_dir, "multi_modal_highlights.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'video_id': video_id,
                'top_highlights': top_highlights,
                'all_scores': all_scores,
                'analysis_summary': {
                    'total_intervals': len(chat_activity),
                    'analyzed_intervals': len(all_scores),
                    'selection_efficiency': f"{len(all_scores)}/{len(chat_activity)} ({100*len(all_scores)/len(chat_activity):.1f}%)",
                    'top_score': top_highlights[0]['combined_score'] if top_highlights else 0.0,
                    'avg_score': sum(s['combined_score'] for s in all_scores) / len(all_scores) if all_scores else 0.0
                }
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Smart multi-modal analysis complete!")
        print(f"📊 Analyzed {len(all_scores)} intervals (selected from {len(chat_activity)} total)")
        print(f"⚡ Efficiency: {100*len(all_scores)/len(chat_activity):.1f}% of intervals analyzed")
        print(f"🏆 Top score: {top_highlights[0]['combined_score']:.3f}" if top_highlights else "🏆 No highlights found")
        print(f"💾 Results saved to: {output_path}")
        
        return {
            'video_id': video_id,
            'top_highlights': top_highlights,
            'output_file': output_path,
            'total_intervals_analyzed': len(all_scores),
            'total_intervals_available': len(chat_activity),
            'selection_efficiency': len(all_scores) / len(chat_activity) if chat_activity else 0
        }
        
    except Exception as e:
        print(f"❌ Error during smart multi-modal analysis: {e}")
        return {
            'video_id': video_id,
            'error': str(e),
            'top_highlights': [],
            'total_intervals_analyzed': 0
        }


def smart_interval_selection(chat_activity: Dict, interval_data_dir: str) -> List[str]:
    """
    Smart interval selection based on chat engagement with fallbacks for low-chat streams.
    
    Args:
        chat_activity: Dictionary of intervals and their chat message counts
        interval_data_dir: Directory containing emotion and laugh data
    
    Returns:
        List of selected intervals to analyze
    """
    # Sort intervals by chat activity
    sorted_chat = sorted(chat_activity.items(), key=lambda x: x[1], reverse=True)
    total_intervals = len(sorted_chat)
    
    print(f"📊 Chat activity analysis:")
    print(f"   Total intervals: {total_intervals}")
    print(f"   Chat range: {sorted_chat[0][1]} to {sorted_chat[-1][1]} messages")
    
    # Strategy 1: High chat engagement streams
    top_10_percent = max(1, int(total_intervals * 0.1))
    high_chat_intervals = [interval for interval, count in sorted_chat[:top_10_percent] if count > 0]
    
    print(f"   Top 10% intervals: {len(high_chat_intervals)} (threshold: {sorted_chat[top_10_percent-1][1] if sorted_chat else 0} messages)")
    
    # Strategy 2: Low chat engagement fallback
    if len(high_chat_intervals) < 5 or (sorted_chat[0][1] if sorted_chat else 0) < 10:
        print(f"🔄 Low chat engagement detected, using fallback strategy...")
        return low_chat_fallback_selection(chat_activity, interval_data_dir, top_10_percent)
    
    # Strategy 3: Filter high chat intervals by emotion and laugh signals
    selected_intervals = filter_by_multimodal_signals(high_chat_intervals, interval_data_dir)
    
    print(f"✅ Selected {len(selected_intervals)} intervals for analysis")
    return selected_intervals


def low_chat_fallback_selection(chat_activity: Dict, interval_data_dir: str, min_intervals: int) -> List[str]:
    """
    Fallback selection for streams with low chat engagement.
    Looks for any signals from emotions and laughs.
    """
    print(f"🎭 Low-chat fallback: Looking for emotion and laugh signals...")
    
    candidate_intervals = set()
    
    # Add any intervals with emotions (not "no_face")
    emotions_path = os.path.join(interval_data_dir, 'video_emotions.json')
    if os.path.exists(emotions_path):
        with open(emotions_path, 'r', encoding='utf-8') as f:
            emotions_data = json.load(f)
        
        emotion_intervals = [interval for interval, emotion in emotions_data.items() 
                           if emotion != "no_face" and emotion != ""]
        candidate_intervals.update(emotion_intervals)
        print(f"   Found {len(emotion_intervals)} intervals with face emotions")
    
    # Add any intervals with laughs
    laughs_path = os.path.join(interval_data_dir, 'audio_laughs.json')
    if os.path.exists(laughs_path):
        with open(laughs_path, 'r', encoding='utf-8') as f:
            laughs_data = json.load(f)
        
        laugh_intervals = [interval for interval, has_laugh in laughs_data.items() 
                          if has_laugh is True]
        candidate_intervals.update(laugh_intervals)
        print(f"   Found {len(laugh_intervals)} intervals with detected laughs")
    
    # If still not enough, add top chat intervals regardless of count
    if len(candidate_intervals) < min_intervals:
        sorted_chat = sorted(chat_activity.items(), key=lambda x: x[1], reverse=True)
        top_chat_intervals = [interval for interval, _ in sorted_chat[:min_intervals]]
        candidate_intervals.update(top_chat_intervals)
        print(f"   Added top {min_intervals} chat intervals as minimum baseline")
    
    return list(candidate_intervals)


def filter_by_multimodal_signals(chat_intervals: List[str], interval_data_dir: str) -> List[str]:
    """
    Filter high-chat intervals by emotion and laugh signals to avoid analyzing empty intervals.
    """
    filtered_intervals = []
    
    # Load emotion and laugh data
    emotions_data = {}
    laughs_data = {}
    
    emotions_path = os.path.join(interval_data_dir, 'video_emotions.json')
    if os.path.exists(emotions_path):
        with open(emotions_path, 'r', encoding='utf-8') as f:
            emotions_data = json.load(f)
    
    laughs_path = os.path.join(interval_data_dir, 'audio_laughs.json')
    if os.path.exists(laughs_path):
        with open(laughs_path, 'r', encoding='utf-8') as f:
            laughs_data = json.load(f)
    
    for interval in chat_intervals:
        # Check if interval has meaningful signals
        has_emotion = emotions_data.get(interval, "no_face") not in ["no_face", ""]
        has_laugh = laughs_data.get(interval, False) is True
        
        # Include interval if it has chat + (emotion OR laugh OR is in top chat regardless)
        # This ensures we don't miss high-chat moments even without other signals
        if has_emotion or has_laugh:
            filtered_intervals.append(interval)
        # Always include if it's a very high chat activity (to not miss chat-only highlights)
        else:
            filtered_intervals.append(interval)  # Keep all high-chat intervals for now
    
    print(f"🔍 Filtered intervals:")
    print(f"   With emotions: {sum(1 for i in chat_intervals if emotions_data.get(i, 'no_face') not in ['no_face', ''])}")
    print(f"   With laughs: {sum(1 for i in chat_intervals if laughs_data.get(i, False) is True)}")
    print(f"   Total selected: {len(filtered_intervals)}")
    
    return filtered_intervals