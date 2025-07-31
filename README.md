# Short Form Content Extraction (SFCExtraction)

An AI-powered Python project for automatically extracting highlights from stream recordings using multi-modal analysis: chat activity, video emotion detection, audio laugh recognition, and LLM-powered content understanding. Currently supports only Twitch.

## 🚀 What It Does

This project processes Twitch VODs to automatically identify and extract the most engaging moments by:

1. **Multi-Modal Analysis**: Combines chat engagement, facial emotions, and audio laughs
2. **Smart Interval Selection**: Prioritizes high-chat activity periods with fallback for low-engagement streams
3. **AI-Powered Descriptions**: Generates contextual highlight descriptions using LangChain/LangGraph
4. **Video Extraction**: Creates standalone MP4 clips with configurable pre-padding
5. **Database Tracking**: PostgreSQL integration for managing processed content and metadata

## 🛠 Installation

### 1. Prerequisites

- **Python 3.8+**
- **PostgreSQL** (for metadata storage)
- **FFmpeg** (for video processing)
- **GPU recommended** (for faster AI model inference)

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt update
sudo apt install postgresql postgresql-contrib ffmpeg

# Create database
sudo -u postgres createdb sfce
```

### 2. Python Dependencies

```bash
# Clone and navigate to project
git clone <repository-url>
cd SFCExtraction

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Configuration Setup


Edit `config.py` with your settings:


### 4. LLM Setup

**Option A: Local Ollama**
```bash
# Automated setup
python setup_local_llm.py

# Or manual setup
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen2.5:3b
ollama serve
```

**Option B: Cloud APIs**
- OpenAI: Get API key from [platform.openai.com](https://platform.openai.com)
- Qwen: Get API key from [DashScope Console](https://dashscope.console.aliyun.com/)

## 📊 Pipeline Overview

```
VOD URL → Download → Extract Audio → Chat Analysis
    ↓           ↓              ↓            ↓
Video File → Emotion Analysis → Audio Laughs → Chat Activity
    ↓           ↓              ↓            ↓
Multi-Modal Analysis (LangGraph) → Smart Interval Selection
    ↓
Highlight Extraction → AI Descriptions → MP4 Clips + JSON
```

### Processing Steps:

1. **Download & Setup**: Downloads Twitch VOD, extracts WAV audio, saves chat as CSV
2. **Interval-Based Analysis**: 
   - Chat activity counting (configurable intervals, default 5s)
   - Video emotion detection using MediaPipe + HSEmotion
   - Audio laugh detection using YAMNet
   - Audio transcription with Whisper for high-activity segments
3. **Smart Selection**: Prioritizes top 10% chat intervals, with fallback for emotion/laugh signals
4. **Multi-Modal Scoring**: LangGraph workflow combines all signals for highlight ranking
5. **Video Extraction**: Creates MP4 clips with pre-padding and merges sequential intervals
6. **AI Descriptions**: Generates contextual descriptions using chat + transcript analysis

## 🎯 Usage

### Basic Usage
```bash
# Process a Twitch VOD with default settings
python main.py https://www.twitch.tv/videos/2518537772

# Custom parameters: 10s intervals, extract 15 highlights, 3s pre-padding
python main.py https://www.twitch.tv/videos/2518537772 10 15 3

```

### Parameters

- `vod_url`: Twitch VOD URL (required)
- `interval_seconds`: Analysis window in seconds (default: 5)
- `top_highlights`: Number of highlights to extract (default: 10)
- `pre_padding_seconds`: Seconds to add before each highlight (default: 5)
- `--reset-db`: Reset database before processing

### Output Structure

```
/../Projects/twitch/
└── {video_id}/
    ├── video_id.mp4              # Original video
    ├── video_id.wav              # Extracted audio
    ├── video_id_chat.csv         # Chat messages
    └── {interval}/               # Per-interval analysis
        ├── chat_activity.json    # Message counts per timeframe
        ├── video_emotions.json   # Facial emotion analysis
        ├── audio_laughs.json     # Laugh detection results
        ├── chat_transcriptions.json  # Audio transcripts
        ├── highlight_descriptions.json  # AI-generated descriptions
        └── highlights/           # Extracted video clips
            ├── highlight_1.mp4
            ├── highlight_2.mp4
            └── ...
```

## 🔧 Features

### Multi-Modal Analysis
- **Chat Activity**: Identifies high-engagement periods from message frequency
- **Video Emotions**: Detects joy, surprise, neutral expressions using MediaPipe face detection
- **Audio Laughs**: Uses TensorFlow YAMNet model to detect laughs and cheering
- **Smart Fallback**: Ensures highlights are found even in low-chat streams

### AI-Powered Descriptions
- **LangGraph Workflow**: Multi-step LLM analysis combining chat context and audio transcripts
- **Multiple LLM Support**: OpenAI GPT, Qwen models, or local Ollama
- **Contextual Understanding**: Generates descriptions that understand the content and community

### Database Integration
- **PostgreSQL Backend**: Tracks all processed files and metadata
- **Composite Keys**: Supports multiple analysis intervals per video
- **Path Management**: Automatic file organization and database updates

## 📁 Project Structure

```
SFCExtraction/
├── main.py                     # Main entry point
├── config.py                   # Configuration 
├── setup_local_llm.py          # Automated LLM setup
├── requirements.txt            # Python dependencies
└── utils/
    ├── utils.py               # Core workflow functions
    ├── db.py                  # Database models and functions
    ├── segment_analysis.py    # LangGraph multi-modal analysis
    ├── video.py               # Video emotion detection
    ├── audio.py               # Audio laugh detection
    └── chat.py                # Chat processing and transcription
```

## 🐛 Troubleshooting

**Database Connection Issues**:
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Reset database if needed
python main.py <vod_url> --reset-db
```

**LLM Issues**:
```bash
# Check Ollama status
ollama list
ollama serve

# Test model
ollama run qwen2.5:3b "Hello"
```

**Video Processing Slow**:
- Ensure FFmpeg is installed and in PATH
- Use SSD storage for temp files
- Consider smaller interval sizes for initial testing 