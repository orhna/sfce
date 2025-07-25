# Short Form Content Extraction (SFCExtraction)

A Python project for extracting highlights from stream recordings using video emotion analysis, audio laugh detection, and chat activity analysis.

## Quick Setup

### 1. Configuration Setup

Before running the project, you need to set up your configuration:

```bash
# Copy the example configuration file
cp config.example.py config.py
```

Then edit `config.py` and update the database connection string:

```python
# Replace with your actual PostgreSQL connection details
DATABASE_URL = 'postgresql://username:password@localhost:5432/database_name'

# LLM Configuration (for AI highlight generation)
# Option 1: OpenAI (requires API key)
# OPENAI_API_KEY = 'your_openai_api_key_here'

# Option 2: Local Ollama (free, recommended)
OLLAMA_MODEL = 'llama3.2:3b'
OLLAMA_BASE_URL = 'http://localhost:11434'
```

### 2. Database Setup

Make sure you have PostgreSQL running and create the database:

```sql
CREATE DATABASE sfce;
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Local LLM (Recommended)

For AI-powered highlight generation, you have two options:

**Option A: Local Ollama (Free, Recommended)**
```bash
# Run the automated setup script
python setup_local_llm.py
```

Or manually:
```bash
# Install Ollama from https://ollama.ai/download
# Then download a model:
ollama pull llama3.2:3b

# Start Ollama service
ollama serve
```

**Option B: OpenAI (Requires API Key)**
```bash
# Set your API key in config.py
# OPENAI_API_KEY = 'your-key-here'
```

### 5. Initialize Database

```bash
python example_workflow.py --setup
```

## Usage

### Process a new Twitch VOD:
```bash
python example_workflow.py https://www.twitch.tv/videos/1234567890
```

### Quick analysis on existing video:
```bash
python example_workflow.py --quick 1234567890
```

### Test laugh detection thresholds:
```bash
python example_workflow.py --test-thresholds 1234567890
```

### Generate AI highlight descriptions:
```bash
python example_workflow.py --highlights 1234567890 5
```

### Analyze specific time interval:
```bash
python example_workflow.py --analyze 1234567890 267:00-267:05
```

### List processed videos:
```bash
python example_workflow.py --list
```

## Features

- **Video Emotion Analysis**: Detects emotions from streamer's face using MediaPipe and HSEmotion
- **Audio Laugh Detection**: Uses YAMNet to detect laughs and cheering in audio
- **Chat Activity Analysis**: Analyzes chat message patterns and transcribes high-activity segments
- **Database Integration**: PostgreSQL database for tracking all processed files and metadata

## Project Structure

```
SFCExtraction/
├── config.py              # Your configuration (not in git)
├── config.example.py       # Example configuration
├── example_workflow.py     # Main workflow script
├── utils/
│   ├── db.py              # Database functions
│   ├── utils.py           # Download and utility functions
│   ├── video.py           # Video emotion analysis
│   ├── audio.py           # Audio laugh detection
│   └── chat.py            # Chat processing and transcription
└── README.md              # This file
```

## Important Notes

- `config.py` is excluded from version control for security
- The project downloads and processes large video files
- GPU support is recommended for faster processing
- All processed files are saved under `/mnt/d/Projects/twitch/{video_id}/` 