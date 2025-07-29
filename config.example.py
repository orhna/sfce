"""
Example configuration file for SFCExtraction project.

Copy this file to 'config.py' and update the values according to your environment.
"""

# Database Configuration
# Replace with your actual PostgreSQL connection string
DATABASE_URL = 'postgresql://username:password@localhost:5432/database_name'

# LLM Configuration
# Option 1: OpenAI (requires API key and credits)
# OPENAI_API_KEY = 'your_openai_api_key_here'

# Option 2: Qwen (Alibaba Cloud) - Good balance of quality and cost
# Get API key from: https://dashscope.console.aliyun.com/
# QWEN_API_KEY = 'your_qwen_api_key_here'
# QWEN_MODEL = 'qwen-plus'  # Options: qwen-plus, qwen-turbo, qwen-max, qwen-long
# QWEN_BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'  # Default endpoint

# Option 3: Local Ollama (free, runs on your machine)
# Leave OPENAI_API_KEY and QWEN_API_KEY empty/commented to use Ollama
# Install Ollama: https://ollama.ai/download
# 
# Install models:
# ollama pull qwen2.5:3b      # Qwen 3B (recommended for highlights)
# ollama pull qwen2.5:1.5b    # Qwen 1.5B (faster)
# ollama pull qwen2.5:7b      # Qwen 7B (higher quality)
# ollama pull llama3.2:3b     # Alternative option
OLLAMA_MODEL = 'qwen2.5:3b'  # Can be: qwen2.5:3b, qwen2.5:1.5b, qwen2.5:7b, llama3.2:3b
OLLAMA_BASE_URL = 'http://localhost:11434'  # Default Ollama server

# Priority order: OpenAI -> Qwen -> Ollama
# The system will try each option in order until it finds a working configuration

# You can add other configuration settings here as needed:
# TWITCH_CLIENT_ID = 'your_twitch_client_id_here'
# TWITCH_CLIENT_SECRET = 'your_twitch_client_secret_here'
# DEFAULT_DOWNLOAD_PATH = '/path/to/your/download/directory' 