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

# Option 2: Local Ollama (free, runs on your machine)
# Leave OPENAI_API_KEY empty/commented to use Ollama
# Install Ollama: https://ollama.ai/download
# Run: ollama pull llama3.2:3b
OLLAMA_MODEL = 'llama3.2:3b'  # Can be: llama3.2:3b, llama3.2:1b, llama3.1:8b
OLLAMA_BASE_URL = 'http://localhost:11434'  # Default Ollama server

# You can add other configuration settings here as needed:
# TWITCH_CLIENT_ID = 'your_twitch_client_id_here'
# TWITCH_CLIENT_SECRET = 'your_twitch_client_secret_here'
# DEFAULT_DOWNLOAD_PATH = '/path/to/your/download/directory' 