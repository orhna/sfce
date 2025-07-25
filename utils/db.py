from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
import re
import os
import sys

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import DATABASE_URL
except ImportError:
    print("Error: config.py not found!")
    print("Please copy config.example.py to config.py and update your database settings.")
    print("Example: cp config.example.py config.py")
    sys.exit(1)

Base = declarative_base()
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

class Video(Base):
    __tablename__ = 'videos'
    video_id = Column(String, primary_key=True)  # Changed to string primary key
    vod_url = Column(String, unique=True)
    video_path = Column(String)
    audio_path = Column(String)
    chat_csv_path = Column(String)  # Updated from chat_txt_path to chat_csv_path
    chat_json_path = Column(String)  # chat_activity.json
    chat_transcript_path = Column(String)  # chat_transcript.json
    video_emotions_path = Column(String)  # video_emotions.json
    audio_laughs_path = Column(String)  # audio_laughs.json
    highlight_descriptions_path = Column(String)  # highlight_descriptions.json
    processed_at = Column(DateTime, default=datetime.datetime.utcnow)

def create_tables():
    """Create all tables. Call this explicitly to set up the database."""
    Base.metadata.create_all(engine)
    print("Database tables created/updated successfully")

def extract_video_id_from_url(vod_url):
    """Extract video ID from Twitch VOD URL"""
    match = re.search(r'/videos/(\d+)', vod_url)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Could not extract video ID from URL: {vod_url}")

def add_or_update_video(vod_url, video_path=None, audio_path=None, chat_csv_path=None, 
                       chat_json_path=None, chat_transcript_path=None, video_emotions_path=None, 
                       audio_laughs_path=None, highlight_descriptions_path=None):
    """Add new video or update existing video with new paths"""
    session = Session()
    try:
        video_id = extract_video_id_from_url(vod_url)
        
        # Check if video already exists
        existing_video = session.query(Video).filter_by(video_id=video_id).first()
        
        if existing_video:
            # Update existing video
            if video_path:
                existing_video.video_path = video_path
            if audio_path:
                existing_video.audio_path = audio_path
            if chat_csv_path:
                existing_video.chat_csv_path = chat_csv_path
            if chat_json_path:
                existing_video.chat_json_path = chat_json_path
            if chat_transcript_path:
                existing_video.chat_transcript_path = chat_transcript_path
            if video_emotions_path:
                existing_video.video_emotions_path = video_emotions_path
            if audio_laughs_path:
                existing_video.audio_laughs_path = audio_laughs_path
            if highlight_descriptions_path:
                existing_video.highlight_descriptions_path = highlight_descriptions_path
            existing_video.processed_at = datetime.datetime.utcnow()
            print(f"Updated video {video_id} in database")
        else:
            # Create new video
            video = Video(
                video_id=video_id,
                vod_url=vod_url,
                video_path=video_path,
                audio_path=audio_path,
                chat_csv_path=chat_csv_path,
                chat_json_path=chat_json_path,
                chat_transcript_path=chat_transcript_path,
                video_emotions_path=video_emotions_path,
                audio_laughs_path=audio_laughs_path,
                highlight_descriptions_path=highlight_descriptions_path
            )
            session.add(video)
            print(f"Added new video {video_id} to database")
        
        session.commit()
        return video_id
    except Exception as e:
        session.rollback()
        print(f"Error adding/updating video: {e}")
        raise
    finally:
        session.close()

def get_video_paths(video_id):
    """Get all file paths for a video ID"""
    session = Session()
    try:
        video = session.query(Video).filter_by(video_id=video_id).first()
        if video:
            return {
                'video_id': video.video_id,
                'vod_url': video.vod_url,
                'video_path': video.video_path,
                'audio_path': video.audio_path,
                'chat_csv_path': video.chat_csv_path,
                'chat_json_path': video.chat_json_path,
                'chat_transcript_path': video.chat_transcript_path,
                'video_emotions_path': video.video_emotions_path,
                'audio_laughs_path': video.audio_laughs_path,
                'highlight_descriptions_path': video.highlight_descriptions_path,
                'processed_at': video.processed_at
            }
        else:
            return None
    finally:
        session.close()

def get_video_path(video_id):
    """Get video file path for a video ID"""
    paths = get_video_paths(video_id)
    return paths['video_path'] if paths else None

def get_audio_path(video_id):
    """Get audio file path for a video ID"""
    paths = get_video_paths(video_id)
    return paths['audio_path'] if paths else None

def get_chat_paths(video_id):
    """Get chat file paths for a video ID"""
    paths = get_video_paths(video_id)
    if paths:
        return {
            'chat_csv_path': paths['chat_csv_path'],
            'chat_json_path': paths['chat_json_path'],
            'chat_transcript_path': paths['chat_transcript_path']
        }
    return None

def update_file_path(video_id, **file_paths):
    """Update any file paths for a video"""
    session = Session()
    try:
        video = session.query(Video).filter_by(video_id=video_id).first()
        if video:
            # Update any provided paths
            if 'chat_json_path' in file_paths:
                video.chat_json_path = file_paths['chat_json_path']
            if 'chat_transcript_path' in file_paths:
                video.chat_transcript_path = file_paths['chat_transcript_path']
            if 'video_emotions_path' in file_paths:
                video.video_emotions_path = file_paths['video_emotions_path']
            if 'audio_laughs_path' in file_paths:
                video.audio_laughs_path = file_paths['audio_laughs_path']
            if 'highlight_descriptions_path' in file_paths:
                video.highlight_descriptions_path = file_paths['highlight_descriptions_path']
            
            session.commit()
            print(f"Updated file paths for video {video_id}: {list(file_paths.keys())}")
        else:
            print(f"Video {video_id} not found in database")
    finally:
        session.close()

def update_chat_json_path(video_id, chat_json_path):
    """Update the chat JSON path for a video (kept for backwards compatibility)"""
    update_file_path(video_id, chat_json_path=chat_json_path)

def list_all_videos():
    """List all videos in the database"""
    session = Session()
    try:
        videos = session.query(Video).all()
        return [{
            'video_id': v.video_id,
            'vod_url': v.vod_url,
            'video_path': v.video_path,
            'audio_path': v.audio_path,
            'chat_csv_path': v.chat_csv_path,
            'chat_json_path': v.chat_json_path,
            'chat_transcript_path': v.chat_transcript_path,
            'video_emotions_path': v.video_emotions_path,
            'audio_laughs_path': v.audio_laughs_path,
            'highlight_descriptions_path': v.highlight_descriptions_path,
            'processed_at': v.processed_at
        } for v in videos]
    finally:
        session.close()