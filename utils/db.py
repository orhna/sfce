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
    
    # Composite primary key: video_id + interval_seconds
    video_id = Column(String, primary_key=True)
    interval_seconds = Column(Integer, primary_key=True)
    
    # Basic video info (same for all intervals)
    vod_url = Column(String)
    video_path = Column(String)  # Base video file (same for all intervals)
    audio_path = Column(String)  # Base audio file (same for all intervals)
    chat_csv_path = Column(String)  # Base chat CSV (same for all intervals)
    
    # Interval-specific paths
    chat_json_path = Column(String)  # /twitch/VIDEO_ID/INTERVAL/chat_activity.json
    chat_transcript_path = Column(String)  # /twitch/VIDEO_ID/INTERVAL/chat_transcriptions.json
    video_emotions_path = Column(String)  # /twitch/VIDEO_ID/INTERVAL/video_emotions.json
    audio_laughs_path = Column(String)  # /twitch/VIDEO_ID/INTERVAL/audio_laughs.json
    highlight_descriptions_path = Column(String)  # /twitch/VIDEO_ID/INTERVAL/highlights/highlight_descriptions.json
    highlights_dir = Column(String)  # /twitch/VIDEO_ID/INTERVAL/highlights/
    
    processed_at = Column(DateTime, default=datetime.datetime.utcnow)

def create_tables():
    """Create all tables. Call this explicitly to set up the database."""
    Base.metadata.create_all(engine)
    print("Database tables created/updated successfully")

def setup_database():
    """Setup database tables."""
    print("🔧 Setting up database...")
    try:
        create_tables()
        print("💾 Database setup complete!")
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        raise

def extract_video_id_from_url(vod_url):
    """Extract video ID from Twitch VOD URL"""
    match = re.search(r'/videos/(\d+)', vod_url)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Could not extract video ID from URL: {vod_url}")

def add_or_update_video(vod_url, interval_seconds, video_path=None, audio_path=None, 
                       chat_csv_path=None, chat_json_path=None, chat_transcript_path=None, 
                       video_emotions_path=None, audio_laughs_path=None, 
                       highlight_descriptions_path=None, highlights_dir=None):
    """Add new video interval or update existing video interval with new paths"""
    session = Session()
    try:
        video_id = extract_video_id_from_url(vod_url)
        
        # Check if video+interval combination already exists
        existing_video = session.query(Video).filter_by(
            video_id=video_id, 
            interval_seconds=interval_seconds
        ).first()
        
        if existing_video:
            # Update existing video interval
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
            if highlights_dir:
                existing_video.highlights_dir = highlights_dir
            existing_video.processed_at = datetime.datetime.utcnow()
            print(f"Updated video {video_id} (interval: {interval_seconds}s) in database")
        else:
            # Create new video interval entry
            video = Video(
                video_id=video_id,
                interval_seconds=interval_seconds,
                vod_url=vod_url,
                video_path=video_path,
                audio_path=audio_path,
                chat_csv_path=chat_csv_path,
                chat_json_path=chat_json_path,
                chat_transcript_path=chat_transcript_path,
                video_emotions_path=video_emotions_path,
                audio_laughs_path=audio_laughs_path,
                highlight_descriptions_path=highlight_descriptions_path,
                highlights_dir=highlights_dir
            )
            session.add(video)
            print(f"Added new video {video_id} (interval: {interval_seconds}s) to database")
        
        session.commit()
        return video_id
    except Exception as e:
        session.rollback()
        print(f"Error adding/updating video: {e}")
        raise
    finally:
        session.close()

def get_video_paths(video_id, interval_seconds=None):
    """Get all file paths for a video ID and specific interval, or base paths if no interval specified"""
    session = Session()
    try:
        if interval_seconds is not None:
            # Get specific interval data
            video = session.query(Video).filter_by(
                video_id=video_id, 
                interval_seconds=interval_seconds
            ).first()
        else:
            # Get any entry for this video (prefer smallest interval for base paths)
            video = session.query(Video).filter_by(video_id=video_id).order_by(Video.interval_seconds).first()
        
        if video:
            return {
                'video_id': video.video_id,
                'interval_seconds': video.interval_seconds,
                'vod_url': video.vod_url,
                'video_path': video.video_path,
                'audio_path': video.audio_path,
                'chat_csv_path': video.chat_csv_path,
                'chat_json_path': video.chat_json_path,
                'chat_transcript_path': video.chat_transcript_path,
                'video_emotions_path': video.video_emotions_path,
                'audio_laughs_path': video.audio_laughs_path,
                'highlight_descriptions_path': video.highlight_descriptions_path,
                'highlights_dir': video.highlights_dir,
                'processed_at': video.processed_at
            }
        else:
            return None
    finally:
        session.close()

def get_video_intervals(video_id):
    """Get all intervals available for a video ID"""
    session = Session()
    try:
        videos = session.query(Video).filter_by(video_id=video_id).order_by(Video.interval_seconds).all()
        return [{
            'interval_seconds': v.interval_seconds,
            'chat_json_path': v.chat_json_path,
            'video_emotions_path': v.video_emotions_path,
            'audio_laughs_path': v.audio_laughs_path,
            'highlights_dir': v.highlights_dir,
            'processed_at': v.processed_at
        } for v in videos]
    finally:
        session.close()

def get_video_path(video_id):
    """Get video file path for a video ID (returns base video path)"""
    paths = get_video_paths(video_id)
    return paths['video_path'] if paths else None

def get_audio_path(video_id):
    """Get audio file path for a video ID (returns base audio path)"""
    paths = get_video_paths(video_id)
    return paths['audio_path'] if paths else None

def get_chat_paths(video_id, interval_seconds=None):
    """Get chat file paths for a video ID and specific interval"""
    paths = get_video_paths(video_id, interval_seconds)
    if paths:
        return {
            'chat_csv_path': paths['chat_csv_path'],
            'chat_json_path': paths['chat_json_path'],
            'chat_transcript_path': paths['chat_transcript_path']
        }
    return None

def update_file_path(video_id, interval_seconds, **file_paths):
    """Update any file paths for a video and specific interval"""
    session = Session()
    try:
        video = session.query(Video).filter_by(
            video_id=video_id, 
            interval_seconds=interval_seconds
        ).first()
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
            if 'highlights_dir' in file_paths:
                video.highlights_dir = file_paths['highlights_dir']
            
            session.commit()
            print(f"Updated file paths for video {video_id} (interval: {interval_seconds}s): {list(file_paths.keys())}")
        else:
            print(f"Video {video_id} with interval {interval_seconds}s not found in database")
    finally:
        session.close()

def update_chat_json_path(video_id, chat_json_path, interval_seconds=5):
    """Update the chat JSON path for a video (kept for backwards compatibility)"""
    update_file_path(video_id, interval_seconds, chat_json_path=chat_json_path)

def list_all_videos():
    """List all videos in the database with their intervals"""
    session = Session()
    try:
        videos = session.query(Video).order_by(Video.video_id, Video.interval_seconds).all()
        return [{
            'video_id': v.video_id,
            'interval_seconds': v.interval_seconds,
            'vod_url': v.vod_url,
            'video_path': v.video_path,
            'audio_path': v.audio_path,
            'chat_csv_path': v.chat_csv_path,
            'chat_json_path': v.chat_json_path,
            'chat_transcript_path': v.chat_transcript_path,
            'video_emotions_path': v.video_emotions_path,
            'audio_laughs_path': v.audio_laughs_path,
            'highlight_descriptions_path': v.highlight_descriptions_path,
            'highlights_dir': v.highlights_dir,
            'processed_at': v.processed_at
        } for v in videos]
    finally:
        session.close()

def migrate_existing_data():
    """Migrate existing single-interval entries to new schema with default interval_seconds=5"""
    session = Session()
    try:
        # This function helps migrate old data structure to new composite key structure
        # Run this once after updating the schema
        print("Note: If you have existing data, you may need to manually migrate it.")
        print("The new schema requires interval_seconds to be specified for all entries.")
        print("Consider running: ALTER TABLE videos ADD COLUMN interval_seconds INTEGER DEFAULT 5;")
        print("Then: UPDATE videos SET interval_seconds = 5 WHERE interval_seconds IS NULL;")
    finally:
        session.close()

def reset_database():
    """Reset the database by deleting all entries from all tables"""
    session = Session()
    try:
        # Count existing entries before deletion
        video_count = session.query(Video).count()
        
        if video_count == 0:
            print("🔄 Database is already empty - no entries to reset.")
            return
        
        print(f"🔄 Resetting database... Found {video_count} video entries to delete.")
        
        # Delete all video entries
        session.query(Video).delete()
        session.commit()
        
        print(f"✅ Database reset complete! Removed {video_count} video entries.")
        
    except Exception as e:
        session.rollback()
        print(f"❌ Error resetting database: {e}")
        raise
    finally:
        session.close()