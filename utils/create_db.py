from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

Base = declarative_base()
engine = create_engine('postgresql://aorhu:Avatar61!@localhost:5432/sfce')
Session = sessionmaker(bind=engine)


class Video(Base):
    __tablename__ = 'videos'
    id = Column(Integer, primary_key=True)
    vod_url = Column(String, unique=True)
    video_path = Column(String)
    processed_at = Column(DateTime, default=datetime.datetime.utcnow)

class Audio(Base):
    __tablename__ = 'audios'
    id = Column(Integer, primary_key=True)
    video_id = Column(Integer, ForeignKey('videos.id'))
    audio_path = Column(String)
    processed_at = Column(DateTime, default=datetime.datetime.utcnow)

class Chat(Base):
    __tablename__ = 'chats'
    id = Column(Integer, primary_key=True)
    video_id = Column(Integer, ForeignKey('videos.id'))
    chat_txt_path = Column(String)
    chat_json_path = Column(String)
    processed_at = Column(DateTime, default=datetime.datetime.utcnow)

# Replace with your actual connection string
#engine = create_engine('postgresql://aorhu:Avatar61!@localhost:5432/sfce')

# This will create all tables defined by Base subclasses if they don't exist
#Base.metadata.create_all(engine)

def add_video(vod_url, video_path):
    session = Session()
    video = Video(vod_url=vod_url, video_path=video_path)
    session.add(video)
    session.commit()
    print(f"Added video: {vod_url}")
    session.close()
    return video.id  # Return the new video's ID for linking

def add_audio(video_id, audio_path):
    session = Session()
    audio = Audio(video_id=video_id, audio_path=audio_path)
    session.add(audio)
    session.commit()
    print(f"Added audio for video_id: {video_id}")
    session.close()
    return audio.id

def add_chat(video_id, chat_txt_path, chat_json_path):
    session = Session()
    chat = Chat(video_id=video_id, chat_txt_path=chat_txt_path, chat_json_path=chat_json_path)
    session.add(chat)
    session.commit()
    print(f"Added chat for video_id: {video_id}")
    session.close()
    return chat.id