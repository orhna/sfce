import yt_dlp
from pydub import AudioSegment
import os
from chat_downloader import ChatDownloader
from datetime import datetime
import json
import csv
import re
from collections import Counter, OrderedDict
    
def download_twitch_vod(vod_url, output_dir="/mnt/d/Projects/twitch_videos"):
    """
    Download a Twitch VOD in 1080p (if available), extract 16kHz audio as WAV, and save chat logs.
    
    Args:
        vod_url (str): URL of the Twitch VOD
        output_dir (str): Directory to save the files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract video ID from URL for naming
    video_id = vod_url.split('/')[-1]
    base_filename = f"twitch_{video_id}"
    
    # Configure yt-dlp options for 1080p
    ydl_opts = {
        'format': 'bestvideo[height<=1080]+bestaudio/best[height<=1080]',
        'outtmpl': os.path.join(output_dir, f'{base_filename}.%(ext)s'),
        'quiet': False,
        # Format selection to prefer 1080p
        'format_sort': ['res:1080'],
    }
    
    try:
        # Download video
        print("Downloading video...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(vod_url, download=True)
            video_file = ydl.prepare_filename(info)
        
        # Convert video to WAV with 16kHz
        print("Extracting audio and converting to 16kHz...")
        video = AudioSegment.from_file(video_file)
        # Convert to mono and set sample rate to 16kHz
        audio = video.set_channels(1).set_frame_rate(16000)
        wav_path = os.path.join(output_dir, f"{base_filename}.wav")
        audio.export(wav_path, format="wav", parameters=["-ar", "16000"])
        
        # Download chat
        print("Downloading chat...")
        chat_path = os.path.join(output_dir, f"{base_filename}_chat.txt")
        chat_downloader = ChatDownloader()
        chat = chat_downloader.get_chat(vod_url)
        
        with open(chat_path, 'w', encoding='utf-8') as f:
            for message in chat:
                timestamp = datetime.fromtimestamp(message['timestamp'] / 1000).strftime('%H:%M:%S')
                author = message.get('author', {}).get('name', 'Anonymous')
                content = message.get('message', '')
                f.write(f"[{timestamp}] {author}: {content}\n")
        
        print(f"Successfully processed VOD {video_id}")
        print(f"Files saved in {output_dir}:")
        print(f"- Video: {os.path.basename(video_file)} (1080p or best available)")
        print(f"- Audio: {base_filename}.wav (16kHz mono)")
        print(f"- Chat: {base_filename}_chat.txt")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Example usage:
# download_twitch_vod("https://www.twitch.tv/videos/YOUR_VOD_ID")

def download_chat(vod_url, chat_path = "/mnt/d/Projects/twitch_videos/chat.txt"):
    """
    Download the chat for a Twitch VOD and save it to a CSV file.
    
    Args:
        vod_url (str): URL of the Twitch VOD
        output_dir (str): Directory to save the files
    """
    
    print("Downloading chat...")
    chat_downloader = ChatDownloader()
    chat = chat_downloader.get_chat(vod_url)

    chat_messages = []
    print("Collecting chat messages...")
    for message in chat:
        chat_messages.append(message)

    # Write messages to CSV
    print("Writing chat messages to CSV...")
    with open(chat_path, 'w', newline='', encoding='utf-8') as f:
        csv_writer = csv.writer(f)
        # Write header
        csv_writer.writerow(['Timestamp', 'Author', 'Message', 'Badges', 'Subscriber', 'Moderator'])
        
        # Write messages
        for message in chat_messages:
            try:
                # Get message details
                author = message.get('author', {}).get('name', 'Anonymous')
                content = message.get('message', '')
                time_text = message.get("time_text", "")
                badges = [badge.get('title', '') for badge in message.get('author', {}).get('badges', [])]
                badges_str = ','.join(badges) if badges else ''
                
                # Check if user is subscriber or moderator
                is_subscriber = any('Subscriber' in badge for badge in badges)
                is_moderator = any('Moderator' in badge for badge in badges)
                
                # Write to CSV
                csv_writer.writerow([
                    time_text,
                    author,
                    content,
                    badges_str,
                    is_subscriber,
                    is_moderator
                ])
            except Exception as e:
                print(f"Error processing message: {str(e)}")
                continue

def chat_message_activity_to_json(vod_url, interval_seconds=5, output_json_path="/mnt/d/Projects/twitch_videos/chat_activity.json"):
    """
    Count how many messages appear in each timeframe and save as JSON, ordered by count descending.
    Args:
        chat_messages (list): List of chat message dicts (must include 'time_text')
        interval_seconds (int): Size of each timeframe in seconds
        output_json_path (str): Path to save the JSON output
    Output JSON format:
        {
            "0:00-0:05": 3,
            "0:05-0:10": 7,
            ...
        }
    """
    
    def time_text_to_seconds(time_text):
        # Supports HH:MM:SS or MM:SS
        parts = [int(p) for p in re.split(r":", time_text)]
        if len(parts) == 3:
            return parts[0]*3600 + parts[1]*60 + parts[2]
        elif len(parts) == 2:
            return parts[0]*60 + parts[1]
        return 0

    print("Downloading chat...")
    chat_downloader = ChatDownloader()
    chat = chat_downloader.get_chat(vod_url)

    chat_messages = []
    print("Collecting chat messages...")
    for message in chat:
        chat_messages.append(message)

    # Collect all message times in seconds
    times = []
    for msg in chat_messages:
        t = msg.get('time_text', None)
        if t:
            times.append(time_text_to_seconds(t))
    if not times:
        print("No valid time_text found in chat messages.")
        return

    # Bin messages
    bins = Counter()
    for t in times:
        bin_start = (t // interval_seconds) * interval_seconds
        bins[bin_start] += 1

    # Prepare results as dict, sorted by count descending
    bin_items = []
    for bin_start in bins:
        bin_end = bin_start + interval_seconds
        m_start = f"{bin_start//60}:{bin_start%60:02d}"
        m_end = f"{bin_end//60}:{bin_end%60:02d}"
        key = f"{m_start}-{m_end}"
        bin_items.append((key, bins[bin_start]))
    bin_items.sort(key=lambda x: x[1], reverse=True)
    results = OrderedDict(bin_items)

    # Save to JSON
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Chat activity saved to {output_json_path}")
