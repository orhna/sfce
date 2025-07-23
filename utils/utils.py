import yt_dlp
from pydub import AudioSegment
import os
from chat_downloader import ChatDownloader
from datetime import datetime
import json
import csv
import re
import subprocess
from collections import Counter, OrderedDict
from .db import add_or_update_video, extract_video_id_from_url, get_chat_paths, update_chat_json_path
    
def download_twitch_vod(vod_url, base_output_dir="/mnt/d/Projects/twitch"):
    """
    Download a Twitch VOD in 1080p (if available), extract 16kHz audio as WAV, and save chat logs.
    Skips steps if output files already exist.
    
    Args:
        vod_url (str): URL of the Twitch VOD
        base_output_dir (str): Base directory to save files under video_id folder
    
    Returns:
        dict: Information about processed files, or dict with error info if failed
    """
    # Extract video ID from URL
    video_id = vod_url.split('/')[-1]
    
    # Create video-specific directory
    output_dir = os.path.join(base_output_dir, video_id)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing video {video_id} in directory: {output_dir}")
    base_filename = f"twitch_{video_id}"
    
    # Define expected file paths
    wav_path = os.path.join(output_dir, f"{base_filename}.wav")
    chat_path = os.path.join(output_dir, f"{base_filename}_chat.csv")
    
    # Check for existing files
    video_file = None
    video_exists = False
    audio_exists = os.path.exists(wav_path)
    chat_exists = os.path.exists(chat_path)
    
    # Look for existing video file (could have different extensions)
    video_extensions = ['.mp4', '.mkv', '.webm', '.flv', '.m4v']
    for ext in video_extensions:
        potential_video = os.path.join(output_dir, f"{base_filename}{ext}")
        if os.path.exists(potential_video):
            video_file = potential_video
            video_exists = True
            break
    
    print(f"File status check:")
    print(f"- Video: {'✓ Found' if video_exists else '✗ Missing'}")
    print(f"- Audio: {'✓ Found' if audio_exists else '✗ Missing'}")
    print(f"- Chat: {'✓ Found' if chat_exists else '✗ Missing'}")
    
    # Initialize result dictionary with defaults
    result = {
        "video_id": video_id,
        "output_directory": output_dir,
        "video_file": None,
        "audio_file": None,
        "chat_file": None,
        "skipped": {
            "video": video_exists,
            "audio": audio_exists,
            "chat": chat_exists
        },
        "success": False,
        "errors": []
    }
    
    try:
        # Download video if not exists
        if not video_exists:
            print("Downloading video...")
            try:
                ydl_opts = {
                    'format': 'bestvideo[height<=1080]+bestaudio/best[height<=1080]',
                    'outtmpl': os.path.join(output_dir, f'{base_filename}.%(ext)s'),
                    'quiet': False,
                    # Format selection to prefer 1080p
                    'format_sort': ['res:1080'],
                }
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(vod_url, download=True)
                    video_file = ydl.prepare_filename(info)
                print(f"✓ Video downloaded: {os.path.basename(video_file)}")
            except Exception as e:
                error_msg = f"Failed to download video: {str(e)}"
                print(f"✗ {error_msg}")
                result["errors"].append(error_msg)
        else:
            print(f"✓ Video already exists: {os.path.basename(video_file)}")
        
        # Extract audio if not exists
        if not audio_exists:
            if video_file and os.path.exists(video_file):
                print("Extracting audio and converting to 16kHz using FFmpeg...")
                try:
                    # Use FFmpeg directly for better large file handling
                    ffmpeg_cmd = [
                        'ffmpeg', '-i', video_file,
                        '-ar', '16000',        # Sample rate 16kHz
                        '-ac', '1',            # Mono channel
                        '-c:a', 'pcm_s16le',   # PCM 16-bit little endian
                        '-y',                  # Overwrite output file
                        wav_path
                    ]
                    
                    result_code = subprocess.run(
                        ffmpeg_cmd,
                        capture_output=True,
                        text=True,
                        timeout=3600  # 1 hour timeout for very large files
                    )
                    
                    if result_code.returncode == 0:
                        print(f"✓ Audio extracted: {base_filename}.wav")
                    else:
                        error_msg = f"FFmpeg failed: {result_code.stderr}"
                        print(f"✗ {error_msg}")
                        result["errors"].append(error_msg)
                        
                except subprocess.TimeoutExpired:
                    error_msg = "Audio extraction timed out (>1 hour)"
                    print(f"✗ {error_msg}")
                    result["errors"].append(error_msg)
                except FileNotFoundError:
                    # Fallback to pydub if FFmpeg not available
                    print("FFmpeg not found, falling back to pydub...")
                    try:
                        video = AudioSegment.from_file(video_file)
                        audio = video.set_channels(1).set_frame_rate(16000)
                        audio.export(wav_path, format="wav", parameters=["-ar", "16000"])
                        print(f"✓ Audio extracted with pydub: {base_filename}.wav")
                    except Exception as e:
                        error_msg = f"Audio extraction failed (pydub): {str(e)}"
                        print(f"✗ {error_msg}")
                        result["errors"].append(error_msg)
                except Exception as e:
                    error_msg = f"Audio extraction failed: {str(e)}"
                    print(f"✗ {error_msg}")
                    result["errors"].append(error_msg)
            else:
                error_msg = "Cannot extract audio: video file not found"
                print(f"✗ {error_msg}")
                result["errors"].append(error_msg)
        else:
            print(f"✓ Audio already exists: {base_filename}.wav")
        
        # Download chat if not exists
        if not chat_exists:
            print("Downloading chat...")
            try:
                chat_downloader = ChatDownloader()
                chat = chat_downloader.get_chat(vod_url)
                
                message_count = 0
                skipped_count = 0
                
                # Write messages to CSV
                print("Writing chat messages to CSV...")
                with open(chat_path, 'w', newline='', encoding='utf-8') as f:
                    csv_writer = csv.writer(f)
                    # Write header
                    csv_writer.writerow(['Timestamp', 'Author', 'Message', 'Badges', 'Subscriber', 'Moderator'])
                    
                    # Write messages
                    for message in chat:
                        try:                     
                            # Get message details
                            timestamp = message.get('time_text', '')
                            author = message.get('author', {}).get('name', 'Anonymous')
                            content = message.get('message', '')
                            badges = [badge.get('title', '') for badge in message.get('author', {}).get('badges', [])]
                            badges_str = ','.join(badges) if badges else ''
                            
                            # Check if user is subscriber or moderator
                            is_subscriber = any('Subscriber' in badge for badge in badges)
                            is_moderator = any('Moderator' in badge for badge in badges)
                            
                            # Only write if we have valid content
                            if content.strip():
                                # Write to CSV
                                csv_writer.writerow([
                                    timestamp,    # HH:MM:SS format
                                    author,
                                    content,
                                    badges_str,
                                    is_subscriber,
                                    is_moderator
                                ])
                                message_count += 1
                                
                        except Exception as e:
                            # Skip messages with any issues
                            skipped_count += 1
                            continue
                
                print(f"✓ Chat downloaded: {base_filename}_chat.csv")
                print(f"  Messages saved: {message_count}")
                if skipped_count > 0:
                    print(f"  Messages skipped (errors): {skipped_count}")
                    
            except Exception as e:
                error_msg = f"Chat download failed: {str(e)}"
                print(f"✗ {error_msg}")
                result["errors"].append(error_msg)
        else:
            print(f"✓ Chat already exists: {base_filename}_chat.csv")
        
        # Update result with final file paths
        result["video_file"] = video_file if video_file and os.path.exists(video_file) else None
        result["audio_file"] = wav_path if os.path.exists(wav_path) else None
        result["chat_file"] = chat_path if os.path.exists(chat_path) else None
        
        # Save/update paths to database
        try:
            db_video_id = add_or_update_video(
                vod_url=vod_url,
                video_path=result["video_file"],
                audio_path=result["audio_file"],
                chat_csv_path=result["chat_file"]  # Updated to use chat_csv_path
            )
            print(f"✓ Database updated for video {db_video_id}")
        except Exception as e:
            error_msg = f"Database update failed: {str(e)}"
            print(f"⚠ Warning: {error_msg}")
            result["errors"].append(error_msg)
        
        # Determine overall success
        result["success"] = len(result["errors"]) == 0
        
        # Final status report
        print(f"\n📁 Processing {'complete' if result['success'] else 'completed with errors'} for VOD {video_id}")
        print(f"Files in {output_dir}:")
        if result["video_file"]:
            print(f"✓ Video: {os.path.basename(result['video_file'])} (1080p or best available)")
        if result["audio_file"]:
            print(f"✓ Audio: {base_filename}.wav (16kHz mono)")
        if result["chat_file"]:
            print(f"✓ Chat: {base_filename}_chat.csv")
            
        if result["errors"]:
            print(f"⚠ Errors encountered: {len(result['errors'])}")
            for error in result["errors"]:
                print(f"  - {error}")
        
        return result
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"❌ {error_msg}")
        result["errors"].append(error_msg)
        result["success"] = False
        return result

# Example usage:
# download_twitch_vod("https://www.twitch.tv/videos/YOUR_VOD_ID")

def download_chat(vod_url, chat_path=None, base_output_dir="/mnt/d/Projects/twitch"):
    """
    Download the chat for a Twitch VOD and save it to a CSV file.
    
    Args:
        vod_url (str): URL of the Twitch VOD
        chat_path (str, optional): Custom path for chat file. If None, uses video_id folder structure
        base_output_dir (str): Base directory to save files under video_id folder
    """
    # Extract video ID and set up paths
    video_id = vod_url.split('/')[-1]
    
    if chat_path is None:
        # Use new folder structure
        video_dir = os.path.join(base_output_dir, video_id)
        os.makedirs(video_dir, exist_ok=True)
        chat_path = os.path.join(video_dir, "chat.txt")
    
    print(f"Downloading chat for video {video_id}...")
    print(f"Saving to: {chat_path}")
    
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
    
    print(f"Chat saved to: {chat_path}")
    return {
        "video_id": video_id,
        "chat_file": chat_path,
        "message_count": len(chat_messages)
    }

def chat_message_activity_to_json(video_id, interval_seconds=5, output_json_path=None):
    """
    Count how many messages appear in each timeframe and save as JSON, ordered by count descending.
    Args:
        video_id (str): Video ID to process
        interval_seconds (int): Size of each timeframe in seconds
        output_json_path (str, optional): Custom path for JSON output. If None, gets from database
    Output JSON format:
        {
            "0:00-0:05": 3,
            "0:05-0:10": 7,
            ...
        }
    """
    # Get chat file path from database
    chat_paths = get_chat_paths(video_id)
    if not chat_paths or not chat_paths['chat_csv_path']:
        raise ValueError(f"Chat file not found in database for video_id: {video_id}")
    
    chat_file_path = chat_paths['chat_csv_path']
    
    if output_json_path is None:
        # Use database structure - create chat_activity.json in same directory as chat.csv
        output_json_path = os.path.join(os.path.dirname(chat_file_path), "chat_activity.json")
    
    def time_text_to_seconds(time_text):
        # Supports HH:MM:SS or MM:SS
        parts = [int(p) for p in re.split(r":", time_text)]
        if len(parts) == 3:
            return parts[0]*3600 + parts[1]*60 + parts[2]
        elif len(parts) == 2:
            return parts[0]*60 + parts[1]
        return 0

    print(f"Processing chat activity for video {video_id}...")
    print(f"Output will be saved to: {output_json_path}")
    
    # Read chat messages from CSV file
    chat_messages = []
    print("Reading chat messages from file...")
    try:
        with open(chat_file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                chat_messages.append(row)
    except Exception as e:
        raise ValueError(f"Error reading chat file {chat_file_path}: {e}")

    # Collect all message times in seconds
    times = []
    for msg in chat_messages:
        # Use 'Timestamp' column from CSV (which contains time_text format like "01:23:45")
        t = msg.get('Timestamp', None)
        if t and t.strip():
            try:
                times.append(time_text_to_seconds(t))
            except (ValueError, AttributeError):
                # Skip messages with invalid timestamps
                continue
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
    
    # Update database with chat JSON path
    try:
        from .db import update_file_path
        update_file_path(video_id, chat_json_path=output_json_path)
        print(f"✓ Database updated with chat activity JSON path")
    except Exception as e:
        print(f"Warning: Could not update database with chat JSON path: {e}")
    
    return {
        "video_id": video_id,
        "activity_file": output_json_path,
        "total_messages": len(chat_messages),
        "total_intervals": len(results),
        "interval_seconds": interval_seconds
    }
