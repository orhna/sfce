import yt_dlp
import os
from chat_downloader import ChatDownloader
from datetime import datetime
import json
import csv
import re
import subprocess
from collections import Counter, OrderedDict
from .db import add_or_update_video, extract_video_id_from_url, get_chat_paths, update_chat_json_path
    
def download_twitch_vod(vod_url, output_dir="/mnt/d/Projects/twitch_videos", interval_seconds=5):
    """
    Download a Twitch VOD and extract its audio file as WAV.
    
    Args:
        vod_url: Twitch VOD URL
        output_dir: Directory to save files
        interval_seconds: Interval for analysis (used for database storage)
    
    Returns:
        dict: Paths to downloaded files
    """
    import os
    import yt_dlp
    from utils.db import add_or_update_video
    
    try:
        # Extract video ID from URL
        video_id = vod_url.split('/')[-1]
        print(f"🎥 Processing video ID: {video_id}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Define file paths
        video_path = os.path.join(output_dir, f"{video_id}.mp4")
        audio_path = os.path.join(output_dir, f"{video_id}.wav")
        chat_csv_path = os.path.join(output_dir, f"{video_id}_chat.csv")
        
        # Download video if it doesn't exist
        if not os.path.exists(video_path):
            print(f"📥 Downloading video...")
            ydl_opts = {
                'format': 'best[height=1080]/best[height<1080]/best',
                'outtmpl': os.path.join(output_dir, f"{video_id}.%(ext)s"),
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([vod_url])
            print(f"✅ Video downloaded: {video_path}")
        else:
            print(f"✅ Video already exists: {video_path}")
        
        # Extract audio if it doesn't exist
        if not os.path.exists(audio_path):
            print(f"🎵 Extracting audio...")
            
            # Use ffmpeg directly instead of pydub for large file support
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # WAV format
                '-ar', '44100',  # Sample rate
                '-ac', '2',  # Stereo
                '-y',  # Overwrite output file
                audio_path
            ]
            
            try:
                result = subprocess.run(
                    ffmpeg_cmd,
                    check=True,
                    capture_output=True,
                    text=True
                )
                print(f"✅ Audio extracted: {audio_path}")
            except subprocess.CalledProcessError as e:
                print(f"❌ FFmpeg error: {e.stderr}")
                raise Exception(f"Audio extraction failed: {e.stderr}")
            except FileNotFoundError:
                print("❌ FFmpeg not found. Please install ffmpeg.")
                raise Exception("FFmpeg not found. Please install ffmpeg with: sudo apt update && sudo apt install ffmpeg")
        else:
            print(f"✅ Audio already exists: {audio_path}")
        
        # Download chat if it doesn't exist
        if not os.path.exists(chat_csv_path):
            print(f"💬 Downloading chat...")
            download_chat(vod_url, chat_csv_path)
        else:
            print(f"✅ Chat already exists: {chat_csv_path}")
        
        # Update database with interval-aware storage
        print(f"💾 Updating database (interval: {interval_seconds}s)...")
        add_or_update_video(
            vod_url=vod_url,
            interval_seconds=interval_seconds,
            video_path=video_path,
            audio_path=audio_path,
            chat_csv_path=chat_csv_path
        )
        
        return {
            'video_id': video_id,
            'video_path': video_path,
            'audio_path': audio_path,
            'chat_csv_path': chat_csv_path
        }
    
    except Exception as e:
        print(f"❌ Error downloading VOD: {e}")
        raise

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

def comprehensive_multi_modal_workflow(vod_url, interval_seconds=5, top_highlights=10, extract_videos=True, pre_padding_seconds=5):
    """
    Comprehensive workflow that processes everything from download to highlight extraction.
    
    Args:
        vod_url: Twitch VOD URL
        interval_seconds: Time interval for processing (creates folder structure)
        top_highlights: Number of top highlights to extract as videos
        extract_videos: Whether to cut highlight videos from the original
        pre_padding_seconds: Seconds to add before each highlight (default: 5)
    """
    print(f"🚀 Starting Comprehensive Multi-Modal Workflow")
    print(f"📺 VOD URL: {vod_url}")
    print(f"⏱️  Interval: {interval_seconds} seconds")
    print(f"🏆 Top highlights to extract: {top_highlights}")
    print(f"⏪ Pre-padding: {pre_padding_seconds} seconds")
    print("=" * 80)
    
    try:
        from .db import setup_database, get_video_paths
        
        # Step 1: Setup database
        print("\n📋 Step 1: Setting up database...")
        setup_database()
        
        # Step 2: Extract video ID
        video_id = vod_url.split('/')[-1]
        print(f"🆔 Video ID: {video_id}")
        
        # Step 3: Create interval-specific directory structure
        base_dir = f"/mnt/d/Projects/twitch/{video_id}"
        interval_dir = os.path.join(base_dir, str(interval_seconds))
        highlights_dir = os.path.join(interval_dir, "highlights")
        
        os.makedirs(interval_dir, exist_ok=True)
        os.makedirs(highlights_dir, exist_ok=True)
        print(f"📁 Created directory structure: {interval_dir}")
        
        # Step 4: Download and process VOD (if not already done)
        paths = get_video_paths(video_id, interval_seconds)
        
        if not paths or not paths.get('video_path') or not os.path.exists(paths['video_path']):
            print("\n📥 Step 2: Downloading VOD...")
            download_twitch_vod(vod_url, base_dir, interval_seconds)
            paths = get_video_paths(video_id, interval_seconds)
        else:
            print("\n✓ Step 2: VOD already downloaded, skipping...")
        
        if not paths:
            raise ValueError("Failed to get video paths from database")
        
        # Step 5: Generate chat activity with correct interval
        chat_activity_path = os.path.join(interval_dir, 'chat_activity.json')
        if not os.path.exists(chat_activity_path):
            print(f"\n📊 Step 3: Generating chat activity analysis ({interval_seconds}s intervals)...")
            
            # Call with video_id instead of chat_messages
            chat_message_activity_to_json(
                video_id=video_id,
                interval_seconds=interval_seconds,
                output_json_path=chat_activity_path
            )
        else:
            print(f"\n✓ Step 3: Chat activity already generated, skipping...")
        
        # Step 6: Generate video emotions
        emotions_path = os.path.join(interval_dir, 'video_emotions.json')
        if not os.path.exists(emotions_path):
            print(f"\n😊 Step 4: Analyzing video emotions ({interval_seconds}s intervals)...")
            from .video import analyze_video_emotions_by_id
            analyze_video_emotions_by_id(
                video_id=video_id,
                interval_sec=interval_seconds,
                output_json_path=emotions_path
            )
        else:
            print(f"\n✓ Step 4: Video emotions already analyzed, skipping...")
        
        # Step 7: Generate audio laughs
        laughs_path = os.path.join(interval_dir, 'audio_laughs.json')
        if not os.path.exists(laughs_path):
            print(f"\n😂 Step 5: Detecting audio laughs ({interval_seconds}s intervals)...")
            from .audio import detect_audio_laughs_by_id
            detect_audio_laughs_by_id(
                video_id=video_id,
                interval_sec=interval_seconds,
                output_json_path=laughs_path,
                threshold=0.2
            )
        else:
            print(f"\n✓ Step 5: Audio laughs already detected, skipping...")
        
        # Step 8: Generate audio transcriptions
        transcriptions_path = os.path.join(interval_dir, 'chat_transcriptions.json')
        if not os.path.exists(transcriptions_path):
            print(f"\n🎤 Step 6: Generating audio transcriptions ({interval_seconds}s intervals)...")
            
            # Load chat activity to get top intervals
            with open(chat_activity_path, 'r', encoding='utf-8') as f:
                chat_activity = json.load(f)
            
            # Get top N intervals by message count
            top_intervals = sorted(chat_activity.items(), key=lambda x: x[1], reverse=True)[:50]
            
            generate_audio_transcriptions(
                audio_path=paths['audio_path'],
                intervals=top_intervals,
                interval_seconds=interval_seconds,
                output_path=transcriptions_path
            )
        else:
            print(f"\n✓ Step 6: Audio transcriptions already generated, skipping...")
        
        # Step 9: Run multi-modal analysis
        print(f"\n🎯 Step 7: Running multi-modal highlight analysis...")
        from .segment_analysis import analyze_multi_modal_highlights
        
        results = analyze_multi_modal_highlights(
            video_id=video_id,
            interval_data_dir=interval_dir,
            top_n=top_highlights * 2  # Analyze more than we'll extract
        )
        
        if results.get('error'):
            raise ValueError(f"Multi-modal analysis failed: {results['error']}")
        
        # Step 10: Extract highlight videos
        if extract_videos and results.get('top_highlights'):
            print(f"\n🎬 Step 8: Extracting top {top_highlights} highlight videos...")
            extract_highlight_videos(
                video_path=paths['video_path'],
                highlights=results['top_highlights'][:top_highlights],
                output_dir=highlights_dir,
                interval_seconds=interval_seconds,
                pre_padding_seconds=pre_padding_seconds,
                video_id=video_id,
                interval_data_dir=interval_dir
            )
        
        # Step 11: Update database with all paths
        print(f"\n💾 Step 9: Updating database...")
        from .db import update_file_path
        update_file_path(
            video_id,
            interval_seconds,
            chat_json_path=chat_activity_path,
            chat_transcript_path=transcriptions_path,
            video_emotions_path=emotions_path,
            audio_laughs_path=laughs_path,
            highlights_dir=highlights_dir
        )
        
        # Final summary
        print(f"\n✅ COMPREHENSIVE WORKFLOW COMPLETE!")
        print(f"📁 All files saved to: {interval_dir}")
        print(f"🎬 Highlight videos saved to: {highlights_dir}")
        print(f"📊 Total intervals analyzed: {results.get('total_intervals_analyzed', 0)}")
        if results.get('top_highlights'):
            top_score = results['top_highlights'][0]['combined_score']
            print(f"🏆 Best highlight score: {top_score:.3f}")
            print(f"🎯 Best highlight type: {results['top_highlights'][0]['highlight_type']}")
        
        return {
            'success': True,
            'video_id': video_id,
            'interval_dir': interval_dir,
            'highlights_dir': highlights_dir,
            'results': results
        }
        
    except Exception as e:
        print(f"\n❌ WORKFLOW FAILED: {e}")
        return {
            'success': False,
            'error': str(e)
        }


def generate_audio_transcriptions(audio_path, intervals, interval_seconds, output_path):
    """Generate audio transcriptions for specific intervals without creating extra folders."""
    try:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        import librosa
        
        print(f"🎤 Generating transcriptions for {len(intervals)} intervals...")
        
        # Initialize speech-to-text model
        print("Initializing speech-to-text model...")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        model_id = "distil-whisper/distil-large-v3"
        
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(device)
        
        processor = AutoProcessor.from_pretrained(model_id)
        
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=torch_dtype,
            device=device,
        )
        
        # Process each interval
        transcriptions = {}
        
        for i, (time_range, message_count) in enumerate(intervals, 1):
            print(f"  Processing {i}/{len(intervals)}: {time_range} ({message_count} messages)")
            
            try:
                # Parse time range
                start_time_str, end_time_str = time_range.split('-')
                
                # Convert time to seconds
                def time_to_seconds(time_str):
                    parts = time_str.split(':')
                    if len(parts) == 2:
                        return int(parts[0]) * 60 + int(parts[1])
                    elif len(parts) == 3:
                        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                    return 0
                
                start_seconds = time_to_seconds(start_time_str)
                end_seconds = start_seconds + interval_seconds
                
                # Load audio segment
                audio_segment, sr = librosa.load(audio_path, sr=16000, offset=start_seconds, duration=interval_seconds)
                
                if len(audio_segment) > 0:
                    # Transcribe
                    result = pipe(audio_segment, generate_kwargs={"language": "english"})
                    transcriptions[time_range] = {
                        "text": result["text"],
                        "start_time": start_seconds,
                        "end_time": end_seconds,
                        "message_count": message_count
                    }
                else:
                    transcriptions[time_range] = {
                        "text": "",
                        "start_time": start_seconds,
                        "end_time": end_seconds,
                        "message_count": message_count
                    }
            
            except Exception as e:
                print(f"    ❌ Error processing {time_range}: {e}")
                transcriptions[time_range] = {
                    "text": "",
                    "error": str(e)
                }
        
        # Save transcriptions
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(transcriptions, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Transcriptions saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error generating transcriptions: {e}")
        # Create empty file so workflow can continue
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({}, f)


def merge_sequential_intervals(highlights, interval_seconds):
    """
    Merge sequential intervals into longer clips.
    
    Args:
        highlights: List of highlight dictionaries with 'time_interval' keys
        interval_seconds: Duration of each interval in seconds
    
    Returns:
        List of merged highlight dictionaries
    """
    if not highlights:
        return highlights
    
    def time_to_seconds(time_str):
        """Convert time string to seconds."""
        parts = time_str.split(':')
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        return 0
    
    def seconds_to_time(seconds):
        """Convert seconds back to time string."""
        if seconds >= 3600:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            secs = seconds % 60
            return f"{hours}:{minutes:02d}:{secs:02d}"
        else:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{minutes}:{secs:02d}"
    
    # Parse and sort highlights by start time
    parsed_highlights = []
    for highlight in highlights:
        interval = highlight['time_interval']
        start_str, end_str = interval.split('-')
        start_seconds = time_to_seconds(start_str)
        end_seconds = time_to_seconds(end_str)
        
        parsed_highlights.append({
            'original': highlight,
            'start_seconds': start_seconds,
            'end_seconds': end_seconds,
            'start_str': start_str,
            'end_str': end_str
        })
    
    # Sort by start time
    parsed_highlights.sort(key=lambda x: x['start_seconds'])
    
    # Merge sequential intervals
    merged = []
    current_group = [parsed_highlights[0]]
    
    for i in range(1, len(parsed_highlights)):
        prev = current_group[-1]
        curr = parsed_highlights[i]
        
        # Check if current interval starts where previous ends (sequential)
        if curr['start_seconds'] == prev['end_seconds']:
            current_group.append(curr)
        else:
            # Process current group and start new group
            merged.append(process_group(current_group, interval_seconds))
            current_group = [curr]
    
    # Process last group
    if current_group:
        merged.append(process_group(current_group, interval_seconds))
    
    print(f"🔗 Merged {len(highlights)} intervals into {len(merged)} clips:")
    for merged_highlight in merged:
        interval_count = merged_highlight.get('merged_count', 1)
        if interval_count > 1:
            print(f"  📎 {merged_highlight['time_interval']} (merged {interval_count} intervals)")
        else:
            print(f"  📋 {merged_highlight['time_interval']} (single interval)")
    
    return merged


def process_group(group, interval_seconds):
    """Process a group of sequential intervals."""
    if len(group) == 1:
        # Single interval, return as-is
        return group[0]['original']
    
    # Multiple sequential intervals, merge them
    start_seconds = group[0]['start_seconds']
    end_seconds = group[-1]['end_seconds']
    
    # Convert back to time strings
    def seconds_to_time(seconds):
        if seconds >= 3600:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            secs = seconds % 60
            return f"{hours}:{minutes:02d}:{secs:02d}"
        else:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{minutes}:{secs:02d}"
    
    start_str = seconds_to_time(start_seconds)
    end_str = seconds_to_time(end_seconds)
    merged_interval = f"{start_str}-{end_str}"
    
    # Calculate combined score (weighted average)
    total_score = sum(h['original']['combined_score'] for h in group)
    avg_score = total_score / len(group)
    
    # Determine dominant highlight type
    types = [h['original']['highlight_type'] for h in group]
    dominant_type = max(set(types), key=types.count)
    
    # Create merged highlight
    merged_highlight = {
        'time_interval': merged_interval,
        'combined_score': avg_score,
        'highlight_type': f"merged_{dominant_type}",
        'merged_count': len(group),
        'original_intervals': [h['original']['time_interval'] for h in group],
        'reasoning': f"Merged {len(group)} sequential intervals: {dominant_type} moments"
    }
    
    return merged_highlight


def extract_highlight_videos(video_path, highlights, output_dir, interval_seconds, pre_padding_seconds=5, video_id=None, interval_data_dir=None):
    """Extract highlight video clips from the main video with sequential interval merging and generate descriptions."""
    try:
        # Merge sequential intervals first
        merged_highlights = merge_sequential_intervals(highlights, interval_seconds)
        print(f"🎬 Extracting {len(merged_highlights)} highlight videos (merged from {len(highlights)} intervals)...")
        
        highlight_descriptions = []
        
        for i, highlight in enumerate(merged_highlights, 1):
            interval = highlight['time_interval']
            score = highlight['combined_score']
            highlight_type = highlight['highlight_type']
            
            # Parse start time - handle both MM:SS and H:MM:SS formats
            start_time_str = interval.split('-')[0]
            parts = start_time_str.split(':')
            if len(parts) == 2:
                # MM:SS format
                minutes, seconds = map(int, parts)
                start_seconds = minutes * 60 + seconds
            elif len(parts) == 3:
                # H:MM:SS format
                hours, minutes, seconds = map(int, parts)
                start_seconds = hours * 3600 + minutes * 60 + seconds
            else:
                print(f"    ❌ Invalid time format: {start_time_str}")
                continue
            
            # Calculate duration for merged intervals
            end_time_str = interval.split('-')[1]
            end_parts = end_time_str.split(':')
            if len(end_parts) == 2:
                end_minutes, end_seconds = map(int, end_parts)
                end_total_seconds = end_minutes * 60 + end_seconds
            elif len(end_parts) == 3:
                end_hours, end_minutes, end_seconds = map(int, end_parts)
                end_total_seconds = end_hours * 3600 + end_minutes * 60 + end_seconds
            else:
                end_total_seconds = start_seconds + interval_seconds
            
            duration = end_total_seconds - start_seconds + 2  # Add 2 seconds padding at end
            actual_start = max(0, start_seconds - pre_padding_seconds)  # Don't go before video start
            actual_duration = duration + min(pre_padding_seconds, start_seconds)  # Adjust duration
            
            # Create output filename
            safe_interval = interval.replace(':', '-')
            merged_count = highlight.get('merged_count', 1)
            if merged_count > 1:
                output_filename = f"highlight_{i:02d}_{safe_interval}_{highlight_type}_merged{merged_count}_score_{score:.3f}.mp4"
            else:
                output_filename = f"highlight_{i:02d}_{safe_interval}_{highlight_type}_score_{score:.3f}.mp4"
            output_path = os.path.join(output_dir, output_filename)
            
            # Use optimized ffmpeg command for speed
            cmd = [
                'ffmpeg', '-y',  # -y to overwrite existing files
                '-ss', str(actual_start),  # Seek BEFORE input (much faster)
                '-i', video_path,
                '-t', str(actual_duration),
                '-c', 'copy',  # Copy streams without re-encoding (MUCH faster)
                '-avoid_negative_ts', 'make_zero',  # Fix timestamp issues
                output_path
            ]
            
            merged_info = f" (merged {merged_count} intervals)" if merged_count > 1 else ""
            print(f"  🎥 Extracting highlight {i}/{len(merged_highlights)}: {interval} ({highlight_type}){merged_info}...")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"    ✓ Saved: {output_filename}")
                
                # Generate description for this highlight with LLM analysis
                description = generate_highlight_description(highlight, interval, merged_count, video_id=video_id, interval_data_dir=interval_data_dir)
                
                highlight_descriptions.append({
                    "highlight_number": i,
                    "filename": output_filename,
                    "original_interval": interval,
                    "padded_start_time": format_seconds_to_time(actual_start),
                    "duration_seconds": actual_duration,
                    "score": score,
                    "highlight_type": highlight_type,
                    "merged_count": merged_count,
                    "pre_padding_seconds": min(pre_padding_seconds, start_seconds),
                    "description": description
                })
                
            else:
                print(f"    ❌ Failed to extract {interval}: {result.stderr}")
        
        # Save highlight descriptions
        descriptions_path = os.path.join(output_dir, "highlight_descriptions.json")
        with open(descriptions_path, 'w', encoding='utf-8') as f:
            json.dump({
                "video_info": {
                    "total_highlights": len(highlight_descriptions),
                    "pre_padding_seconds": pre_padding_seconds,
                    "generation_timestamp": datetime.now().isoformat()
                },
                "highlights": highlight_descriptions
            }, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Highlight extraction complete! Videos saved to: {output_dir}")
        print(f"📝 Descriptions saved to: {descriptions_path}")
        
    except Exception as e:
        print(f"❌ Error extracting highlight videos: {e}")


def format_seconds_to_time(seconds):
    """Convert seconds to H:MM:SS or MM:SS format."""
    if seconds >= 3600:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}:{secs:02d}"


def generate_highlight_description(highlight, interval, merged_count, video_id=None, interval_data_dir=None):
    """Generate an AI-powered description for the highlight based on chat and transcript analysis."""
    try:
        # If we have access to the data, use LLM analysis
        if video_id and interval_data_dir:
            return generate_llm_highlight_description(highlight, interval, merged_count, video_id, interval_data_dir)
        else:
            # Fallback to template-based description
            return generate_template_description(highlight, interval, merged_count)
    
    except Exception as e:
        print(f"    ⚠️ LLM description failed, using template fallback: {e}")
        return generate_template_description(highlight, interval, merged_count)


def generate_llm_highlight_description(highlight, interval, merged_count, video_id, interval_data_dir):
    """Generate LLM-powered description using chat and transcript analysis."""
    from .segment_analysis import analyze_segment_by_interval, SegmentAnalysisState, create_segment_analysis_graph
    
    print(f"    🤖 Generating AI description for {interval}...")
    
    try:
        # Create workflow for this specific interval
        workflow = create_segment_analysis_graph()
        
        # Create state for the interval
        state = SegmentAnalysisState(
            video_id=video_id,
            time_interval=interval,
            chat_messages=[],  # Will be loaded by extract_interval_data
            audio_transcription=None,
            chat_summary=None,
            audio_summary=None,
            highlight_description=None,
            errors=[]
        )
        
        # Run the LangGraph workflow to get chat and audio summaries
        final_state = workflow.invoke(state)
        
        # Handle both dict and state object returns
        if isinstance(final_state, dict):
            highlight_desc = final_state.get('highlight_description')
            chat_summary = final_state.get('chat_summary', 'No chat summary available')
            audio_summary = final_state.get('audio_summary', 'No audio summary available')
        else:
            highlight_desc = final_state.highlight_description
            chat_summary = final_state.chat_summary or 'No chat summary available'
            audio_summary = final_state.audio_summary or 'No audio summary available'
        
        # Create enhanced description combining LLM analysis with metadata
        if highlight_desc and hasattr(highlight_desc, 'dict'):
            llm_desc = highlight_desc.dict()
            base_description = llm_desc.get('combined_description', 'AI-generated highlight description')
            keywords = llm_desc.get('keywords', [])
            
            # Add merge context if applicable
            if merged_count > 1:
                original_intervals = highlight.get('original_intervals', [])
                merge_info = f"This extended highlight merges {merged_count} consecutive intervals"
                if original_intervals:
                    merge_info += f" ({', '.join(original_intervals[:2])}{'...' if len(original_intervals) > 2 else ''})"
                merge_info += ". "
            else:
                merge_info = ""
            
            # Combine all elements
            full_description = f"{merge_info}{base_description}"
            
            if keywords:
                full_description += f" Key moments: {', '.join(keywords[:5])}."
            
            # Add chat and audio insights
            if chat_summary and chat_summary != 'No chat summary available':
                full_description += f" Chat activity: {chat_summary}"
            
            if audio_summary and audio_summary != 'No audio summary available':
                full_description += f" Audio content: {audio_summary}"
            
            return full_description
        
        else:
            # LLM analysis available but not in expected format
            score = highlight['combined_score']
            return f"AI-analyzed highlight (score: {score:.3f}). Chat: {chat_summary}. Audio: {audio_summary}"
    
    except Exception as e:
        print(f"    ❌ LLM analysis failed: {e}")
        raise e


def generate_template_description(highlight, interval, merged_count):
    """Generate a template-based description as fallback."""
    highlight_type = highlight['highlight_type']
    score = highlight['combined_score']
    reasoning = highlight.get('reasoning', 'High activity detected')
    
    # Base description based on highlight type
    type_descriptions = {
        'high_engagement': 'High chat engagement moment',
        'emotional_moment': 'Emotional reaction captured',
        'funny_moment': 'Comedic moment with laughter',
        'content_rich': 'Rich audio content detected',
        'multi_modal': 'Multi-signal highlight',
        'merged_high_engagement': 'Extended high engagement sequence',
        'merged_emotional_moment': 'Extended emotional sequence',
        'merged_funny_moment': 'Extended comedic sequence',
        'merged_content_rich': 'Extended content-rich sequence'
    }
    
    base_desc = type_descriptions.get(highlight_type, 'Notable moment detected')
    
    # Add merge context
    if merged_count > 1:
        merge_context = f"Extended {merged_count}-interval highlight featuring {base_desc.lower()}. "
    else:
        merge_context = f"{base_desc}. "
    
    # Add score and reasoning
    description = f"{merge_context}Score: {score:.3f}/1.0. {reasoning}"
    
    return description
