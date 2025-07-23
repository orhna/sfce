#!/usr/bin/env python3
"""
Example workflow for Short Form Content Extraction from Twitch streams.

This script demonstrates the new database-integrated workflow:
1. Download VOD with download_twitch_vod() - saves paths to database
2. Use video_id for all subsequent operations
3. Functions automatically retrieve paths from database
"""

from utils.utils import download_twitch_vod, chat_message_activity_to_json
from utils.video import analyze_video_emotions_by_id
from utils.audio import detect_audio_laughs_by_id, analyze_laugh_threshold_sensitivity
from utils.chat import process_chat_by_id
from utils.db import list_all_videos, get_video_paths, create_tables


def setup_database():
    """Initialize the database with the updated schema."""
    print("=== Setting up Database ===")
    try:
        create_tables()
        print("✅ Database setup complete")
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return False
    return True


def main_workflow(vod_url):
    """
    Complete workflow for processing a Twitch VOD.
    
    Args:
        vod_url (str): Twitch VOD URL (e.g., "https://www.twitch.tv/videos/2465574148")
    """
    print("=== Short Form Content Extraction Workflow ===")
    print(f"Processing VOD: {vod_url}")
    
    # Step 0: Setup database if needed
    if not setup_database():
        return
    
    # Step 1: Download VOD and save paths to database
    print("\n1. Downloading VOD (video, audio, chat)...")
    try:
        result = download_twitch_vod(vod_url)
        video_id = result['video_id']
        print(f"✅ Successfully downloaded VOD with video_id: {video_id}")
    except Exception as e:
        print(f"❌ Error downloading VOD: {e}")
        return
    
    # Step 2: Process chat activity (if not already done)
    print(f"\n2. Processing chat activity for video_id: {video_id}...")
    try:
        chat_result = chat_message_activity_to_json(video_id, interval_seconds=5)
        print(f"✅ Chat activity processed: {chat_result['total_intervals']} intervals found")
    except Exception as e:
        print(f"❌ Error processing chat activity: {e}")
        return
    
    # Step 3: Analyze video emotions (first 2 minutes for demo)
    print(f"\n3. Analyzing video emotions for video_id: {video_id}...")
    try:
        emotion_result = analyze_video_emotions_by_id(video_id, interval_sec=5, time_period="00:00-02:00")
        emotion_count = len(emotion_result['emotion_segments'])
        print(f"✅ Video emotions analyzed: {emotion_count} segments processed")
        
        # Show sample emotions
        sample_emotions = list(emotion_result['emotion_segments'].items())[:5]
        print("Sample emotions:")
        for time_key, emotion in sample_emotions:
            print(f"  {time_key}: {emotion}")
    except Exception as e:
        print(f"❌ Error analyzing video emotions: {e}")
    
    # Step 4: Detect audio laughs (first 2 minutes for demo)
    print(f"\n4. Detecting audio laughs for video_id: {video_id}...")
    try:
        laugh_result = detect_audio_laughs_by_id(
            video_id=video_id, 
            interval_sec=5, 
            chunk_duration=10, 
            threshold=0.3  # Lower threshold for more sensitivity
        )
        laugh_count = laugh_result['intervals_with_laughs']
        total_intervals = laugh_result['total_intervals']
        print(f"✅ Audio laugh detection completed: {laugh_count}/{total_intervals} intervals contain laughs")
        
        # Show sample laugh intervals
        laugh_intervals = [k for k, v in laugh_result['laugh_segments'].items() if v][:5]
        if laugh_intervals:
            print("Sample laugh intervals:")
            for interval in laugh_intervals:
                print(f"  {interval}: Laugh detected")
        else:
            print("No laughs detected in analyzed segments")
    except Exception as e:
        print(f"❌ Error detecting audio laughs: {e}")
    
    # Step 5: Process chat with speech-to-text (top 5 segments)
    print(f"\n5. Processing chat transcriptions for video_id: {video_id}...")
    try:
        transcription_result = process_chat_by_id(
            video_id=video_id,
            top_n=5,  # Reduced for demo
            interval_seconds=5,
            audio_padding_seconds=2
        )
        segments_count = transcription_result['total_segments_processed']
        print(f"✅ Chat transcriptions completed: {segments_count} segments transcribed")
        
        # Show sample transcriptions
        sample_transcriptions = list(transcription_result['transcribed_segments'].items())[:3]
        print("Sample transcriptions:")
        for time_range, data in sample_transcriptions:
            print(f"  {time_range} ({data['message_count']} messages): {data['text'][:100]}...")
    except Exception as e:
        print(f"❌ Error processing chat transcriptions: {e}")
    
    print(f"\n=== Workflow completed for video_id: {video_id} ===")
    
    # Show file locations and database status
    show_processing_summary(video_id)


def show_processing_summary(video_id):
    """Show a summary of all processed files for a video ID."""
    paths = get_video_paths(video_id)
    if not paths:
        print(f"❌ No database entry found for video_id: {video_id}")
        return
    
    print(f"\n📊 Processing Summary for {video_id}:")
    print(f"Files saved in: /mnt/d/Projects/twitch/{video_id}/")
    
    # Core files
    print(f"📹 Video: {'✅' if paths['video_path'] else '❌'} {paths['video_path'] or 'Not available'}")
    print(f"🎵 Audio: {'✅' if paths['audio_path'] else '❌'} {paths['audio_path'] or 'Not available'}")
    print(f"💬 Chat CSV: {'✅' if paths['chat_csv_path'] else '❌'} {paths['chat_csv_path'] or 'Not available'}")
    
    # Analysis files
    print(f"📊 Chat Activity: {'✅' if paths['chat_json_path'] else '❌'} {paths['chat_json_path'] or 'Not available'}")
    print(f"🎭 Video Emotions: {'✅' if paths['video_emotions_path'] else '❌'} {paths['video_emotions_path'] or 'Not available'}")
    print(f"😂 Audio Laughs: {'✅' if paths['audio_laughs_path'] else '❌'} {paths['audio_laughs_path'] or 'Not available'}")
    print(f"📝 Chat Transcript: {'✅' if paths['chat_transcript_path'] else '❌'} {paths['chat_transcript_path'] or 'Not available'}")


def list_processed_videos():
    """List all videos in the database."""
    print("=== Processed Videos in Database ===")
    videos = list_all_videos()
    
    if not videos:
        print("No videos found in database.")
        return
    
    for video in videos:
        print(f"\nVideo ID: {video['video_id']}")
        print(f"  URL: {video['vod_url']}")
        print(f"  Processed: {video['processed_at']}")
        print(f"  Core Files:")
        print(f"    Video: {'✅' if video['video_path'] else '❌'}")
        print(f"    Audio: {'✅' if video['audio_path'] else '❌'}")
        print(f"    Chat CSV: {'✅' if video['chat_csv_path'] else '❌'}")
        print(f"  Analysis Files:")
        print(f"    Chat Activity: {'✅' if video['chat_json_path'] else '❌'}")
        print(f"    Video Emotions: {'✅' if video['video_emotions_path'] else '❌'}")
        print(f"    Audio Laughs: {'✅' if video['audio_laughs_path'] else '❌'}")
        print(f"    Chat Transcript: {'✅' if video['chat_transcript_path'] else '❌'}")


def quick_analysis(video_id):
    """Run quick analysis on an already downloaded video."""
    print(f"=== Quick Analysis for {video_id} ===")
    
    try:
        # Chat activity
        print("1. Processing chat activity...")
        chat_result = chat_message_activity_to_json(video_id, interval_seconds=5)
        print(f"✅ Chat activity: {chat_result['total_intervals']} intervals")
        
        # Video emotions (first minute)
        print("2. Analyzing video emotions...")
        emotion_result = analyze_video_emotions_by_id(video_id, interval_sec=5, time_period="00:00-01:00")
        print(f"✅ Video emotions: {len(emotion_result['emotion_segments'])} segments")
        
        # Audio laughs (first minute)
        print("3. Detecting audio laughs...")
        laugh_result = detect_audio_laughs_by_id(video_id, interval_sec=5)
        laugh_count = laugh_result['intervals_with_laughs']
        total_intervals = laugh_result['total_intervals']
        print(f"✅ Audio laughs: {laugh_count}/{total_intervals} intervals with laughs")
        
        print(f"✅ Quick analysis completed for {video_id}")
        show_processing_summary(video_id)
        
    except Exception as e:
        print(f"❌ Error in quick analysis: {e}")


def test_laugh_thresholds(video_id):
    """Test different laugh detection thresholds to optimize sensitivity."""
    print(f"=== Testing Laugh Detection Thresholds for {video_id} ===")
    
    try:
        results = analyze_laugh_threshold_sensitivity(video_id)
        
        # Recommend optimal threshold
        print(f"\n💡 Recommendations:")
        print(f"  - For conservative detection (fewer false positives): Use threshold 0.4-0.5")
        print(f"  - For balanced detection: Use threshold 0.3")
        print(f"  - For sensitive detection (catch more laughs): Use threshold 0.1-0.2")
        print(f"\nRe-run laugh detection with your preferred threshold:")
        print(f"  detect_audio_laughs_by_id('{video_id}', threshold=0.3)")
        
    except Exception as e:
        print(f"❌ Error testing thresholds: {e}")


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python example_workflow.py <vod_url>                    # Process new VOD")
        print("  python example_workflow.py --list                      # List processed videos")
        print("  python example_workflow.py --quick <video_id>          # Quick analysis on existing video")
        print("  python example_workflow.py --test-thresholds <video_id> # Test laugh detection thresholds")
        print("  python example_workflow.py --setup                     # Setup database only")
        print("\nExample:")
        print("  python example_workflow.py https://www.twitch.tv/videos/2465574148")
        print("  python example_workflow.py --quick 2465574148")
        print("  python example_workflow.py --test-thresholds 2465574148")
        sys.exit(1)
    
    if sys.argv[1] == "--list":
        list_processed_videos()
    elif sys.argv[1] == "--setup":
        setup_database()
    elif sys.argv[1] == "--quick" and len(sys.argv) > 2:
        video_id = sys.argv[2]
        quick_analysis(video_id)
    elif sys.argv[1] == "--test-thresholds" and len(sys.argv) > 2:
        video_id = sys.argv[2]
        test_laugh_thresholds(video_id)
    else:
        vod_url = sys.argv[1]
        main_workflow(vod_url) 