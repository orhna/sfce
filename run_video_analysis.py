#!/usr/bin/env python3
"""
Script to run video analysis on stream recordings.
This will analyze emotions and scene changes to identify highlight moments.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.video import analyze_video_emotions, save_emotion_analysis, get_emotion_statistics, find_highlight_moments

def main():
    # === CONFIGURATION ===
    video_path = "/mnt/d/Projects/twitch_videos/twitch_2465574148.mp4"
    output_json = "emotion_analysis.json"
    
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return
    
    print("=" * 60)
    print("STREAM HIGHLIGHT DETECTION - VIDEO ANALYSIS")
    print("=" * 60)
    print(f"Video: {video_path}")
    print(f"Output: {output_json}")
    print("=" * 60)
    
    # === RUNNING THE ANALYSIS ===
    try:
        # Run emotion analysis with 5-second intervals
        emotion_data = analyze_video_emotions(video_path, interval_sec=5)
        
        # Save results to JSON
        save_emotion_analysis(emotion_data, output_json)
        
        # === DISPLAY RESULTS ===
        print("\n" + "=" * 60)
        print("ANALYSIS RESULTS")
        print("=" * 60)
        
        # Get statistics
        stats = get_emotion_statistics(emotion_data)
        print("\nEmotion Distribution:")
        for emotion, percentage in sorted(stats['percentages'].items(), key=lambda x: x[1], reverse=True):
            count = stats['counts'][emotion]
            print(f"  {emotion:<12}: {percentage:6.1f}% ({count:4d} intervals)")
        
        # Find potential highlights
        highlights = find_highlight_moments(emotion_data, target_emotions=['happy', 'surprise'], min_duration=10)
        
        print(f"\nPotential Highlight Moments ({len(highlights)} found):")
        if highlights:
            for i, highlight in enumerate(highlights, 1):
                print(f"  {i:2d}. {highlight['start']} - {highlight['end']} ({highlight['duration']}s)")
        else:
            print("  No significant highlight moments detected based on emotions.")
        
        # Show sample of results
        print(f"\nSample Results (first 10 intervals):")
        for i, (time_range, emotion) in enumerate(list(emotion_data.items())[:10]):
            print(f"  {time_range}: {emotion}")
        
        print(f"\nFull results saved to: {output_json}")
        print("Analysis complete!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 