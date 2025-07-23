import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import csv
import io
import json
import os
from tqdm import tqdm
from .db import get_audio_path

# Try to import scipy for smoothing, fallback to simple average if not available
try:
    from scipy.ndimage import uniform_filter1d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available, using simple smoothing for laugh detection")

# Configure GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print("GPU is available and configured for audio processing")
else:
    print("No GPU detected. Running audio processing on CPU")


def process_audio_chunk(model, audio_chunk, class_names, threshold=0.5):
    """
    Process a single chunk of audio data to detect laughs and related sounds.
    
    Args:
        model: YAMNet model
        audio_chunk: Audio tensor
        class_names: List of YAMNet class names
        threshold: Confidence threshold for laugh detection
    
    Returns:
        list: Timestamps where laughs were detected
    """
    try:
        with tf.device('/GPU:0' if len(tf.config.list_physical_devices('GPU')) > 0 else '/CPU:0'):
            scores, embeddings, spectrogram = model(audio_chunk)
    except:
        # Fallback to CPU if GPU fails
        with tf.device('/CPU:0'):
            scores, embeddings, spectrogram = model(audio_chunk)
    
    # Find laugh-related classes (more comprehensive search)
    laugh_keywords = ['laugh', 'giggle', 'chuckle', 'snicker', 'cackle', 'guffaw', 'cheer', 'applause']
    laugh_indices = []
    
    for i, name in enumerate(class_names):
        name_lower = name.lower()
        if any(keyword in name_lower for keyword in laugh_keywords):
            laugh_indices.append(i)
    
    # Print available laugh-related classes for debugging (only once)
    if not hasattr(process_audio_chunk, "_classes_printed"):
        laugh_classes = [class_names[i] for i in laugh_indices]
        print(f"Using laugh-related classes: {laugh_classes}")
        process_audio_chunk._classes_printed = True
    
    # Get timestamps where laughs were detected
    timestamps = []
    frame_duration = 0.025  # YAMNet uses 25ms frames
    
    # Process scores
    scores_np = scores.numpy()
    
    # Enhanced detection with smoothing
    laugh_scores = scores_np[:, laugh_indices] if laugh_indices else np.array([])
    
    if laugh_scores.size > 0:
        # Get max score across all laugh classes for each frame
        max_laugh_scores = np.max(laugh_scores, axis=1)
        
        # Apply simple smoothing to reduce noise
        if len(max_laugh_scores) > 3:
            if SCIPY_AVAILABLE:
                smoothed_scores = uniform_filter1d(max_laugh_scores, size=3)
            else:
                # Simple moving average fallback
                smoothed_scores = np.convolve(max_laugh_scores, np.ones(3)/3, mode='same')
        else:
            smoothed_scores = max_laugh_scores
        
        # Find frames above threshold
        for frame_idx in range(len(smoothed_scores)):
            if smoothed_scores[frame_idx] > threshold:
                time = frame_idx * frame_duration
                timestamps.append({
                    'time': time,
                    'confidence': float(smoothed_scores[frame_idx])
                })
    
    return timestamps


def detect_audio_laughs(audio_path, interval_sec=5, chunk_duration=10, threshold=0.5, output_json_path=None):
    """
    Detect laughs in audio using YAMNet, processing the audio in chunks.
    
    Args:
        audio_path (str): Path to the audio file
        interval_sec (int): Time interval in seconds for analysis (default: 5)
        chunk_duration (int): Duration in seconds for processing chunks (default: 10)
        threshold (float): Confidence threshold for laugh detection (default: 0.5)
        output_json_path (str, optional): Custom path for JSON output. If None, saves to audio directory
    
    Returns:
        dict: JSON-like dictionary with audio_id, laugh_segments, and output_file path
    """
    # Get audio ID from filename
    audio_id = os.path.splitext(os.path.basename(audio_path))[0]
    
    # Set default output path if not provided
    if output_json_path is None:
        audio_dir = os.path.dirname(audio_path)
        output_json_path = os.path.join(audio_dir, "audio_laughs.json")
    
    print(f"Loading YAMNet model...")
    # Load the model
    try:
        with tf.device('/GPU:0' if len(tf.config.list_physical_devices('GPU')) > 0 else '/CPU:0'):
            model = hub.load('https://tfhub.dev/google/yamnet/1')
    except:
        # Fallback to CPU if GPU fails
        with tf.device('/CPU:0'):
            model = hub.load('https://tfhub.dev/google/yamnet/1')
    
    # Get class names once
    class_map_path = model.class_map_path().numpy()
    class_map_csv = io.StringIO(tf.io.read_file(class_map_path).numpy().decode('utf-8'))
    class_names = [display_name for (class_index, mid, display_name) in csv.reader(class_map_csv)]
    
    print(f"Loading audio file: {audio_path}")
    # Load audio file
    audio, sr = librosa.load(audio_path, sr=16000)  # YAMNet expects 16kHz
    
    # Calculate chunk size in samples
    chunk_size = int(chunk_duration * sr)
    
    # Process audio in chunks to detect laughs
    print(f"Processing audio in {chunk_duration}s chunks...")
    all_timestamps = []
    
    total_chunks = (len(audio) + chunk_size - 1) // chunk_size
    
    with tqdm(total=total_chunks, desc="Detecting laughs") as pbar:
        for chunk_start in range(0, len(audio), chunk_size):
            # Get chunk
            chunk = audio[chunk_start:chunk_start + chunk_size]
            
            # If the chunk is too short (last chunk), pad it
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            
            # Convert to tensor
            chunk_tensor = tf.convert_to_tensor(chunk, dtype=tf.float32)
            
            # Process chunk
            chunk_timestamps = process_audio_chunk(model, chunk_tensor, class_names, threshold)
            
            # Adjust timestamps to account for chunk position
            chunk_start_time = chunk_start / sr
            for ts in chunk_timestamps:
                ts['time'] += chunk_start_time
            
            all_timestamps.extend(chunk_timestamps)
            pbar.update(1)
    
    # Sort timestamps by time
    all_timestamps.sort(key=lambda x: x['time'])
    
    # Convert to interval-based format
    audio_duration = len(audio) / sr
    laugh_segments = {}
    
    print(f"Converting to {interval_sec}s intervals...")
    for current_sec in range(0, int(audio_duration), interval_sec):
        # Format time key as MM:SS-MM:SS (interval format)
        start_min, start_s = divmod(current_sec, 60)
        end_time = current_sec + interval_sec
        end_min, end_s = divmod(end_time, 60)
        time_key = f"{start_min}:{start_s:02d}-{end_min}:{end_s:02d}"
        
        # Check if any laugh timestamp falls within this interval
        interval_start = current_sec
        interval_end = current_sec + interval_sec
        
        # Get all laugh detections in this interval
        interval_laughs = [
            ts for ts in all_timestamps 
            if interval_start <= ts['time'] < interval_end
        ]
        
        # Use more sophisticated detection criteria:
        # 1. Any high-confidence detection (>0.7)
        # 2. Multiple detections with medium confidence (>0.4)
        # 3. Long duration of detections
        has_laugh = False
        
        if interval_laughs:
            max_confidence = max(ts['confidence'] for ts in interval_laughs)
            num_detections = len(interval_laughs)
            avg_confidence = sum(ts['confidence'] for ts in interval_laughs) / num_detections
            
            # Detection criteria
            if (max_confidence > 0.7 or  # High confidence detection
                (num_detections >= 3 and avg_confidence > 0.4) or  # Multiple medium confidence
                (num_detections >= 5)):  # Many detections regardless of confidence
                has_laugh = True
        
        laugh_segments[time_key] = has_laugh
    
    # Save laugh segments to JSON file
    print(f"Saving laugh analysis to: {output_json_path}")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(laugh_segments, f, ensure_ascii=False, indent=2)
    
    laugh_count = sum(1 for has_laugh in laugh_segments.values() if has_laugh)
    total_detections = len(all_timestamps)
    avg_confidence = np.mean([ts['confidence'] for ts in all_timestamps]) if all_timestamps else 0
    
    print(f"✓ Audio laugh analysis complete!")
    print(f"  Total intervals analyzed: {len(laugh_segments)}")
    print(f"  Intervals with laughs: {laugh_count} ({laugh_count/len(laugh_segments)*100:.1f}%)")
    print(f"  Raw laugh detections: {total_detections}")
    print(f"  Average confidence: {avg_confidence:.3f}")
    print(f"  Detection threshold: {threshold}")
    print(f"  Output saved to: {output_json_path}")
    
    # Update database with audio laughs path (if audio_id matches video_id pattern)
    try:
        from .create_db import update_file_path
        # Try to extract video_id from audio_id (assuming format like twitch_12345678)
        if audio_id.startswith('twitch_'):
            video_id = audio_id.replace('twitch_', '')
            update_file_path(video_id, audio_laughs_path=output_json_path)
            print(f"✓ Database updated with audio laughs path")
    except Exception as e:
        print(f"Warning: Could not update database with audio laughs path: {e}")
    
    return {
        "audio_id": audio_id,
        "laugh_segments": laugh_segments,
        "output_file": output_json_path,
        "total_intervals": len(laugh_segments),
        "intervals_with_laughs": laugh_count,
        "laugh_percentage": laugh_count/len(laugh_segments)*100 if laugh_segments else 0,
        "raw_detections": total_detections,
        "average_confidence": avg_confidence,
        "interval_seconds": interval_sec,
        "threshold": threshold
    }


def detect_audio_laughs_by_id(video_id, interval_sec=5, chunk_duration=10, threshold=0.5, output_json_path=None):
    """
    Detect laughs in audio using video_id to get path from database.
    
    Args:
        video_id (str): Video ID to process
        interval_sec (int): Time interval in seconds for analysis (default: 5)
        chunk_duration (int): Duration in seconds for processing chunks (default: 10)
        threshold (float): Confidence threshold for laugh detection (default: 0.5)
        output_json_path (str, optional): Custom path for JSON output. If None, saves to audio directory
    
    Returns:
        dict: JSON-like dictionary with audio_id, laugh_segments, and output_file path
    """
    # Get audio path from database
    audio_path = get_audio_path(video_id)
    if not audio_path or not os.path.exists(audio_path):
        raise ValueError(f"Audio file not found in database for video_id: {video_id}")
    
    # Set default output path to video directory
    if output_json_path is None:
        video_dir = os.path.dirname(audio_path)
        output_json_path = os.path.join(video_dir, "audio_laughs.json")
    
    return detect_audio_laughs(audio_path, interval_sec, chunk_duration, threshold, output_json_path)


def analyze_laugh_threshold_sensitivity(video_id, interval_sec=5, chunk_duration=10, test_thresholds=None):
    """
    Test different thresholds to help optimize laugh detection sensitivity.
    
    Args:
        video_id (str): Video ID to process
        interval_sec (int): Time interval in seconds for analysis
        chunk_duration (int): Duration in seconds for processing chunks
        test_thresholds (list): List of thresholds to test (default: [0.1, 0.2, 0.3, 0.4, 0.5])
    
    Returns:
        dict: Results for each threshold tested
    """
    if test_thresholds is None:
        test_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    print(f"🔬 Testing laugh detection sensitivity for video_id: {video_id}")
    print(f"Testing thresholds: {test_thresholds}")
    
    results = {}
    
    for threshold in test_thresholds:
        print(f"\n--- Testing threshold: {threshold} ---")
        try:
            result = detect_audio_laughs_by_id(
                video_id=video_id,
                interval_sec=interval_sec,
                chunk_duration=chunk_duration,
                threshold=threshold,
                output_json_path=None  # Don't save intermediate files
            )
            
            results[threshold] = {
                'intervals_with_laughs': result['intervals_with_laughs'],
                'laugh_percentage': result['laugh_percentage'],
                'raw_detections': result['raw_detections'],
                'average_confidence': result['average_confidence']
            }
            
            print(f"  Intervals with laughs: {result['intervals_with_laughs']} ({result['laugh_percentage']:.1f}%)")
            print(f"  Raw detections: {result['raw_detections']}")
            print(f"  Avg confidence: {result['average_confidence']:.3f}")
            
        except Exception as e:
            print(f"  Error at threshold {threshold}: {e}")
            results[threshold] = {'error': str(e)}
    
    # Print summary
    print(f"\n📊 Threshold Sensitivity Summary:")
    print("Threshold | Laugh% | Raw Detections | Avg Confidence")
    print("-" * 55)
    for threshold in test_thresholds:
        if 'error' not in results[threshold]:
            r = results[threshold]
            print(f"{threshold:8.1f} | {r['laugh_percentage']:5.1f}% | {r['raw_detections']:13d} | {r['average_confidence']:13.3f}")
    
    return results
