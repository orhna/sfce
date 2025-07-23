import cv2
import json
import os
from tqdm import tqdm
import mediapipe as mp
import numpy as np
from .db import get_video_path

def analyze_video_emotions_by_id(video_id, interval_sec=5, time_period=None, output_json_path=None):
    """
    Analyze emotions in a video using video_id to get path from database.
    
    Args:
        video_id (str): Video ID to process
        interval_sec (int): Time interval in seconds for analysis (default: 5)
        time_period (str, optional): Time period to analyze in format "MM:SS-MM:SS" (e.g., "00:00-00:10")
        output_json_path (str, optional): Custom path for JSON output. If None, saves to video directory
    
    Returns:
        dict: JSON-like dictionary with video_id and emotion_segments
    """
    # Get video path from database
    video_path = get_video_path(video_id)
    if not video_path or not os.path.exists(video_path):
        raise ValueError(f"Video file not found in database for video_id: {video_id}")
    
    # Set default output path to video directory
    if output_json_path is None:
        video_dir = os.path.dirname(video_path)
        output_json_path = os.path.join(video_dir, "video_emotions.json")
    
    return analyze_video_emotions(video_path, interval_sec, time_period, output_json_path)

def analyze_video_emotions(video_path, interval_sec=5, time_period=None, output_json_path=None):
    """
    Analyze emotions in a video using MediaPipe for face detection and basic emotion classification.
    
    Args:
        video_path (str): Path to the video file
        interval_sec (int): Time interval in seconds for analysis (default: 5)
        time_period (str, optional): Time period to analyze in format "MM:SS-MM:SS" (e.g., "00:00-00:10")
        output_json_path (str, optional): Custom path for JSON output. If None, saves to video directory
    
    Returns:
        dict: JSON-like dictionary with video_id, emotion_segments, and output_file path
    """
    # Get video ID from filename
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    
    # Set default output path if not provided
    if output_json_path is None:
        video_dir = os.path.dirname(video_path)
        output_json_path = os.path.join(video_dir, "video_emotions.json")
    
    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    # Parse time period if provided
    start_sec = 0
    end_sec = duration
    
    if time_period:
        try:
            start_str, end_str = time_period.split('-')
            start_min, start_s = map(int, start_str.split(':'))
            end_min, end_s = map(int, end_str.split(':'))
            start_sec = start_min * 60 + start_s
            end_sec = end_min * 60 + end_s
        except ValueError:
            print(f"Invalid time_period format: {time_period}. Using full video duration.")
    
    emotion_segments = {}
    
    # Calculate total iterations for progress bar
    total_iterations = int((end_sec - start_sec) // interval_sec)
    
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        with tqdm(total=total_iterations, desc="Analyzing emotions") as pbar:
            for current_sec in range(int(start_sec), int(end_sec), interval_sec):
                # Format time key as MM:SS-MM:SS (interval format)
                start_min, start_s = divmod(current_sec, 60)
                end_time = current_sec + interval_sec
                end_min, end_s = divmod(end_time, 60)
                time_key = f"{start_min}:{start_s:02d}-{end_min}:{end_s:02d}"
                
                # Generate 10 evenly spaced time points within the interval
                sample_times = []
                for i in range(10):
                    # Calculate time position within the interval (0-9 for 10 samples)
                    time_offset = (i + 0.5) * (interval_sec / 10)  # +0.5 to sample middle of each sub-interval
                    sample_time = current_sec + time_offset
                    sample_times.append(sample_time)
                
                # Collect emotions from all sampled frames
                interval_emotions = []
                emotion_confidences = []
                
                for sample_time in sample_times:
                    # Set video position to sample time
                    cap.set(cv2.CAP_PROP_POS_MSEC, sample_time * 1000)
                    
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    
                    # Convert BGR to RGB for MediaPipe
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Detect faces
                    results = face_detection.process(rgb_frame)
                    
                    if results.detections:
                        # Take the first (most confident) detection
                        detection = results.detections[0]
                        
                        # Get bounding box
                        bbox = detection.location_data.relative_bounding_box
                        h, w, _ = frame.shape
                        
                        # Convert relative coordinates to absolute
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        width = int(bbox.width * w)
                        height = int(bbox.height * h)
                        
                        # Check if face area is reasonable
                        face_area = width * height
                        frame_area = h * w
                        
                        if face_area >= (frame_area * 0.005) and face_area <= (frame_area * 0.8):
                            # Extract face region for emotion analysis
                            face_roi = frame[y:y+height, x:x+width]
                            
                            if face_roi.size > 0:
                                # Get emotion and confidence from HSEmotion
                                emotion, confidence = _classify_basic_emotion_with_confidence(face_roi)
                                if emotion != 'no_face':
                                    interval_emotions.append(emotion)
                                    emotion_confidences.append(confidence)
                
                # Determine the most confident/frequent emotion for this interval
                if interval_emotions:
                    final_emotion = _get_dominant_emotion(interval_emotions, emotion_confidences)
                    emotion_segments[time_key] = final_emotion
                else:
                    emotion_segments[time_key] = 'no_face'
                
                pbar.update(1)
    
    cap.release()
    
    # Save emotion segments to JSON file
    print(f"Saving emotion analysis to: {output_json_path}")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(emotion_segments, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Video emotion analysis complete!")
    print(f"  Total intervals analyzed: {len(emotion_segments)}")
    print(f"  Output saved to: {output_json_path}")
    
    # Update database with video emotions path
    try:
        from .create_db import update_file_path
        # Extract just the video ID number if filename has 'twitch_' prefix
        db_video_id = video_id.replace('twitch_', '') if video_id.startswith('twitch_') else video_id
        update_file_path(db_video_id, video_emotions_path=output_json_path)
        print(f"✓ Database updated with video emotions path for video_id: {db_video_id}")
    except Exception as e:
        print(f"Warning: Could not update database with video emotions path: {e}")
    
    return {
        "video_id": video_id,
        "emotion_segments": emotion_segments,
        "output_file": output_json_path,
        "total_intervals": len(emotion_segments),
        "interval_seconds": interval_sec
    }


def _classify_basic_emotion_with_confidence(face_roi):
    """
    Emotion classification using HSEmotion-ONNX - returns emotion and confidence.
    """
    # Initialize HSEmotion-ONNX model (cached after first use)
    if not hasattr(_classify_basic_emotion_with_confidence, "emotion_model"):
        from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
        _classify_basic_emotion_with_confidence.emotion_model = HSEmotionRecognizer(
            model_name='enet_b0_8_best_afew'  # Fast and accurate model
        )
    
    try:
        # Convert BGR to RGB 
        rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        
        # Get emotion prediction with probability scores
        emotion, scores = _classify_basic_emotion_with_confidence.emotion_model.predict_emotions(
            rgb_face, logits=False
        )
        
        # Map emotion names to lowercase for consistency
        emotion_mapping = {
            'Anger': 'angry',
            'Contempt': 'contempt',
            'Disgust': 'disgust', 
            'Fear': 'fear',
            'Happiness': 'happy',
            'Neutral': 'neutral',
            'Sadness': 'sad',
            'Surprise': 'surprised'
        }
        
        emotion_label = emotion_mapping.get(emotion, emotion.lower())
        
        # Get confidence (max probability score)
        if scores is not None and len(scores) > 0:
            max_confidence = max(scores)
            # Lower threshold for ONNX model as it tends to be more conservative
            if max_confidence > 0.25:  
                return emotion_label, max_confidence
        
        return 'neutral', 0.3  # Default confidence for neutral
        
    except Exception as e:
        return 'no_face', 0.0


def _get_dominant_emotion(emotions, confidences):
    """
    Determine the dominant emotion based on frequency and confidence.
    Prioritizes emotions with higher confidence and frequency.
    """
    if not emotions:
        return 'no_face'
    
    # Count emotion frequencies
    emotion_counts = {}
    emotion_confidence_sums = {}
    
    for emotion, confidence in zip(emotions, confidences):
        if emotion in emotion_counts:
            emotion_counts[emotion] += 1
            emotion_confidence_sums[emotion] += confidence
        else:
            emotion_counts[emotion] = 1
            emotion_confidence_sums[emotion] = confidence
    
    # Calculate weighted scores (frequency * average confidence)
    emotion_scores = {}
    for emotion in emotion_counts:
        frequency = emotion_counts[emotion]
        avg_confidence = emotion_confidence_sums[emotion] / frequency
        # Weight: frequency counts more than confidence for consistency
        emotion_scores[emotion] = frequency * 0.7 + avg_confidence * 0.3
    
    # Return emotion with highest weighted score
    return max(emotion_scores, key=emotion_scores.get)
