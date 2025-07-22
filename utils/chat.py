from utils.utils import download_chat, chat_message_activity_to_json
import json
import os
from pydub import AudioSegment
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from tqdm import tqdm


def process_chat(vod_url, audio_file_path, top_n=10, interval_seconds=5, 
                 chat_output_path="/mnt/d/Projects/twitch_videos/chat_activity.json",
                 transcription_output_path="/mnt/d/Projects/twitch_videos/chat_transcriptions.json"):
    """
    Process chat activity and transcribe top N segments from audio.
    
    Args:
        vod_url (str): URL of the Twitch VOD
        audio_file_path (str): Path to the audio file to transcribe
        top_n (int): Number of top chat activity segments to transcribe (default: 10)
        interval_seconds (int): Time interval in seconds for chat analysis (default: 5)
        chat_output_path (str): Path to save chat activity JSON
        transcription_output_path (str): Path to save transcription results
    
    Returns:
        dict: JSON with video_id and transcribed segments
    """
    # Step 1: Download chat and process activity
    print("Downloading and processing chat activity...")
    download_chat(vod_url)
    chat_message_activity_to_json(vod_url, interval_seconds, chat_output_path)
    
    # Step 2: Load chat activity data
    with open(chat_output_path, 'r', encoding='utf-8') as f:
        chat_activity = json.load(f)
    
    # Step 3: Get top N segments by message count
    top_segments = list(chat_activity.items())[:top_n]
    print(f"Selected top {len(top_segments)} segments for transcription:")
    for time_range, count in top_segments:
        print(f"  {time_range}: {count} messages")
    
    # Step 4: Initialize speech-to-text model
    print("Initializing speech-to-text model...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    # Using Distil-Whisper-Large-v3 for best speed/accuracy balance
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
        torch_dtype=torch_dtype,
        device=device,
    )
    
    # Step 5: Load audio file
    print(f"Loading audio file: {audio_file_path}")
    audio = AudioSegment.from_file(audio_file_path)
    
    # Step 6: Transcribe each segment
    transcriptions = {}
    video_id = os.path.splitext(os.path.basename(audio_file_path))[0]
    
    print("Transcribing segments...")
    for time_range, message_count in tqdm(top_segments, desc="Processing segments"):
        try:
            # Parse time range (e.g., "161:00-161:05")
            start_str, end_str = time_range.split('-')
            start_min, start_sec = map(int, start_str.split(':'))
            end_min, end_sec = map(int, end_str.split(':'))
            
            # Convert to milliseconds for pydub
            start_ms = (start_min * 60 + start_sec) * 1000
            end_ms = (end_min * 60 + end_sec) * 1000
            
            # Extract audio segment
            audio_segment = audio[start_ms:end_ms]
            
            # Convert to numpy array for the model
            audio_array = audio_segment.get_array_of_samples()
            
            # Convert to float and normalize
            import numpy as np
            audio_np = np.array(audio_array, dtype=np.float32)
            
            # Normalize audio
            if audio_segment.channels == 2:
                # Convert stereo to mono
                audio_np = audio_np.reshape((-1, 2)).mean(axis=1)
            
            # Normalize to [-1, 1] range
            audio_np = audio_np / (2**15)  # 16-bit audio normalization
            
            # Transcribe
            result = pipe({"array": audio_np, "sampling_rate": audio_segment.frame_rate})
            
            transcriptions[time_range] = {
                "text": result["text"].strip(),
                "message_count": message_count,
                "duration_seconds": interval_seconds
            }
            
        except Exception as e:
            print(f"Error processing segment {time_range}: {str(e)}")
            transcriptions[time_range] = {
                "text": "[ERROR: Could not transcribe]",
                "message_count": message_count,
                "duration_seconds": interval_seconds,
                "error": str(e)
            }
    
    # Step 7: Prepare final result
    result = {
        "video_id": video_id,
        "transcribed_segments": transcriptions,
        "total_segments_processed": len(transcriptions),
        "model_used": model_id
    }
    
    # Step 8: Save transcription results
    with open(transcription_output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"Transcription completed! Results saved to: {transcription_output_path}")
    
    return result


def process_chat_with_model_choice(vod_url, audio_file_path, top_n=10, interval_seconds=5,
                                   model_choice="distil-large-v3"):
    """
    Alternative function with model selection options.
    
    Args:
        model_choice (str): Choose from:
            - "distil-large-v3": Fast and accurate (recommended)
            - "distil-large-v3.5": Fastest, latest model
            - "whisper-large-v3-turbo": Best accuracy, slower
            - "distil-small.en": Smallest, for memory-constrained setups
    """
    model_options = {
        "distil-large-v3": "distil-whisper/distil-large-v3",
        "distil-large-v3.5": "distil-whisper/distil-large-v3.5", 
        "whisper-large-v3-turbo": "openai/whisper-large-v3-turbo",
        "distil-small.en": "distil-whisper/distil-small.en"
    }
    
    if model_choice not in model_options:
        print(f"Invalid model choice. Available options: {list(model_options.keys())}")
        model_choice = "distil-large-v3"
    
    print(f"Using model: {model_choice} ({model_options[model_choice]})")
    
    # Use the main function with selected model
    return process_chat(vod_url, audio_file_path, top_n, interval_seconds)

