from utils.utils import download_twitch_vod, download_chat, chat_message_activity_to_json
from utils.video import analyze_video_emotions

def main():
    vod_url = "https://www.twitch.tv/videos/2465574148"
    download_twitch_vod(vod_url)
    #download_chat(vod_url)
    chat_message_activity_to_json(vod_url)
    analyze_video_emotions("/mnt/d/Projects/twitch_videos/twitch_2465574148.mp4", interval_sec=5, time_period="10:00-20:00")

if __name__ == "__main__":
    main()