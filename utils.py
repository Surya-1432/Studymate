import os
from datetime import datetime
import json

def timestamp_now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def save_session_transcript(session_history):
    os.makedirs("data/transcripts", exist_ok=True)
    filename = f"data/transcripts/transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(session_history, f, ensure_ascii=False, indent=2)
    return filename
