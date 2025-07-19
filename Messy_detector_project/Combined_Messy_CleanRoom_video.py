import os
from gtts import gTTS
import subprocess

os.environ["PATH"] += os.pathsep + r"C:\Anaconda\envs\messy_env\Library\bin"


# === Step 1: Define files ===
messy_video = "Output_MessyRoom.avi"
clean_video = "Output_CleanRoom.avi"
messy_audio = "messy_voice.mp3"
clean_audio = "clean_voice.mp3"
messy_output = "messy_with_voice.mp4"
clean_output = "clean_with_voice.mp4"
final_output = "final_combined_video.mp4"

# === Step 2: Generate voice-over ===
messy_text = "Mess detected.Let's fix it!."
clean_text = "Great!you did it!"

gTTS(messy_text).save(messy_audio)
gTTS(clean_text).save(clean_audio)

print(" Voice files created.")

# === Step 3: Merge audio with each video ===
def merge_audio_video(video_file, audio_file, output_file):
    cmd = [
        "ffmpeg", "-y",
        "-i", video_file,
        "-i", audio_file,
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        output_file
    ]
    subprocess.run(cmd)
    print(f" Merged: {output_file}")

merge_audio_video(messy_video, messy_audio, messy_output)
merge_audio_video(clean_video, clean_audio, clean_output)

# === Step 4: Combine both videos ===
def combine_videos(video1, video2, output_file):
    # First create a text file for ffmpeg to read
    with open("videos_to_merge.txt", "w") as f:
        f.write(f"file '{os.path.abspath(video1)}'\n")
        f.write(f"file '{os.path.abspath(video2)}'\n")
    
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", "videos_to_merge.txt",
        "-c", "copy",
        output_file
    ]
    subprocess.run(cmd)
    print(f" Final combined video: {output_file}")

combine_videos(messy_output, clean_output, final_output)

# === Optional: Clean up temp files ===
os.remove("videos_to_merge.txt")
# os.remove(messy_audio)
# os.remove(clean_audio)
# os.remove(messy_output)
# os.remove(clean_output)
