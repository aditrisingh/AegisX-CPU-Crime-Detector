from moviepy.editor import VideoFileClip

# Load your video
clip = VideoFileClip("last hope/Screen Recording 2025-07-25 224830.mp4")

# Get the last 10 seconds
start_time = max(0, clip.duration - 10)
end_time = clip.duration
last_10_sec_clip = clip.subclip(start_time, end_time)

# Export as GIF
last_10_sec_clip.write_gif("aegis_demo_last10.gif", fps=10)
