import os

# Source folder on your Desktop
video_dir = os.path.expanduser("~/Desktop/jepa_videos")

# This automatically points to the folder your terminal is currently sitting in
current_directory = os.getcwd()
output_csv = os.path.join(current_directory, "vjepa_dataset.csv")

# Video formats V-JEPA accepts
video_extensions = (".mp4", ".mov", ".avi", ".mkv", ".webm")

if not os.path.exists(video_dir):
    print(f"❌ Error: Could not find the folder '{video_dir}' on your Desktop.")
    exit(1)

print(f"Scanning Desktop/jepa_videos...")
count = 0

# Opening in 'w' mode automatically clears/overwrites the file if it already exists
with open(output_csv, "w") as f:
    for root, dirs, files in os.walk(video_dir):
        for file in files:
            if file.lower().endswith(video_extensions):
                # Safety check for spaces
                if " " in file or " " in root:
                    print(f"⚠️ WARNING: Space detected! Rename this file/folder: '{file}'")
                    continue
                
                # Construct absolute path and write format: /path/to/video.mp4 0
                absolute_path = os.path.join(root, file)
                f.write(f"{absolute_path} 0\n")
                count += 1

print(f"🎉 Success! Processed {count} videos.")
print(f"📁 Overwritten and saved directly to: {output_csv}")
