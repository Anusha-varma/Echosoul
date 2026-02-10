import subprocess

def fix_video_audio(input_video, output_video):
    cmd = f'ffmpeg -y -i "{input_video}" -map 0:v:0 -map 0:a:0 -c:v libx264 -c:a aac "{output_video}"'
    subprocess.run(cmd, shell=True, check=True)
