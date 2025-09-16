import json
import os

video_dir = '/home/panding/Code/ray_for_opensora/file_version/videos'
output_path = 'demo.jsonl'
with open(output_path, 'w') as f:
    for video_name in os.listdir(video_dir):
        video_path = os.path.join(video_dir, video_name)
        data = {'videos': [video_path]}
        data['text'] = 'just for fake'
        f.write(json.dumps(data, ensure_ascii=False) + '\n')