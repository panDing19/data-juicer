import json
import os

video_dir = '/mnt_fast/dataset/videos'
output_path = 'demo.jsonl'
n = 1
i = 0
with open(output_path, 'w') as f:
    for video_name in os.listdir(video_dir):
        video_path = os.path.join(video_dir, video_name)
        data = {'videos': [video_path]}
        data['text'] = 'just for fake'
        f.write(json.dumps(data, ensure_ascii=False) + '\n')
        i +=1 
        if i >= n:
            break