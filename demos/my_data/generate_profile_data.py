import json
import os

file_path_dir = '/mnt_fast/new_dataset/panda70m_validation_video_paths.txt'
output_path = 'profile_complete.jsonl'
n = 5721
i = 0
with open(output_path, 'w') as f:
    with open(file_path_dir, 'r') as fr:
        for line in fr:
            video_path = line.strip()
            data = {'videos': [video_path]}
            data['text'] = 'just for fake'
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
            i += 1
            if i >= n:
                break