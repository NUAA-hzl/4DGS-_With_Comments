import os
import cv2
import numpy as np
import glob

source_folder = '/root/My4DGaussians/data/dynerf/cut_roasted_beef' 
target_base = '/root/autodl-tmp/data/dynerf/cut_roasted_beef'  
os.makedirs(target_base, exist_ok=True)
for video_path in glob.glob(os.path.join(source_folder, '*.mp4')):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    target_folder = os.path.join(target_base, video_name, 'images')
    os.makedirs(target_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_filename = os.path.join(target_folder, f'{frame_count:04d}.png')
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
    cap.release()
    print(f'Processed {video_name} into {frame_count} frames.')

print('All videos have been processed.')