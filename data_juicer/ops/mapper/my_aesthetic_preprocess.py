import cv2
import os
import torch
import clip
import numpy as np
from typing import Dict, List, Any
from torchvision.datasets.folder import pil_loader
from PIL import Image

from data_juicer.utils.constant import Fields, StatsKeys
from ..base_op import OPERATORS, Mapper

OP_NAME = "my_aesthetic_preprocess"

NUM_FRAMES_POINTS = {
    1: (0.5,),
    2: (0.25, 0.75),
    3: (0.1, 0.5, 0.9),
}

def is_video(path):
    """Check if a file is a video based on its extension."""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    _, ext = os.path.splitext(path.lower())
    return ext in video_extensions

def extract_frames(video_path, points=(0.1, 0.5, 0.9), num_frames=None):
    """Extract frames from a video at specific points.
    
    Args:
        video_path: Path to the video file
        points: Tuple of points (0-1) to extract frames from
        num_frames: Total number of frames (if known)
        
    Returns:
        List of PIL images
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    
    # Get video properties
    if num_frames is None:
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame indices to extract
    frame_indices = [int(p * (num_frames - 1)) for p in points]
    
    # Extract frames
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Failed to read frame {idx} from {video_path}")
            continue
            
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame)
        frames.append(pil_img)
    
    cap.release()
    return frames

@OPERATORS.register_module(OP_NAME)
class MyAestheticPreprocess(Mapper):
    def __init__(self,
        clip_model_path = None,
        num_frames: int = 3,
        verbose: bool = False,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.num_frames = num_frames
        self.points = NUM_FRAMES_POINTS[num_frames]
        self.verbose = verbose


        # Load CLIP model to get preprocessing transforms
        if clip_model_path is not None:
            _, self.clip_preprocess = clip.load(clip_model_path, device='cpu')
        else:
            raise(ValueError("I don't want it load from downloading"))
            _, self.clip_preprocess = clip.load("ViT-L/14", device='cpu')

        if self.verbose:
            print(f"AestheticFilterPreprocess initialized with {num_frames} frames")
    
    def preprocess_clip(self, clip_path: str) -> np.ndarray:
        """Extract and prepare frames from a video/image, then preprocess for CLIP.

        Args:
            clip_path: Path to the video/image file

        Returns:
            Preprocessed tensor ready for the aesthetic model
        """
        try:
            # Extract PIL images
            if not is_video(clip_path):
                pil_frames = [pil_loader(clip_path)]
            else:
                pil_frames = extract_frames(clip_path, points=self.points)

            if not pil_frames:
                return None

            # Apply CLIP preprocessing to each frame
            processed_frames = [self.clip_preprocess(img) for img in pil_frames]
            processed_tensor = torch.stack(processed_frames)

            return processed_tensor.numpy()

        except Exception as e:
            if self.verbose:
                print(f"Error preprocessing {clip_path}: {e}")
            return None
    

    
    def process_single(self, sample, context=False):
        if Fields.stats in sample and StatsKeys.my_aesthetic_score in sample[Fields.stats]:
            return sample
        
        # there is no video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            sample[Fields.stats][StatsKeys.my_aesthetic_score] = np.array([], dtype=np.float64)
            return sample
        
         # load videos
        clip_path_list = sample[self.video_key]
        sample[Fields.my_aes_preprocessed_frames] = []
        for clip_path in clip_path_list:
            processed_frames = self.preprocess_clip(clip_path)
            sample[Fields.my_aes_preprocessed_frames].append(processed_frames)
        
        return sample