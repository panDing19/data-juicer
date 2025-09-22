import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from PIL import Image
from einops import rearrange
from typing import List, Optional
from torchvision.transforms.functional import pil_to_tensor

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.ops.base_op import OPERATORS, Mapper

OP_NAME = "my_optical_preprocess"

@OPERATORS.register_module(OP_NAME)
class MyOpticalPreprocess(Mapper):
    """
    CPU-based preprocessing for optical flow filtering.
    Extracts frames from video clips for later flow computation.
    """
    
    def __init__(
        self,
        num_frames: int = 4,
        verbose: bool = False,
        *args,
        **kwargs,
    ):
        """
        Initialize the optical flow preprocessor.
        
        Args:
            num_frames: Number of frames to sample per clip for flow calculation.
            verbose: Whether to print verbose output.
        """
        super().__init__(*args, **kwargs)
        self.num_frames = num_frames
        self.verbose = verbose
        
        if verbose:
            print(f"Optical Flow Preprocessor initialized to extract {num_frames+1} consecutive frames")
    
    def extract_frames(self, video_path: str) -> List[Image.Image]:
        """Extract frames from a video for optical flow computation."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                if self.verbose:
                    print(f"Failed to open video: {video_path}")
                return []
                
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= 1:
                if self.verbose:
                    print(f"Video has only {frame_count} frame(s): {video_path}")
                cap.release()
                return []
                
            # Calculate frame indices to extract
            # Get frames evenly distributed throughout the video
            frame_indices = np.linspace(0, frame_count-1, self.num_frames+1, dtype=int)[:self.num_frames+1]
            
            frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert numpy array to PIL image
                pil_img = Image.fromarray(frame)
                frames.append(pil_img)
                
            cap.release()
            
            if len(frames) < 2:
                if self.verbose:
                    print(f"Failed to extract enough frames from {video_path}")
                return []
                
            return frames
            
        except Exception as e:
            if self.verbose:
                print(f"Error extracting frames from {video_path}: {str(e)}")
            return []
    
    def preprocess_frames(self, frames: List[Image.Image]) -> Optional[np.ndarray]:
        """Convert PIL images to properly sized tensors for the model."""
        # Stack frames and convert to tensor
        try:
            images = torch.stack([pil_to_tensor(x) for x in frames])
            images = images.float()
            
            # Resize based on aspect ratio
            H, W = images.shape[-2:]
            if H > W:
                images = rearrange(images, "N C H W -> N C W H")
            images = F.interpolate(images, size=(320, 576), mode="bilinear", align_corners=True)
            
            return images.numpy()
        except Exception as e:
            if self.verbose:
                print(f"Error preprocessing frames: {str(e)}")
            return None
    
    
    def process_single(self, sample, context=False):
        if StatsKeys.my_optical_score in sample[Fields.stats]:
            return sample
        
        # there is no video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            sample[Fields.stats][StatsKeys.my_optical_score] = np.array([], dtype=np.float64)
            return sample
        
         # load videos
        clip_path_list = sample[self.video_key]
        sample[Fields.my_optical_preprocessed_frames] = []
        for clip_path in clip_path_list:
            frames = self.extract_frames(clip_path)
            processed_frames = self.preprocess_frames(frames)
            sample[Fields.my_optical_preprocessed_frames].append(processed_frames)

        return sample