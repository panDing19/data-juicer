import os
import cv2
import clip
import torch
import numpy as np
from einops import rearrange
from typing import Dict, List, Union, Tuple, Any, Optional
from torchvision.datasets.folder import pil_loader
from PIL import Image
from loguru import logger 

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.ops.base_op import OPERATORS, Filter
from ...utils.model_utils import get_model, prepare_model

# Points for frame extraction similar to inference.py
NUM_FRAMES_POINTS = {
    1: (0.5,),
    2: (0.25, 0.75),
    3: (0.1, 0.5, 0.9),
}


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


def is_video(path):
    """Check if a file is a video based on its extension."""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    _, ext = os.path.splitext(path.lower())
    return ext in video_extensions

@OPERATORS.register_module('my_aesthetic_filter')
class MyAestheticFilter(Filter):
    _accelerator = "cuda"

    def __init__(self,
        weight_model_path: Optional[str] = None,
        clip_model_path: Optional[str] = None,
        num_frames: int = 3,
        verbose: bool = False,
        threshold: float = 5.0,
        *args, **kwargs
    ):
        kwargs["mem_required"] = "1500MB" if kwargs.get("mem_required", 0) == 0 else kwargs["mem_required"]
        super().__init__(*args, **kwargs)

        # Validate required parameters
        if weight_model_path is None:
            raise ValueError("weight_model_path is required but not provided")
        if clip_model_path is None:
            raise ValueError("clip_model_path is required but not provided")

        self.num_frames = num_frames
        self.points = NUM_FRAMES_POINTS[num_frames]
        self.verbose = verbose
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cpu':
            logger.warning("No GPU available, using CPU which may be slow.")
        self.threshold = threshold



        # Load CLIP model to get preprocessing transforms
        if clip_model_path is not None:
            _, self.clip_preprocess = clip.load(clip_model_path, device='cpu')
        else:
            raise(ValueError("I don't want it load from downloading"))
        
        # self.model = AestheticScorer(768, self.device, clip_model_path=clip_model_path, model_path=weight_model_path)
        self.model_key = prepare_model(
            model_type="my_aesthetic",
            clip_model_path=clip_model_path,
            weight_model_path=weight_model_path,
        )


        if self.verbose:
            print(f"AestheticFilterPreprocess initialized with {num_frames} frames")
    
    def preprocess_clip(self, clip_path: str) -> torch.Tensor:
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

            return processed_tensor

        except Exception as e:
            if self.verbose:
                print(f"Error preprocessing {clip_path}: {e}")
            return None

    def compute_stats_single(self, sample, rank=None, context=False):
        if StatsKeys.my_video_aesthetic_score in sample[Fields.stats]:
            return sample
        
        # there is no video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            sample[Fields.stats][StatsKeys.my_video_aesthetic_score] = np.array([], dtype=np.float64)
            return sample
        
         # load videos
        clip_path_list = sample[self.video_key]
        aesthetics_scores = []
        for clip_path in clip_path_list:
            processed_frames = self.preprocess_clip(clip_path)

            assert processed_frames is not None, f"Failed to preprocess {clip_path}"
            score = self.score_frames(processed_frames, rank=rank)
            aesthetics_scores.append(score)

        sample[Fields.stats][StatsKeys.my_video_aesthetic_score] = aesthetics_scores
        return sample
    
    def process_single(self, sample):
        aesthetics_scores = sample[Fields.stats][StatsKeys.my_video_aesthetic_score]
        if len(aesthetics_scores) <= 0:
            return True
        
        for score in aesthetics_scores:
            if score < self.threshold:
                return False
        return True
        
    def score_frames(self, processed_frames: np.ndarray, rank=None) -> float:
        """Calculate the aesthetic score for preprocessed frame tensors.

        Args:
            processed_frames: Preprocessed tensor of frames ready for the model

        Returns:
            Aesthetic score (0-10 scale)
        """
        try:
            # processed_frames = torch.from_numpy(processed_frames)
            if processed_frames is None or processed_frames.numel() == 0:
                return 0.0

            # Move to device and get score
            processed_frames = processed_frames.to(self.device)
            model = get_model(self.model_key, rank=rank, use_cuda=self.use_cuda())
            with torch.no_grad():
                scores = model(processed_frames)

            # Average score across frames and scale to 0-10
            score = scores.mean().item() * 10.0

            return score

        except Exception as e:
            if self.verbose:
                print(f"Error scoring frames: {e}")
            return 0.0

        
        
    def preprocess(self, batch: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Process a batch of clips and extract preprocessed frames.

        Args:
            batch: Dictionary containing lists of clip paths
                Format: {"clips": [clip1, clip2, ...]}

        Returns:
            Dictionary with clip paths and preprocessed frame tensors
        """
        clip_path_list = batch["clips"]

        # Process each clip
        all_processed_frames = []
        valid_clip_paths = []

        for clip_path in clip_path_list:
            processed_frames = self.preprocess_clip(clip_path)

            if processed_frames is not None:
                all_processed_frames.append(processed_frames)
                valid_clip_paths.append(clip_path)
            elif self.verbose:
                print(f"Warning: No frames extracted from {clip_path}")

        return {
            "clips": valid_clip_paths,
            "frames": all_processed_frames,
        }
    
