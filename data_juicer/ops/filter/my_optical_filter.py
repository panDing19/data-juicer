import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
from loguru import logger
from torch.amp import autocast
from torchvision.transforms.functional import pil_to_tensor
from typing import List, Optional

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.ops.base_op import OPERATORS, Filter

from ...utils.model_utils import get_model, prepare_model

@OPERATORS.register_module('my_optical_filter')
class MyOpticalFilter(Filter):
    _accelerator = "cuda"

    def __init__(
        self,
        model_path: Optional[str] = None,
        num_frames: int = 4,
        threshold: float = 5.0,
        # device: Optional[str] = None,
        verbose: bool = False,
        *args,
        **kwargs
    ):
        kwargs["mem_required"] = "1500MB" if kwargs.get("mem_required", 0) == 0 else kwargs["mem_required"]
        super().__init__(*args, **kwargs)

        # Validate required parameters
        if model_path is None:
            raise ValueError("model_path is required but not provided")

        self.num_frames = num_frames
        self.verbose = verbose
        self.threshold = threshold
        # if device is None:
        #     self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #     if self.device.type == "cpu":
        #         print("Warning: No GPU available, using CPU which may be slow.")
        # else:
        #     self.device = device
        # self.device = torch.device(self.device)

        # self.model = UniMatch(
        #     feature_channels=128,
        #     num_scales=2,
        #     upsample_factor=4,
        #     num_head=1,
        #     ffn_dim_expansion=4,
        #     num_transformer_layers=6,
        #     reg_refine=True,
        #     task="flow",
        # )

        self.device = 'cuda'

        if verbose:
            print(f"Optical Flow Preprocessor initialized to extract {num_frames+1} consecutive frames")
        
        self.model_key = prepare_model(model_type='my_optical', model_path=model_path)
        
        # # Load checkpoint
        # if os.path.exists(model_path):
        #     ckpt = torch.load(model_path, map_location=self.device)
        #     self.model.load_state_dict(ckpt["model"])
        #     self.model = self.model.to(self.device)
        #     self.model.eval()
        #     # self.model.half()  # Use half precision
        #     if verbose:
        #         print(f"Loaded optical flow model from {model_path}")
        # else:
        #     raise FileNotFoundError(f"Model file not found: {model_path}")
    
        # torch.set_float32_matmul_precision('high')
        # # self.model = torch.compile(self.model)
        # samples = self.generate_input(1)
        # images = samples[0].to(self.device)
        # with torch.no_grad():
        #     # with autocast('cuda'):
        #     B = 1
        #     batch_0 = rearrange(images[:-1].unsqueeze(0), "B N C H W -> (B N) C H W").contiguous()
        #     batch_1 = rearrange(images[1:].unsqueeze(0), "B N C H W -> (B N) C H W").contiguous()
        #     self.model(batch_0, batch_1, 
        #             attn_type="swin",
        #                 attn_splits_list=[2, 8],
        #                 corr_radius_list=[-1, 4],
        #                 prop_radius_list=[-1, 1],
        #                 num_reg_refine=6,
        #                 task="flow",
        #                 pred_bidir_flow=False,
        #     )

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

    def generate_input(self, num_samples: int, num_frames: int = 4) -> List[torch.Tensor]:
        """Generate synthetic input data for optical flow testing.

        Args:
            num_samples: Number of clips to generate
            num_frames: Number of frames per clip

        Returns:
            List of dictionaries with clips and preprocessed frame tensors
        """


        batches = []
        for i in range(num_samples):
            # Generate synthetic frame tensors for optical flow processing
            B = 1
            images = torch.randn(num_frames + 1, 3, 320, 576)  # Standard size used in optical flow
            batches.append(images)
        
        return batches

    def compute_stats_single(self, sample, rank=None, context=False):
        if StatsKeys.my_optical_score in sample[Fields.stats]:
            return sample
        
        # there is no video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            sample[Fields.stats][StatsKeys.my_optical_score] = np.array([], dtype=np.float64)
            return sample
        
        # load videos
        clip_path_list = sample[self.video_key]
        optical_scores = []
        for clip_path in clip_path_list:
            # Extract frames from the clip
            frames = self.extract_frames(clip_path)
            if len(frames) >= self.num_frames + 1:
                # Preprocess frames to tensors
                frame_tensor = self.preprocess_frames(frames)
                assert frame_tensor is not None, f"Failed to preprocess frames from {clip_path}"
                optical_score = self.compute_optical_flow_score(frame_tensor, rank=rank)
                optical_scores.append(optical_score)
            else:
                raise ValueError(f"Not enough frames extracted from {clip_path}")
        sample[Fields.stats][StatsKeys.my_optical_score] = optical_scores
        return sample
        
    
    def process_single(self, sample):
        optical_scores = sample[Fields.stats][StatsKeys.my_optical_score]
        if len(optical_scores) <= 0:
            return True
        
        for score in optical_scores:
            if score > self.threshold:
                logger.info(f"Filtering out sample due to high optical flow score: {score} > {self.threshold}")
                return False
        return True
    
    def compute_optical_flow_score(self, images: torch.Tensor, rank=None) -> float:
        """Compute the optical flow score for a sequence of frames."""
        try:
            with torch.no_grad():
                model = get_model(self.model_key, rank=rank, use_cuda=self.use_cuda())
                # with autocast('cuda'):
                B = 1  # Batch size is 1 for a single clip
                # Move images to the same device as the model
                images = images.to(self.device)
                batch_0 = rearrange(images[:-1].unsqueeze(0), "B N C H W -> (B N) C H W").contiguous()
                batch_1 = rearrange(images[1:].unsqueeze(0), "B N C H W -> (B N) C H W").contiguous()
                # with autocast(enabled=True):
                res = model(
                    batch_0,
                    batch_1,
                    attn_type="swin",
                    attn_splits_list=[2, 8],
                    corr_radius_list=[-1, 4],
                    prop_radius_list=[-1, 1],
                    num_reg_refine=6,
                    task="flow",
                    pred_bidir_flow=False,
                )
                flow_maps = res["flow_preds"][-1].cpu()  # [B * (N-1), 2, H, W]
                flow_maps = rearrange(flow_maps, "(B N) C H W -> B N H W C", B=B)
                flow_score = flow_maps.abs().mean(dim=[1, 2, 3, 4]).item()
                
                return flow_score
        except Exception as e:
            if self.verbose:
                print(f"Error computing optical flow: {str(e)}")
            return float('inf')  # Return a high score to filter out problematic clips

    def preprocess_frames(self, frames: List[Image.Image]) -> Optional[torch.Tensor]:
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
            
            return images
        except Exception as e:
            if self.verbose:
                print(f"Error preprocessing frames: {str(e)}")
            return None
    