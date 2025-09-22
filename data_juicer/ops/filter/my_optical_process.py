import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from loguru import logger
from typing import Optional

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.ops.base_op import OPERATORS, Filter

from ...utils.model_utils import get_model, prepare_model

OP_NAME = "my_optical_process"

@OPERATORS.register_module(OP_NAME)
class MyOpticalProcess(Filter):
    _accelerator = "cuda"

    def __init__(
        self,
        model_path: Optional[str] = None,
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

        self.verbose = verbose
        self.threshold = threshold

        self.device = 'cuda'

        self.model_key = prepare_model(model_type='my_optical', model_path=model_path)
        

    def compute_stats_single(self, sample, rank=None, context=False, *args, **kwargs):
        if StatsKeys.my_optical_score in sample[Fields.stats]:
            return sample
        
        # there is no video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            sample[Fields.stats][StatsKeys.my_optical_score] = np.array([], dtype=np.float64)
            return sample
        
        # load videos
        processed_frames_list = sample[Fields.my_optical_preprocessed_frames]
        optical_scores = []
        for processed_frames in processed_frames_list:
            # Preprocess frames to tensors
            optical_score = self.compute_optical_flow_score(processed_frames, rank=rank)
            optical_scores.append(optical_score)
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
    
    def compute_optical_flow_score(self, images: np.ndarray, rank=None) -> float:
        """Compute the optical flow score for a sequence of frames."""
        try:
            images = torch.from_numpy(images).to(self.device)
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

    