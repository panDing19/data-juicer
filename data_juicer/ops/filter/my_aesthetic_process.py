import clip
import torch
import numpy as np
from typing import  Optional
from loguru import logger 

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.ops.base_op import OPERATORS, Filter
from ...utils.model_utils import get_model, prepare_model


OP_NAME = 'my_aesthetic_process'

@OPERATORS.register_module(OP_NAME)
class MyAestheticProcess(Filter):
    _accelerator = "cuda"

    def __init__(self,
        weight_model_path: Optional[str] = None,
        clip_model_path: Optional[str] = None,
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
        
        self.model_key = prepare_model(
            model_type="my_aesthetic",
            clip_model_path=clip_model_path,
            weight_model_path=weight_model_path,
        )


    def compute_stats_single(self, sample, rank=None, context=False, *args, **kwargs):
        if StatsKeys.my_aesthetic_score in sample[Fields.stats]:
            return sample
        
        # there is no video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            sample[Fields.stats][StatsKeys.my_aesthetic_score] = np.array([], dtype=np.float64)
            return sample
        
         # load videos
        processed_frames_list = sample[Fields.my_aes_preprocessed_frames]
        assert len(processed_frames_list) > 0, f"There is no preprocessed frames in {sample}"
        aesthetics_scores = []
        for processed_frames in processed_frames_list:
            score = self.score_frames(processed_frames, rank=rank)
            aesthetics_scores.append(score)

        del sample[Fields.my_aes_preprocessed_frames]
        sample[Fields.stats][StatsKeys.my_aesthetic_score] = aesthetics_scores
        return sample
    
    def process_single(self, sample):
        aesthetics_scores = sample[Fields.stats][StatsKeys.my_aesthetic_score]
        if len(aesthetics_scores) <= 0:
            return True
        
        for score in aesthetics_scores:
            if score < self.threshold:
                logger.warning(f"Filtering out sample due to low aesthetic score: {score} < {self.threshold}")
                return False
        return True
        
    def score_frames(self, processed_frames: np.ndarray, rank=None) -> float:
        """Calculate the aesthetic score for preprocessed frame tensors.

        Args:
            processed_frames: Preprocessed tensor of frames ready for the model

        Returns:
            Aesthetic score (0-10 scale)
        """
        # try:
        processed_frames = torch.from_numpy(processed_frames)
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

        # except Exception as e:
        #     if self.verbose:
        #         print(f"Error scoring frames: {e}")
        #     return 0.0

  
    
