
import cv2
import os
import numpy as np
from typing import List, Any

from .my_ocr_utils import resize_aspect_ratio, normalizeMeanVariance
from data_juicer.utils.constant import Fields, StatsKeys
from ..base_op import OPERATORS, Mapper

OP_NAME = "my_ocr_preprocess"

@OPERATORS.register_module(OP_NAME)
class MyOCRPreprocess(Mapper):
    def __init__(self,
            num_frames=3,
            verbose=False,
            *args,
            **kwargs):
        super().__init__(*args, **kwargs)
        self.num_frames = num_frames
        self.verbose = verbose
        
        if verbose:
            print(f"OCR Filter Preprocessor initialized to extract {num_frames} frames")


    def extract_frames(self, clip_path: str) -> list[np.ndarray]:
        try:
            # For videos, extract frames at uniform intervals
            cap = cv2.VideoCapture(clip_path)
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame indices to sample
            if total_frames <= self.num_frames:
                # If video has fewer frames than requested, take all frames
                frame_indices = list(range(total_frames))
            else:
                # Sample frames at uniform intervals
                frame_indices = [int(i * total_frames / self.num_frames) 
                                        for i in range(self.num_frames)]
            
            # Extract the frames
            frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # # Convert to PIL Image
                    # pil_frame = Image.fromarray(frame_rgb)
                    frames.append(frame_rgb)
            
            # Release the video capture
            cap.release()
            
                
            return frames
            
        except Exception as e:
            # If any error occurs, try to open as an image
        
            print(f"Error processing {clip_path}: {e}")
            return []
        
    def preprocess(self, images: List[np.ndarray], canvas_size = 720, mag_ratio = 1.):
        assert type(images) == list
        for i, image in enumerate(images):
            if len(image.shape) == 2: # grayscale
                img_cv_grey = image
                img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 1:
                img_cv_grey = np.squeeze(image)
                img = cv2.cvtColor(img_cv_grey, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 3: # BGRscale
                img = image
            elif len(image.shape) == 3 and image.shape[2] == 4: # RGBAscale
                img = image[:,:,:3]
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            images[i] = img
        total_area = img.shape[0] * img.shape[1]
        img_resized_list = []
        # resize
        for img in images:
            img_resized, target_ratio, size_heatmap = resize_aspect_ratio(img, canvas_size,
                                interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
            img_resized_list.append(img_resized)
        ratio_h = ratio_w = 1 / target_ratio
        # preprocessing
        x = [np.transpose(normalizeMeanVariance(n_img), (2, 0, 1))
            for n_img in img_resized_list]
        x = np.array(x)
        return x, ratio_h, ratio_w, total_area

    
    def process_single(self, sample, context=False):
        if StatsKeys.my_ocr_score in sample[Fields.stats]:
            return sample
        
        # there is no video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            sample[Fields.stats][StatsKeys.my_ocr_score] = np.array([], dtype=np.float64)
            return sample
        
         # load videos
        clip_path_list = sample[self.video_key]
        all_frames = []
        ratio_h_list = []
        ratio_w_list = []
        total_area_list = []
        for clip_path in clip_path_list:
            frames = self.extract_frames(clip_path)
            assert frames is not None, f"Failed to extract frames from {clip_path}"
            processed_frames, ratio_h, ratio_w, total_area = self.preprocess(frames)
            all_frames.append(processed_frames)
            ratio_h_list.append(ratio_h)
            ratio_w_list.append(ratio_w)
            total_area_list.append(total_area)
        sample[Fields.my_ocr_preprocessed_frames] = all_frames 
        sample[StatsKeys.my_ocr_ratio_h] = ratio_h_list
        sample[StatsKeys.my_ocr_ratio_w] = ratio_w_list
        sample[StatsKeys.my_ocr_total_area] = total_area_list

        return sample
