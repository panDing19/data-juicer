import cv2
import numpy as np
import torch
from torch.cuda.amp import autocast
from typing import Any, List

from loguru import logger

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.ops.base_op import OPERATORS, Filter

from .my_ocr_utils import resize_aspect_ratio, normalizeMeanVariance, copyStateDict, getDetBoxes,\
        adjustResultCoordinates, group_text_box, diff, triangle_area
# from .ocr_model import CRAFT
from ...utils.model_utils import get_model, prepare_model


@OPERATORS.register_module('my_ocr_filter')
class MyOCRFilter(Filter):
    _accelerator = "cuda"

    def __init__(
            self, 
            model_path,
            device = None,
            num_frames=3, 
            threshold=0.05,
            verbose=False, 
            *args, 
            **kwargs
        ):
        kwargs["mem_required"] = "1500MB" if kwargs.get("mem_required", 0) == 0 else kwargs["mem_required"]
        super().__init__(*args, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cpu":
            print("Warning: No GPU available, using CPU which may be slow.")
        self.num_frames = num_frames
        self.verbose = verbose
        # self.model = CRAFT()
        # self.model.load_state_dict(copyStateDict(torch.load(model_path, map_location=device, weights_only=False)))
        # self.model = self.model.to(device)
        # self.model.eval()
        self.model_key = prepare_model(model_type='my_ocr', model_path=model_path)
        self.threshold = threshold

        if self.verbose:
            print(f"OCRFilter initialized with {num_frames} frames")
    

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
    
    def postprocess(self, y, ratio_h, ratio_w, total_area,
            text_threshold = 0.7, link_threshold = 0.4, low_text = 0.4, poly = False,
            optimal_num_chars=None, slope_ths = 0.1, ycenter_ths = 0.5, height_ths = 0.5,
            width_ths = 0.5, add_margin = 0.1, min_size = 20):
        boxes_list, polys_list = [], []
        estimate_num_chars = optimal_num_chars is not None
        for out in y:
            # make score and link map
            score_text = out[:, :, 0]
            score_link = out[:, :, 1]

            # Post-processing
            boxes, polys, mapper = getDetBoxes(
                score_text, score_link, text_threshold, link_threshold, low_text,\
                    poly, estimate_num_chars)

            # coordinate adjustment
            boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
            polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
            if estimate_num_chars:
                boxes = list(boxes)
                polys = list(polys)
            for k in range(len(polys)):
                if estimate_num_chars:
                    boxes[k] = (boxes[k], mapper[k])
                if polys[k] is None:
                    polys[k] = boxes[k]
            boxes_list.append(boxes)
            polys_list.append(polys)

        if estimate_num_chars:
            polys_list = [[p for p, _ in sorted(polys, key=lambda x:
                abs(optimal_num_chars - x[1]))] for polys in polys_list]

        text_box_list = []
        for polys in polys_list:
            single_img_result = []
            for i, box in enumerate(polys):
                poly = np.array(box).astype(np.int32).reshape((-1))
                single_img_result.append(poly)
            text_box_list.append(single_img_result)

        horizontal_list_agg, free_list_agg = [], []
        for text_box in text_box_list:
            horizontal_list, free_list = group_text_box(text_box, slope_ths,
                                                        ycenter_ths, height_ths,
                                                        width_ths, add_margin,
                                                        (optimal_num_chars is None))
            if min_size:
                horizontal_list = [i for i in horizontal_list if max(
                    i[1] - i[0], i[3] - i[2]) > min_size]
                free_list = [i for i in free_list if max(
                    diff([c[0] for c in i]), diff([c[1] for c in i])) > min_size]
            horizontal_list_agg.append(horizontal_list)
            free_list_agg.append(free_list)

        frame_ocr_area_ratios = []
        for horizontal_list, free_list in zip(horizontal_list_agg, free_list_agg):
            # Calculate text area from rectangles
            rect_area = 0
            for xmin, xmax, ymin, ymax in horizontal_list:
                if xmax < xmin or ymax < ymin:
                    continue
                rect_area += (xmax - xmin) * (ymax - ymin)
            
            # Calculate text area from free-form polygons
            quad_area = 0
            for points in free_list:
                triangle1 = points[:3]
                quad_area += triangle_area(*triangle1)
                triangle2 = points[2:] + [points[0]]
                quad_area += triangle_area(*triangle2)
            
            # Total text area and ratio
            text_area = rect_area + quad_area
            area_ratio = text_area / total_area
            frame_ocr_area_ratios.append(area_ratio)
        return np.mean(frame_ocr_area_ratios) if frame_ocr_area_ratios else 0.0

    def compute_stats_single(self, sample, rank=None, context=False):
        if StatsKeys.my_ocr_score in sample[Fields.stats]:
            return sample
        
        # there is no video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            sample[Fields.stats][StatsKeys.my_ocr_score] = np.array([], dtype=np.float64)
            return sample
        
         # load videos
        clip_path_list = sample[self.video_key]
        ocr_scores = []
        for clip_path in clip_path_list:
            frames = self.extract_frames(clip_path)

            assert frames is not None, f"Failed to extract frames from {clip_path}"
            processed_frames, ratio_h, ratio_w, total_area = self.preprocess(frames)
            x = torch.from_numpy(processed_frames).to(self.device)
            model = get_model(self.model_key, rank=rank, use_cuda=self.use_cuda())
            with torch.no_grad():
                with autocast():
                    y, _ = model(x)
                y = y.float().cpu().numpy()
            
            area_ratio = self.postprocess(y, ratio_h, ratio_w, total_area)
            ocr_scores.append(area_ratio)

        sample[Fields.stats][StatsKeys.my_ocr_score] = ocr_scores
        return sample
        
    def process_single(self, sample):
        ocr_scores = sample[Fields.stats][StatsKeys.my_ocr_score]
        if len(ocr_scores) <= 0:
            return True
        
        for score in ocr_scores:
            if score > self.threshold:
                return False
        return True