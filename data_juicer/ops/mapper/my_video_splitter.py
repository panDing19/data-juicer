import os
import subprocess

from loguru import logger
from typing import List
from imageio_ffmpeg import get_ffmpeg_exe
from scenedetect import FrameTimecode, AdaptiveDetector, detect

from ..base_op import OPERATORS, Mapper

OP_NAME = "my_video_splitter"

@OPERATORS.register_module(OP_NAME)
class MyVideoSplitter(Mapper):
    _batched_op = True

    def __init__(
        self,
        output_dir: str,
        min_seconds: float = 2.0,
        max_seconds: float = 15.0,
        target_fps: int = 30,
        shorter_size: int = 1080,
        adaptive_threshold: float = 3.0,
        verbose: bool = False,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.output_dir = output_dir
        self.min_seconds = min_seconds
        self.max_seconds = max_seconds
        self.target_fps = target_fps
        self.shorter_size = shorter_size
        self.adaptive_threshold = adaptive_threshold
        self.verbose = verbose
        os.makedirs(output_dir, exist_ok=True)
    
    def detect_scenes(self, video_path, adaptive_threshold=3.0):
        """
        Detect scene changes in a video file.

        Args:
            video_path (str): Path to the video file
            adaptive_threshold (float): Threshold for scene detection sensitivity
            logger: Logger instance for logging messages

        Returns:
            tuple: (success (bool), scene_list (list of tuples))
        """
        detector = AdaptiveDetector(
            adaptive_threshold=adaptive_threshold,
        )

        try:
            scene_list = detect(video_path, detector, start_in_scene=True)
            return True, scene_list
        except Exception as e:
            print(f"Video '{video_path}' with error {e}")
            return False, []
    
    def split_video(
        self,
        video_path,
        output_dir,
        min_seconds=2.0,
        max_seconds=15.0,
        target_fps=30,
        shorter_size=1080,
        adaptive_threshold=3.0,
        verbose=False,
    ):
        """
        Split a video into scenes and save each scene as a separate clip.

        Args:
            video_path (str): Path to the input video file
            output_dir (str): Directory to save the output clips
            min_seconds (float): Minimum scene duration in seconds, shorter scenes will be ignored
            max_seconds (float): Maximum scene duration in seconds, longer scenes will be truncated
            target_fps (int): Target frame rate for the output videos
            shorter_size (int): Resize the shorter side to this value; will not do upscale
            adaptive_threshold (float): Threshold for scene detection sensitivity
            verbose (bool): Whether to print verbose output
            logger: Logger instance for logging messages

        Returns:
            list: List of paths to the saved video clips
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Detect scenes
        success, scene_list = self.detect_scenes(video_path, adaptive_threshold=adaptive_threshold)

        if not success:
            print(f"Failed to detect scenes for {video_path}")
            return []

        # Get ffmpeg executable
        FFMPEG_PATH = get_ffmpeg_exe()

        save_path_list = []
        for idx, scene in enumerate(scene_list):
            if scene is not None:
                s, t = scene  # FrameTimecode
                if min_seconds is not None:     
                    if (t - s).get_seconds() < min_seconds:
                        continue

                duration = t - s
                if max_seconds is not None:
                    fps = s.framerate
                    max_duration = FrameTimecode(timecode="00:00:00", fps=fps)
                    max_duration.frame_num = round(fps * max_seconds)
                    duration = min(max_duration, duration)

            # save path
            fname = os.path.basename(video_path)
            fname_wo_ext = os.path.splitext(fname)[0]
            save_path = os.path.join(output_dir, f"{fname_wo_ext}_scene-{idx}.mp4")

            # ffmpeg cmd
            cmd = [FFMPEG_PATH]

            # clip to cut
            # Note: -ss after -i is very slow; put -ss before -i !!!
            if scene is None:
                cmd += ["-nostdin", "-y", "-i", video_path]
            else:
                cmd += ["-nostdin", "-y", "-ss", str(s.get_seconds()), "-i", video_path, "-t", str(duration.get_seconds())]

            # target fps
            if target_fps is not None:
                cmd += ["-r", f"{target_fps}"]

            # aspect ratio
            if shorter_size is not None:
                cmd += ["-vf", f"scale='if(gt(iw,ih),-2,{shorter_size})':'if(gt(iw,ih),{shorter_size},-2)'"]

            cmd += ["-map", "0:v", save_path]

            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            proc.communicate()

            save_path_list.append(save_path)
            if verbose:
                print(f"Video clip saved to '{save_path}'")

        return save_path_list
    
    def _process_single_video(self, video_path: str) -> List[str]:
        """
        Process a single video, splitting it into scenes.

        Args:
            video_path (str): Path to the input video file

        Returns:
            List[str]: List of paths to the saved video clips
        """
        # Use default output directory
        output_dir = self.output_dir

        # Process the video
        result = self.split_video(
            video_path=video_path,
            output_dir=output_dir,
            min_seconds=self.min_seconds,
            max_seconds=self.max_seconds,
            target_fps=self.target_fps,
            shorter_size=self.shorter_size,
            adaptive_threshold=self.adaptive_threshold,
            verbose=self.verbose,
        )

        return result

    def process_batched(self, samples):
        res_samples = {}
        res_samples[self.video_key] = []
        res_samples[self.text_key] = []
        logger.warning(f"Processing batch of {len(samples[self.video_key])} samples in {self._name} ...")
        for video_paths in samples[self.video_key]:
            for video_path in video_paths:
                clip_paths = self._process_single_video(video_path)
                logger.warning(f"Split video {video_path} into {len(clip_paths)} clips.")
                for clip_path in clip_paths:
                    res_samples[self.video_key].append([clip_path])
                    res_samples[self.text_key].append('just for fake')
        for key, value in samples.items():
            if key not in res_samples:
                res_samples[key] = value
        return res_samples

        
