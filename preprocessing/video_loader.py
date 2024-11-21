# preprocessing/video_loader.py
import cv2
import numpy as np
from typing import List, Tuple
from config.config import Config

class VideoLoader:
    @staticmethod
    def load_video(video_path: str, max_frames: int = 300) -> np.ndarray:
        """
        Load video frames and preprocess them.
        Args:
            video_path (str): Path to the input video
            max_frames (int): Maximum number of frames to load
        Returns:
            np.ndarray: Processed video frames
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            # Resize and normalize
            processed_frame = cv2.resize(frame, (Config.INPUT_SHAPE[0], Config.INPUT_SHAPE[1]))
            processed_frame = processed_frame / 255.0  # Normalize to [0,1]
            frames.append(processed_frame)
        cap.release()
        return np.array(frames)
    @staticmethod
    def extract_frames(video_path: str, output_dir: str, interval: int = 10) -> List[str]:
        """
        Extract frames from video at specified intervals.
        Args:
            video_path (str): Path to input video
            output_dir (str): Directory to save frames
            interval (int): Frame extraction interval
        Returns:
            List[str]: Paths of extracted frames
        """
        cap = cv2.VideoCapture(video_path)
        frame_paths = []
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % interval == 0:
                frame_filename = f'{output_dir}/frame_{frame_count:04d}.jpg'
                cv2.imwrite(frame_filename, frame)
                frame_paths.append(frame_filename)
            frame_count += 1
        cap.release()
        return frame_paths