import os
import sys
import time
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List, Generator
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from src.segmentation import TargetSegmenter
from src.optical_flow import CameraMotionTracker
from src.inpainter import GenerativeInpainter
from src.config import OUTPUTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class CamouflagePipeline:
    """
    End-to-End Real-Time Dynamic AI Camouflage Stream Orchestrator.
    Connects segmentation, dense optical flow motion tracking, and generative inpainting.
    Supports:
    - Moving camera background warping
    - Live webcam streaming (30 FPS)
    - Full video file batch processing with progress feedback
    """

    def __init__(self, inpaint_engine: str = "neural_fast"):
        self.segmenter = TargetSegmenter()
        self.tracker = CameraMotionTracker()
        self.inpainter = GenerativeInpainter()
        self.inpaint_engine = inpaint_engine

        # Temporal state
        self.prev_frame_rgb = None
        self.clean_background_rgb = None
        self.accumulated_homography = np.eye(3, dtype=np.float32)
        
        # Performance Tracking
        self.fps = 30.0
        self.frame_count = 0
        self.total_latency_ms = 0.0

    def capture_clean_background(self, frame_rgb: np.ndarray):
        """Stores a clean initial reference frame for moving-camera temporal background reconstruction."""
        self.clean_background_rgb = frame_rgb.copy()
        self.accumulated_homography = np.eye(3, dtype=np.float32)
        logger.info("Reference clean background buffer captured.")

    def reset(self):
        """Resets temporal buffers and tracking matrices."""
        self.prev_frame_rgb = None
        self.clean_background_rgb = None
        self.accumulated_homography = np.eye(3, dtype=np.float32)
        self.tracker.reset_buffer()
        self.frame_count = 0

    def process_frame(
        self,
        frame_rgb: np.ndarray,
        mode: str = "person",
        bbox: Optional[List[int]] = None,
        color_key: str = "red",
        motion_compensation: bool = True,
        custom_mask: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Processes a single video frame through the AI Camouflage pipeline.

        Args:
            frame_rgb: Current input frame (H, W, 3) uint8
            mode: 'person' | 'bbox' | 'color' | 'custom'
            bbox: [xmin, ymin, xmax, ymax] if mode == 'bbox'
            color_key: 'red' | 'blue' | 'green' if mode == 'color'
            motion_compensation: Whether to track moving camera with optical flow
            custom_mask: User-provided binary mask (optional)

        Returns:
            Dict containing:
                - output_frame: Inpainted result (H, W, 3)
                - mask: Binary target mask (H, W)
                - flow_vis: Optical flow HSV visual (H, W, 3)
                - metrics: Dictionary of latency, FPS, and camera velocity
        """
        start_time = time.perf_counter()
        h, w = frame_rgb.shape[:2]

        # 1. Target Segmentation
        if custom_mask is not None:
            mask = custom_mask
        elif mode == "person":
            mask = self.segmenter.segment_person(frame_rgb)
        elif mode == "bbox" and bbox is not None:
            mask = self.segmenter.segment_bounding_box(frame_rgb, bbox)
        elif mode == "color":
            mask = self.segmenter.segment_color_key(frame_rgb, color_key)
        else:
            mask = np.zeros((h, w), dtype=np.uint8)

        # 2. Camera Motion Tracking & Optical Flow
        flow_vis = np.zeros_like(frame_rgb)
        camera_velocity = 0.0
        warped_bg = None

        if self.prev_frame_rgb is not None:
            # Calculate dense flow for visualization
            dense_flow = self.tracker.compute_dense_flow(self.prev_frame_rgb, frame_rgb)
            flow_vis = self.tracker.visualize_flow_hsv(dense_flow)

            if motion_compensation:
                # Estimate camera homography excluding foreground target
                step_H, camera_velocity = self.tracker.estimate_camera_motion(
                    self.prev_frame_rgb,
                    frame_rgb,
                    exclude_mask=mask
                )
                
                # Chain transformations forward
                self.accumulated_homography = step_H @ self.accumulated_homography

                # If reference background exists, warp it to current camera coordinate
                if self.clean_background_rgb is not None:
                    warped_bg = self.tracker.warp_background_to_current_view(
                        self.clean_background_rgb,
                        self.accumulated_homography,
                        (h, w)
                    )
        else:
            # First frame initializations
            if self.clean_background_rgb is None:
                self.clean_background_rgb = frame_rgb.copy()

        # 3. Generative Inpainting
        engine_to_use = "temporal_fusion" if (warped_bg is not None and motion_compensation) else self.inpaint_engine
        output_frame = self.inpainter.inpaint(
            frame_rgb,
            mask,
            engine=engine_to_use,
            warped_bg=warped_bg
        )

        # Update historical state
        self.prev_frame_rgb = frame_rgb.copy()
        self.frame_count += 1

        # Compute Latency & FPS
        duration = time.perf_counter() - start_time
        latency_ms = duration * 1000.0
        self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(duration, 0.001))

        return {
            "output_frame": output_frame,
            "mask": mask,
            "flow_vis": flow_vis,
            "metrics": {
                "fps": round(self.fps, 1),
                "latency_ms": round(latency_ms, 2),
                "camera_velocity_px": round(camera_velocity, 2),
                "erased_pixels": int(np.sum(mask > 0)),
                "inpaint_engine": engine_to_use
            }
        }

    def process_video_file(
        self,
        input_video_path: str,
        output_video_path: Optional[str] = None,
        mode: str = "person",
        bbox: Optional[List[int]] = None,
        color_key: str = "red",
        motion_compensation: bool = True,
        progress_callback: Optional[callable] = None
    ) -> str:
        """
        Processes an entire moving-camera video file and exports the camouflaged result to MP4.
        """
        if not HAS_CV2:
            raise RuntimeError("OpenCV is required for video file processing.")

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {input_video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

        if output_video_path is None:
            output_video_path = str(OUTPUTS_DIR / f"camouflaged_{int(time.time())}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        self.reset()
        frame_idx = 0

        try:
            while cap.isOpened():
                ret, frame_bgr = cap.read()
                if not ret or frame_bgr is None:
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                res = self.process_frame(
                    frame_rgb,
                    mode=mode,
                    bbox=bbox,
                    color_key=color_key,
                    motion_compensation=motion_compensation
                )

                out_bgr = cv2.cvtColor(res["output_frame"], cv2.COLOR_RGB2BGR)
                writer.write(out_bgr)

                frame_idx += 1
                if progress_callback:
                    progress_callback(frame_idx, total_frames, res["metrics"])

        finally:
            cap.release()
            writer.release()

        logger.info(f"Video processing complete! Saved to {output_video_path}")
        return output_video_path

if __name__ == "__main__":
    pipeline = CamouflagePipeline()
    f1 = np.ones((480, 640, 3), dtype=np.uint8) * 200
    res = pipeline.process_frame(f1, mode="person")
    print(f"Pipeline Verified! Output shape: {res['output_frame'].shape}, FPS: {res['metrics']['fps']}, Latency: {res['metrics']['latency_ms']}ms")
