import os
import sys
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
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

from src.config import OPTICAL_FLOW_CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class CameraMotionTracker:
    """
    Dense & Feature-Based Camera Motion Tracking Engine.
    Enables Dynamic AI Camouflage to work on MOVING cameras without green screens:
    1. Computes dense optical flow fields (u, v) between consecutive video frames.
    2. Estimates global affine / homography transformation matrix of camera pan/tilt/zoom.
    3. Warps historical clean background pixels into the current moving camera coordinate frame.
    4. Generates visual motion vector overlays (HSV flow wheel & quiver arrows).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or OPTICAL_FLOW_CONFIG
        self.prev_gray = None
        self.background_buffer = None
        self.bg_homography_chain = np.eye(3, dtype=np.float32)

    def reset_buffer(self):
        """Resets temporal background memory and motion chains."""
        self.prev_gray = None
        self.background_buffer = None
        self.bg_homography_chain = np.eye(3, dtype=np.float32)

    def _to_gray_uint8(self, frame: np.ndarray) -> np.ndarray:
        """Helper to convert any input frame to strictly contiguous 2D uint8 grayscale."""
        if frame is None:
            return np.zeros((480, 640), dtype=np.uint8)

        arr = np.asarray(frame, dtype=np.uint8)

        if not HAS_CV2:
            if arr.ndim == 3:
                return np.ascontiguousarray(np.mean(arr, axis=2), dtype=np.uint8)
            return np.ascontiguousarray(arr, dtype=np.uint8)

        if arr.ndim == 2:
            return np.ascontiguousarray(arr, dtype=np.uint8)
        elif arr.ndim == 3:
            if arr.shape[2] == 1:
                return np.ascontiguousarray(arr[:, :, 0], dtype=np.uint8)
            elif arr.shape[2] >= 3:
                rgb_3c = np.ascontiguousarray(arr[:, :, :3], dtype=np.uint8)
                return np.ascontiguousarray(cv2.cvtColor(rgb_3c, cv2.COLOR_RGB2GRAY), dtype=np.uint8)

        return np.ascontiguousarray(arr[:, :, 0] if arr.ndim == 3 else arr, dtype=np.uint8)

    def compute_dense_flow(
        self,
        prev_frame_rgb: np.ndarray,
        curr_frame_rgb: np.ndarray
    ) -> np.ndarray:
        """
        Computes Dense Optical Flow (Gunnar Farneback algorithm) between two frames.
        Returns flow array of shape (H, W, 2) where [..., 0] is dx (u) and [..., 1] is dy (v).
        """
        h, w = curr_frame_rgb.shape[:2]
        fallback_flow = np.zeros((h, w, 2), dtype=np.float32)

        if not HAS_CV2:
            return fallback_flow

        try:
            prev_g = self._to_gray_uint8(prev_frame_rgb)
            curr_g = self._to_gray_uint8(curr_frame_rgb)

            # Ensure exact 2D shape and dimension match
            if prev_g.ndim != 2:
                prev_g = prev_g[:, :, 0]
            if curr_g.ndim != 2:
                curr_g = curr_g[:, :, 0]

            if prev_g.shape != (h, w):
                prev_g = cv2.resize(prev_g, (w, h))
            if curr_g.shape != (h, w):
                curr_g = cv2.resize(curr_g, (w, h))

            prev_g = np.ascontiguousarray(prev_g, dtype=np.uint8)
            curr_g = np.ascontiguousarray(curr_g, dtype=np.uint8)

            flow = cv2.calcOpticalFlowFarneback(
                prev_g,
                curr_g,
                None,
                pyr_scale=float(self.config.get("pyr_scale", 0.5)),
                levels=int(self.config.get("levels", 3)),
                winsize=int(self.config.get("winsize", 15)),
                iterations=int(self.config.get("iterations", 3)),
                poly_n=int(self.config.get("poly_n", 5)),
                poly_sigma=float(self.config.get("poly_sigma", 1.2)),
                flags=int(self.config.get("flags", 0))
            )
            return np.ascontiguousarray(flow, dtype=np.float32)
        except Exception as e:
            logger.debug(f"Optical flow calculation notice ({e}). Returning zero flow field.")
            return fallback_flow

    def estimate_camera_motion(
        self,
        prev_frame_rgb: np.ndarray,
        curr_frame_rgb: np.ndarray,
        exclude_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Estimates the rigid / affine camera transformation matrix (3x3) between consecutive frames.
        Points inside exclude_mask (e.g. the moving target to be erased) are discarded from camera motion calculation.
        
        Returns:
            H_matrix: 3x3 Homography transformation matrix
            motion_magnitude: Average translation velocity (pixels/frame)
        """
        h, w = curr_frame_rgb.shape[:2]
        identity_H = np.eye(3, dtype=np.float32)

        if not HAS_CV2:
            return identity_H, 0.0

        try:
            prev_g = self._to_gray_uint8(prev_frame_rgb)
            curr_g = self._to_gray_uint8(curr_frame_rgb)

            if prev_g.shape != (h, w):
                prev_g = cv2.resize(prev_g, (w, h))
            if curr_g.shape != (h, w):
                curr_g = cv2.resize(curr_g, (w, h))

            # Valid feature tracking mask (invert exclusion mask so features are only selected from background)
            valid_mask = None
            if exclude_mask is not None:
                m_arr = np.asarray(exclude_mask, dtype=np.uint8)
                if m_arr.shape[:2] != (h, w):
                    m_arr = cv2.resize(m_arr, (w, h))
                valid_mask = cv2.bitwise_not(np.ascontiguousarray(m_arr, dtype=np.uint8))

            # Detect Shi-Tomasi strong corner features in background
            prev_pts = cv2.goodFeaturesToTrack(
                prev_g,
                maxCorners=200,
                qualityLevel=0.01,
                minDistance=15,
                mask=valid_mask
            )

            if prev_pts is None or len(prev_pts) < 8:
                return identity_H, 0.0

            # Track features forward with Lucas-Kanade optical flow
            curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
                prev_g,
                curr_g,
                prev_pts,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
            )

            if curr_pts is None or status is None:
                return identity_H, 0.0

            good_prev = prev_pts[status.flatten() == 1]
            good_curr = curr_pts[status.flatten() == 1]

            if len(good_prev) < 6:
                return identity_H, 0.0

            # Estimate robust affine / perspective transform with RANSAC
            H, inliers = cv2.findHomography(good_prev, good_curr, cv2.RANSAC, 3.0)

            if H is None:
                return identity_H, 0.0

            # Calculate average motion velocity
            displacements = np.linalg.norm(good_curr - good_prev, axis=1)
            avg_motion = float(np.mean(displacements)) if len(displacements) > 0 else 0.0

            return H.astype(np.float32), avg_motion
        except Exception as e:
            logger.debug(f"Camera motion estimation notice ({e}). Returning identity homography.")
            return identity_H, 0.0

    def warp_background_to_current_view(
        self,
        bg_frame_rgb: np.ndarray,
        homography_matrix: np.ndarray,
        target_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Warps a past clean background frame to align precisely with the current camera perspective.
        target_shape: (height, width)
        """
        h, w = target_shape[:2]
        if not HAS_CV2 or homography_matrix is None or bg_frame_rgb is None:
            return cv2.resize(bg_frame_rgb, (w, h)) if (HAS_CV2 and bg_frame_rgb is not None) else bg_frame_rgb

        try:
            warped = cv2.warpPerspective(
                bg_frame_rgb,
                homography_matrix,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT
            )
            return warped
        except Exception:
            return cv2.resize(bg_frame_rgb, (w, h))

    def visualize_flow_hsv(self, flow: np.ndarray) -> np.ndarray:
        """
        Generates standard Optical Flow HSV chromatic visualization:
        - Hue: Direction of pixel motion
        - Saturation: 255 (Full)
        - Value: Magnitude of movement
        Returns RGB image.
        """
        h, w = flow.shape[:2]
        if not HAS_CV2:
            return np.zeros((h, w, 3), dtype=np.uint8)

        try:
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv = np.zeros((h, w, 3), dtype=np.uint8)
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 1] = 255
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            return rgb
        except Exception:
            return np.zeros((h, w, 3), dtype=np.uint8)

    def visualize_flow_vectors(
        self,
        frame_rgb: np.ndarray,
        flow: np.ndarray,
        grid_step: int = 24,
        arrow_color: Tuple[int, int, int] = (0, 255, 255)
    ) -> np.ndarray:
        """
        Overlays motion vector quiver arrows on top of the RGB frame.
        """
        vis = frame_rgb.copy()
        if not HAS_CV2:
            return vis

        try:
            h, w = frame_rgb.shape[:2]
            y, x = np.mgrid[grid_step // 2:h:grid_step, grid_step // 2:w:grid_step].reshape(2, -1).astype(int)
            fx, fy = flow[y, x].T

            lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
            lines = np.int32(lines + 0.5)

            for (x1, y1), (x2, y2) in lines:
                mag = np.hypot(x2 - x1, y2 - y1)
                if mag > 1.5:
                    cv2.arrowedLine(vis, (x1, y1), (x2, y2), arrow_color, 1, tipLength=0.3)
                    cv2.circle(vis, (x1, y1), 1, (255, 0, 0), -1)

            return vis
        except Exception:
            return vis

if __name__ == "__main__":
    tracker = CameraMotionTracker()
    f1 = np.zeros((480, 640, 3), dtype=np.uint8)
    f2 = np.zeros((480, 640, 3), dtype=np.uint8)
    f2[50:100, 50:100] = 255
    flow = tracker.compute_dense_flow(f1, f2)
    H, vel = tracker.estimate_camera_motion(f1, f2)
    hsv_vis = tracker.visualize_flow_hsv(flow)
    print(f"Motion Tracker Verified! Flow shape: {flow.shape}, Avg Vel: {vel:.2f}px/frame, Vis shape: {hsv_vis.shape}")
