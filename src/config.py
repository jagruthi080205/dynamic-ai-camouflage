import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLES_DIR = DATA_DIR / "samples"
OUTPUTS_DIR = DATA_DIR / "outputs"

# Create directories
for d in [DATA_DIR, SAMPLES_DIR, OUTPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Segmentation Modes
SEGMENTATION_MODES = [
    "Interactive Bounding Box / Point Eraser",
    "Real-Time Person & Selfie Removal",
    "Dynamic Color-Keyed Invisibility",
    "Motion-Salient Foreground Eraser"
]

# Inpainting Engines
INPAINTING_ENGINES = {
    "neural_fast": "Fast Deep Inpainting (Texture & Context Aware)",
    "navier_stokes": "Navier-Stokes Fluid Dynamics Inpainting",
    "telea_fast": "Fast Marching Telea Edge Diffusion"
}

# Optical Flow Defaults (Farneback Algorithm for Camera Motion Compensation)
OPTICAL_FLOW_CONFIG = {
    "pyr_scale": 0.5,
    "levels": 3,
    "winsize": 15,
    "iterations": 3,
    "poly_n": 5,
    "poly_sigma": 1.2,
    "flags": 0
}

# Real-time Video Stream Defaults
DEFAULT_FRAME_WIDTH = 640
DEFAULT_FRAME_HEIGHT = 480
TARGET_FPS = 30
INPAINT_DILATION_KERNEL_SIZE = 7
