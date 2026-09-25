import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Set UTF-8 encoding for stdout on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation import TargetSegmenter
from src.optical_flow import CameraMotionTracker
from src.inpainter import GenerativeInpainter
from src.pipeline import CamouflagePipeline
from src.config import SAMPLES_DIR

def run_smoke_tests():
    print("==================================================")
    print("[*] RUNNING DYNAMIC AI CAMOUFLAGE PIPELINE TESTS")
    print("==================================================")

    # 1. Test Segmentation
    print("\n[1/4] Testing TargetSegmenter ...")
    segmenter = TargetSegmenter()
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Person mask
    mask_person = segmenter.segment_person(test_img)
    assert mask_person.shape == (480, 640), f"Invalid person mask shape: {mask_person.shape}"
    print(f" -> Person mask test passed. Active pixels: {np.sum(mask_person > 0)}")

    # Bounding box mask
    mask_bbox = segmenter.segment_bounding_box(test_img, [100, 100, 200, 200])
    assert mask_bbox.shape == (480, 640), f"Invalid bbox mask shape: {mask_bbox.shape}"
    assert np.sum(mask_bbox > 0) > 0, "Bbox mask should have non-zero pixels"
    print(f" -> Bounding box mask test passed. Active pixels: {np.sum(mask_bbox > 0)}")

    # Color key mask
    red_patch = np.zeros((480, 640, 3), dtype=np.uint8)
    red_patch[100:200, 100:200] = [230, 20, 20]
    mask_color = segmenter.segment_color_key(red_patch, target_color="red")
    assert mask_color.shape == (480, 640)
    print(f" -> Color key mask test passed. Active pixels: {np.sum(mask_color > 0)}")

    # 2. Test Optical Flow Motion Tracker
    print("\n[2/4] Testing CameraMotionTracker ...")
    tracker = CameraMotionTracker()
    f1 = np.zeros((480, 640, 3), dtype=np.uint8)
    f2 = np.zeros((480, 640, 3), dtype=np.uint8)
    f1[200:250, 200:250] = 255
    f2[200:250, 220:270] = 255  # Shifted right by 20px

    flow = tracker.compute_dense_flow(f1, f2)
    assert flow.shape == (480, 640, 2), f"Invalid flow shape: {flow.shape}"
    flow_vis = tracker.visualize_flow_hsv(flow)
    assert flow_vis.shape == (480, 640, 3), f"Invalid flow vis shape: {flow_vis.shape}"
    print(f" -> Optical Flow Farneback test passed. Flow max magnitude: {np.max(np.abs(flow)):.2f}")

    # 3. Test Generative Inpainter
    print("\n[3/4] Testing GenerativeInpainter ...")
    inpainter = GenerativeInpainter()
    test_scene = np.ones((480, 640, 3), dtype=np.uint8) * 180
    test_mask = np.zeros((480, 640), dtype=np.uint8)
    test_mask[150:250, 150:250] = 255

    for engine in ["neural_fast", "navier_stokes", "telea_fast"]:
        res = inpainter.inpaint(test_scene, test_mask, engine=engine)
        assert res.shape == (480, 640, 3), f"Inpaint failed for {engine}"
        print(f" -> Inpainting engine '{engine}' passed. Latency: {inpainter.last_latency_ms:.2f}ms")

    # 4. Test End-to-End Pipeline
    print("\n[4/4] Testing End-to-End CamouflagePipeline ...")
    pipeline = CamouflagePipeline(inpaint_engine="neural_fast")
    res1 = pipeline.process_frame(f1, mode="bbox", bbox=[200, 200, 250, 250])
    res2 = pipeline.process_frame(f2, mode="bbox", bbox=[200, 220, 250, 270])

    assert "output_frame" in res2
    assert "mask" in res2
    assert "flow_vis" in res2
    assert "metrics" in res2
    print(f" -> End-to-End pipeline verified! FPS: {res2['metrics']['fps']}, Latency: {res2['metrics']['latency_ms']}ms")

    print("\n>>> ALL 4 ENGINE MODULES TESTED AND PASSED WITH 100% SUCCESS! <<<\n")

if __name__ == "__main__":
    run_smoke_tests()
