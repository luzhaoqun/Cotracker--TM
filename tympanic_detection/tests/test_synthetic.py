"""
Synthetic Data Tests

Validate algorithm correctness using synthetic data with known ground truth.
"""

import numpy as np
from typing import Tuple, Optional
import sys
from pathlib import Path

# Add project root to path for imports (go up 2 levels: tests -> tympanic_detection -> project_root)
_project_root = Path(__file__).parent.parent.parent.absolute()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def generate_synthetic_tracks(
    n_frames: int = 100,
    grid_shape: Tuple[int, int] = (20, 20),
    deformation_start: int = 30,
    deformation_end: int = 50,
    peak_start: int = 50,
    peak_end: int = 80,
    max_deformation: float = 8.0,
    camera_motion_amplitude: float = 5.0,
    tracking_noise: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic point tracks with known camera motion and deformation.
    
    Args:
        n_frames: Number of frames
        grid_shape: (rows, cols) of point grid
        deformation_start: Frame where deformation starts
        deformation_end: Frame where deformation reaches peak
        peak_start: Frame where peak state starts
        peak_end: Frame where peak state ends
        max_deformation: Maximum deformation amplitude in pixels
        camera_motion_amplitude: Amplitude of simulated camera motion
        tracking_noise: Standard deviation of tracking noise
        
    Returns:
        tracks: [T, N, 2] synthetic tracks
        visibility: [T, N] visibility (all 1.0 for synthetic)
        ground_truth_states: [T] ground truth state labels (0, 1, 2)
        ground_truth_deformation: [T, N, 2] pure deformation (no camera motion)
    """
    rows, cols = grid_shape
    n_points = rows * cols
    
    # Create base grid points
    x = np.linspace(100, 400, cols)
    y = np.linspace(100, 400, rows)
    xx, yy = np.meshgrid(x, y)
    base_points = np.stack([xx.ravel(), yy.ravel()], axis=1)  # [N, 2]
    
    # Center of the grid (for radial deformation)
    center = np.array([250, 250])
    
    # Initialize outputs
    tracks = np.zeros((n_frames, n_points, 2))
    ground_truth_states = np.zeros(n_frames, dtype=np.int32)
    ground_truth_deformation = np.zeros((n_frames, n_points, 2))
    
    for t in range(n_frames):
        # ===== Camera Motion (Global Transform) =====
        # Simulate smooth camera motion using sinusoidal functions
        tx = camera_motion_amplitude * np.sin(t * 0.1)
        ty = camera_motion_amplitude * np.cos(t * 0.08)
        
        # Slight zoom variation (simulates camera moving along optical axis)
        scale = 1.0 + 0.02 * np.sin(t * 0.05)
        
        # Apply camera motion to all points
        points_with_camera = (base_points - center) * scale + center
        points_with_camera = points_with_camera + np.array([tx, ty])
        
        # ===== Deformation =====
        deformation = np.zeros((n_points, 2))
        
        if t < deformation_start:
            # Static - no deformation
            ground_truth_states[t] = 0
            
        elif t < deformation_end:
            # Deforming - linearly increasing deformation
            progress = (t - deformation_start) / (deformation_end - deformation_start)
            deformation = _compute_radial_deformation(
                base_points, center, max_deformation * progress
            )
            ground_truth_states[t] = 1
            
        elif t < peak_end:
            # Peak - constant maximum deformation
            deformation = _compute_radial_deformation(
                base_points, center, max_deformation
            )
            ground_truth_states[t] = 2
            
        else:
            # Return to static or gradual return
            # For simplicity, assume it stays at peak
            deformation = _compute_radial_deformation(
                base_points, center, max_deformation
            )
            ground_truth_states[t] = 2
        
        # Store ground truth deformation
        ground_truth_deformation[t] = deformation
        
        # Apply deformation on top of camera motion
        final_points = points_with_camera + deformation
        
        # Add tracking noise
        noise = np.random.randn(n_points, 2) * tracking_noise
        final_points = final_points + noise
        
        tracks[t] = final_points
    
    # Visibility is all 1.0 for synthetic data
    visibility = np.ones((n_frames, n_points))
    
    return tracks, visibility, ground_truth_states, ground_truth_deformation


def _compute_radial_deformation(
    points: np.ndarray,
    center: np.ndarray,
    max_amplitude: float
) -> np.ndarray:
    """
    Compute radial "bulging" deformation.
    Center points move more, edge points move less.
    """
    # Distance from center
    dist_to_center = np.linalg.norm(points - center, axis=1)
    max_dist = dist_to_center.max()
    
    # Deformation amplitude decreases with distance from center
    # Gaussian-like falloff
    amplitude = max_amplitude * np.exp(-0.5 * (dist_to_center / (max_dist * 0.5)) ** 2)
    
    # Direction: radially outward from center
    direction = (points - center) / (dist_to_center[:, None] + 1e-6)
    
    deformation = amplitude[:, None] * direction
    return deformation


def test_homography_on_pure_camera_motion():
    """
    Test that homography perfectly explains pure camera motion.
    Residuals should be near zero.
    """
    print("=" * 60)
    print("Test 1: Pure Camera Motion (No Deformation)")
    print("=" * 60)
    
    # Generate tracks with camera motion but no deformation
    tracks, visibility, _, _ = generate_synthetic_tracks(
        n_frames=50,
        max_deformation=0.0,  # No deformation
        camera_motion_amplitude=10.0,
        tracking_noise=0.3
    )
    
    from tympanic_detection.homography_analysis import HomographyAnalyzer
    
    analyzer = HomographyAnalyzer(ransac_threshold=2.0)
    result = analyzer.analyze(tracks, None)
    
    # Check residuals
    residuals = [r.residual_magnitude.mean() for r in result.frame_results]
    mean_residual = np.mean(residuals)
    max_residual = np.max(residuals)
    
    print(f"Mean residual: {mean_residual:.3f} px")
    print(f"Max residual: {max_residual:.3f} px")
    
    # Should be very small (close to tracking noise level)
    passed = mean_residual < 1.0 and max_residual < 2.0
    print(f"PASSED: {passed}")
    print()
    
    return passed


def test_homography_detects_deformation():
    """
    Test that homography residuals increase when deformation is present.
    """
    print("=" * 60)
    print("Test 2: Camera Motion + Deformation")
    print("=" * 60)
    
    tracks, visibility, gt_states, _ = generate_synthetic_tracks(
        n_frames=100,
        deformation_start=30,
        deformation_end=50,
        peak_start=50,
        peak_end=80,
        max_deformation=8.0,
        camera_motion_amplitude=5.0,
        tracking_noise=0.5
    )
    
    from tympanic_detection.homography_analysis import HomographyAnalyzer
    
    analyzer = HomographyAnalyzer(ransac_threshold=3.0)
    result = analyzer.analyze(tracks, None)
    
    residuals = np.array([r.residual_magnitude.mean() for r in result.frame_results])
    
    # Check residual in different phases
    static_residual = residuals[:30].mean()
    deforming_residual = residuals[40:50].mean()
    peak_residual = residuals[55:75].mean()
    
    print(f"Static phase residual (frames 0-29): {static_residual:.3f} px")
    print(f"Deforming phase residual (frames 40-49): {deforming_residual:.3f} px")
    print(f"Peak phase residual (frames 55-74): {peak_residual:.3f} px")
    
    # Peak residual should be significantly higher than static
    passed = peak_residual > static_residual * 3
    print(f"Peak > 3x Static: {passed}")
    print()
    
    return passed


def test_full_pipeline():
    """
    Test the complete detection pipeline.
    """
    print("=" * 60)
    print("Test 3: Full Pipeline")
    print("=" * 60)
    
    from tympanic_detection.preprocessing import sample_grid_points_in_mask, build_grid_neighbors
    from tympanic_detection.quality_control import TrackQualityController
    from tympanic_detection.homography_analysis import HomographyAnalyzer
    from tympanic_detection.feature_extraction import FeatureExtractor
    from tympanic_detection.classification import AdaptiveThresholdClassifier
    from tympanic_detection.postprocessing import postprocess
    
    # Generate synthetic data
    grid_shape = (20, 20)
    tracks, visibility, gt_states, _ = generate_synthetic_tracks(
        n_frames=100,
        grid_shape=grid_shape,
        deformation_start=30,
        deformation_end=50,
        peak_start=50,
        peak_end=80,
        max_deformation=8.0,
        camera_motion_amplitude=5.0,
        tracking_noise=0.5
    )
    
    n_points = grid_shape[0] * grid_shape[1]
    valid_mask = np.ones(n_points, dtype=bool)
    neighbors = build_grid_neighbors(grid_shape, valid_mask)
    
    # Quality control
    qc = TrackQualityController(neighbor_indices=neighbors)
    qc_result = qc.process(tracks, visibility)
    
    # Homography analysis
    analyzer = HomographyAnalyzer(ransac_threshold=3.0)
    homography_result = analyzer.analyze(qc_result.repaired_tracks, qc_result.weights)
    
    # Feature extraction
    extractor = FeatureExtractor(grid_shape=grid_shape, valid_mask=valid_mask)
    features = extractor.extract_all_features(homography_result)
    
    # Classification
    classifier = AdaptiveThresholdClassifier()
    classification_result = classifier.classify(features)
    
    # Post-processing
    final_result = postprocess(classification_result)
    
    # Evaluate
    detected_states = final_result.smoothed_states
    
    # Frame-level accuracy
    accuracy = (detected_states == gt_states).mean()
    
    # State-wise recall
    for state in [0, 1, 2]:
        mask = gt_states == state
        if mask.sum() > 0:
            recall = (detected_states[mask] == state).mean()
            print(f"State {state} recall: {recall:.2%}")
    
    print(f"Overall accuracy: {accuracy:.2%}")
    
    # Check if key events are detected within tolerance
    tolerance = 5  # frames
    
    def check_event(detected, expected, name):
        if detected is None:
            print(f"{name}: Not detected (expected {expected})")
            return False
        error = abs(detected - expected)
        ok = error <= tolerance
        print(f"{name}: Detected {detected}, Expected {expected}, Error {error} frames - {'OK' if ok else 'FAIL'}")
        return ok
    
    print()
    deform_ok = check_event(final_result.deformation_start, 30, "Deformation start")
    peak_ok = check_event(final_result.peak_start, 50, "Peak start")
    
    passed = accuracy > 0.7 and deform_ok and peak_ok
    print(f"\nPASSED: {passed}")
    print()
    
    return passed


def run_all_tests():
    """Run all synthetic tests."""
    print("\n" + "=" * 60)
    print("RUNNING SYNTHETIC DATA TESTS")
    print("=" * 60 + "\n")
    
    results = []
    
    results.append(("Pure Camera Motion", test_homography_on_pure_camera_motion()))
    results.append(("Camera + Deformation", test_homography_detects_deformation()))
    results.append(("Full Pipeline", test_full_pipeline()))
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name}: {status}")
        all_passed = all_passed and passed
    
    print()
    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests failed.")
    
    return all_passed


if __name__ == "__main__":
    run_all_tests()
