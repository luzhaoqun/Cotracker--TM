"""
Example: Tympanic Membrane Deformation Detection

This script demonstrates how to use the detection pipeline.
"""

import sys
from pathlib import Path

# Add parent directory to path if package not installed
_current_dir = Path(__file__).parent.absolute()
_parent_dir = _current_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from tympanic_detection import (
    TympanicDeformationDetector,
    plot_temporal_profile,
    plot_residual_field,
    plot_tracking_quality
)


def run_example(frames_folder: str, masks_folder: str, output_dir: str = None):
    """
    Run detection on a video.
    
    Args:
        frames_folder: Path to folder containing video frames (e.g., PNG images)
        masks_folder: Path to folder containing binary mask images
        output_dir: Optional output directory for visualizations
    """
    # Create detector
    detector = TympanicDeformationDetector(
        grid_size=20,           # 20x20 grid of tracking points
        device="cuda",          # Use GPU (or "cpu")
        verbose=True            # Print progress
    )
    
    # Run detection
    print("\n" + "=" * 60)
    print("Starting Tympanic Membrane Deformation Detection")
    print("=" * 60 + "\n")
    
    result = detector.detect(
        frames_folder=frames_folder,
        masks_folder=masks_folder,
        frame_pattern="*.png",   # Adjust pattern if needed
        mask_pattern="*.png"
    )
    
    # Print summary
    print("\n" + result.summary())
    
    # Visualize results
    print("\nGenerating visualizations...")
    
    # 1. Temporal profile
    plot_temporal_profile(result)
    
    # 2. Residual field at key frames
    if result.final_result.deformation_start is not None:
        print(f"\nResidual field at deformation start (frame {result.final_result.deformation_start}):")
        plot_residual_field(result, result.final_result.deformation_start)
    
    if result.final_result.peak_start is not None:
        print(f"\nResidual field at peak (frame {result.final_result.peak_start}):")
        plot_residual_field(result, result.final_result.peak_start)
    
    # 3. Tracking quality
    plot_tracking_quality(result)
    
    return result


def run_synthetic_demo():
    """
    Run detection on synthetic data (no real video needed).
    """
    import numpy as np
    from tympanic_detection.tests.test_synthetic import generate_synthetic_tracks
    from tympanic_detection.preprocessing import build_grid_neighbors, PreprocessingResult
    from tympanic_detection.tracking import TrackingResult
    from tympanic_detection import (
        TrackQualityController,
        HomographyAnalyzer,
        FeatureExtractor,
        AdaptiveThresholdClassifier,
        postprocess
    )
    
    print("\n" + "=" * 60)
    print("Synthetic Data Demo")
    print("=" * 60 + "\n")
    
    # Generate synthetic data
    print("Generating synthetic tracks...")
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
    print("Running quality control...")
    qc = TrackQualityController(neighbor_indices=neighbors)
    qc_result = qc.process(tracks, visibility)
    
    # Homography analysis
    print("Running homography analysis...")
    analyzer = HomographyAnalyzer(ransac_threshold=3.0)
    homography_result = analyzer.analyze(qc_result.repaired_tracks, qc_result.weights)
    
    # Feature extraction
    print("Extracting features...")
    extractor = FeatureExtractor(grid_shape=grid_shape, valid_mask=valid_mask)
    features = extractor.extract_all_features(homography_result)
    
    # Classification
    print("Classifying states...")
    classifier = AdaptiveThresholdClassifier()
    classification_result = classifier.classify(features)
    
    # Post-processing
    print("Post-processing...")
    final_result = postprocess(classification_result)
    
    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\nGround Truth Events:")
    print(f"  - Deformation starts: frame 30")
    print(f"  - Peak reached: frame 50")
    print(f"  - Peak ends: frame 80")
    
    print(f"\nDetected Events:")
    print(f"  - Deformation starts: frame {final_result.deformation_start}")
    print(f"  - Peak reached: frame {final_result.peak_start}")
    print(f"  - Peak ends: frame {final_result.peak_end}")
    
    # Accuracy
    accuracy = (final_result.smoothed_states == gt_states).mean()
    print(f"\nOverall accuracy: {accuracy:.2%}")
    print(f"Detection confidence: {final_result.confidence:.2%}")
    
    # Plot temporal comparison
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
    frames = np.arange(len(gt_states))
    
    ax = axes[0]
    ax.step(frames, gt_states, where='mid', label='Ground Truth', linewidth=2)
    ax.set_ylabel('State')
    ax.set_title('Ground Truth States')
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['Static', 'Deforming', 'Peak'])
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.step(frames, final_result.smoothed_states, where='mid', 
            label='Detected', linewidth=2, color='orange')
    ax.set_ylabel('State')
    ax.set_xlabel('Frame')
    ax.set_title(f'Detected States (Accuracy: {accuracy:.1%})')
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['Static', 'Deforming', 'Peak'])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Tympanic membrane deformation detection example")
    parser.add_argument("--frames", type=str, help="Path to frames folder")
    parser.add_argument("--masks", type=str, help="Path to masks folder")
    parser.add_argument("--synthetic", action="store_true", help="Run synthetic data demo")
    args = parser.parse_args()
    
    if args.synthetic:
        run_synthetic_demo()
    elif args.frames and args.masks:
        run_example(args.frames, args.masks)
    else:
        print("Usage:")
        print("  With real data:    python example.py --frames /path/to/frames --masks /path/to/masks")
        print("  With synthetic:    python example.py --synthetic")
        print()
        print("Running synthetic demo...")
        run_synthetic_demo()
