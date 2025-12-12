"""
Real Data Test Script

Test the tympanic membrane deformation detection on real endoscopic videos.
"""

import sys
from pathlib import Path
import argparse
import numpy as np

# Add project root to path
_project_root = Path(__file__).parent.parent.absolute()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from tympanic_detection import (
    TympanicDeformationDetector,
    plot_temporal_profile,
    plot_residual_field,
    plot_tracking_quality,
    create_summary_video
)


def run_detection(
    frames_folder: str,
    masks_folder: str,
    output_folder: str = None,
    grid_size: int = 20,
    device: str = "cuda",
    frame_pattern: str = "*.png",
    mask_pattern: str = "*.png",
    max_frames: int = None,
    save_video: bool = False
):
    """
    Run deformation detection on real endoscopic video.
    
    Args:
        frames_folder: Path to folder containing video frames
        masks_folder: Path to folder containing binary mask images
        output_folder: Optional folder to save results
        grid_size: Grid size for tracking points
        device: Device ('cuda' or 'cpu')
        frame_pattern: Pattern for frame files
        mask_pattern: Pattern for mask files  
        max_frames: Maximum frames to process (None = all)
        save_video: Whether to save summary video
    """
    print("\n" + "=" * 70)
    print("TYMPANIC MEMBRANE DEFORMATION DETECTION - REAL DATA")
    print("=" * 70)
    
    print(f"\nInput:")
    print(f"  Frames: {frames_folder}")
    print(f"  Masks:  {masks_folder}")
    print(f"  Grid:   {grid_size}x{grid_size}")
    print(f"  Device: {device}")
    
    # Create detector
    detector = TympanicDeformationDetector(
        grid_size=grid_size,
        device=device,
        verbose=True
    )
    
    # Run detection
    print("\n" + "-" * 70)
    result = detector.detect(
        frames_folder=frames_folder,
        masks_folder=masks_folder,
        frame_pattern=frame_pattern,
        mask_pattern=mask_pattern,
        max_frames=max_frames
    )
    print("-" * 70)
    
    # Print results
    print("\n" + result.summary())
    
    # Create output folder if specified
    if output_folder:
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save temporal profile
        print("\nSaving visualizations...")
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        # Temporal profile
        from tympanic_detection.visualization import plot_temporal_profile
        plt.figure(figsize=(14, 10))
        plot_temporal_profile(result, save_path=output_path / "temporal_profile.png")
        plt.close()
        
        # Tracking quality
        from tympanic_detection.visualization import plot_tracking_quality
        plt.figure(figsize=(12, 8))
        plot_tracking_quality(result, save_path=output_path / "tracking_quality.png")
        plt.close()
        
        # Residual fields at key frames
        if result.final_result.deformation_start is not None:
            plt.figure(figsize=(15, 5))
            plot_residual_field(
                result, 
                result.final_result.deformation_start,
                save_path=output_path / f"residual_frame_{result.final_result.deformation_start}.png"
            )
            plt.close()
        
        if result.final_result.peak_start is not None:
            plt.figure(figsize=(15, 5))
            plot_residual_field(
                result,
                result.final_result.peak_start, 
                save_path=output_path / f"residual_frame_{result.final_result.peak_start}.png"
            )
            plt.close()
        
        # Save state sequence to file
        np.save(output_path / "states.npy", result.final_result.smoothed_states)
        
        # Save summary to text file
        with open(output_path / "summary.txt", "w") as f:
            f.write(result.summary())
        
        print(f"Results saved to: {output_folder}")
        
        # Optional: Create summary video
        if save_video:
            video_path = output_path / "detection_result.mp4"
            create_summary_video(result, video_path, fps=10)
    
    # Interactive visualization (if not saving)
    else:
        print("\nShowing visualizations (close windows to continue)...")
        plot_temporal_profile(result)
        plot_tracking_quality(result)
        
        # Show residual fields for key frames
        if result.final_result.deformation_start is not None:
            print(f"\nResidual field at deformation start (frame {result.final_result.deformation_start}):")
            plot_residual_field(result, result.final_result.deformation_start)
        
        if result.final_result.peak_start is not None:
            print(f"\nResidual field at peak (frame {result.final_result.peak_start}):")
            plot_residual_field(result, result.final_result.peak_start)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Tympanic membrane deformation detection on real endoscopic video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python test_real.py --frames /path/to/frames --masks /path/to/masks
  
  # Save results
  python test_real.py --frames /path/to/frames --masks /path/to/masks --output ./results
  
  # Use CPU and limit frames
  python test_real.py --frames /path/to/frames --masks /path/to/masks --device cpu --max_frames 100
  
  # Create summary video
  python test_real.py --frames /path/to/frames --masks /path/to/masks --output ./results --save_video
        """
    )
    
    parser.add_argument("--frames", type=str, required=True,
                        help="Path to folder containing video frames")
    parser.add_argument("--masks", type=str, required=True,
                        help="Path to folder containing binary mask images")
    parser.add_argument("--output", type=str, default=None,
                        help="Output folder for results (optional)")
    parser.add_argument("--grid_size", type=int, default=20,
                        help="Grid size for tracking points (default: 20)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: 'cuda' or 'cpu' (default: cuda)")
    parser.add_argument("--frame_pattern", type=str, default="*.png",
                        help="Glob pattern for frame files (default: *.png)")
    parser.add_argument("--mask_pattern", type=str, default="*.png",
                        help="Glob pattern for mask files (default: *.png)")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Maximum number of frames to process")
    parser.add_argument("--save_video", action="store_true",
                        help="Save summary video with overlay")
    
    args = parser.parse_args()
    
    run_detection(
        frames_folder=args.frames,
        masks_folder=args.masks,
        output_folder=args.output,
        grid_size=args.grid_size,
        device=args.device,
        frame_pattern=args.frame_pattern,
        mask_pattern=args.mask_pattern,
        max_frames=args.max_frames,
        save_video=args.save_video
    )


if __name__ == "__main__":
    main()
