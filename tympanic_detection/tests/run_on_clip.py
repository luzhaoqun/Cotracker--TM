"""
Test on Your Specific Data Structure

Data structure:
  clip_XXXX/
  ├── XXXXXX.jpg        (video frames)
  └── maskB/
      └── maskB_XXXXXX.png  (masks)
"""

import sys
from pathlib import Path
import argparse

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
from tympanic_detection.preprocessing import (
    load_frames_from_folder,
    load_masks_from_folder,
    sample_grid_points_in_mask,
    build_grid_neighbors,
    PreprocessingResult
)
import numpy as np
import cv2


def load_your_data(clip_folder: str, max_frames: int = None):
    """
    Load data from your specific folder structure.
    
    Args:
        clip_folder: Path to clip_XXXX folder
        max_frames: Maximum frames to load
        
    Returns:
        frames: [T, H, W, 3] numpy array
        masks: [T, H, W] numpy array
    """
    clip_path = Path(clip_folder)
    mask_path = clip_path / "maskB"
    
    if not clip_path.exists():
        raise ValueError(f"Clip folder not found: {clip_path}")
    if not mask_path.exists():
        raise ValueError(f"Mask folder not found: {mask_path}")
    
    # Get frame files (6-digit .jpg)
    frame_files = sorted(clip_path.glob("*.jpg"))
    
    # Get mask files (maskB_XXXXXX.png)
    mask_files = sorted(mask_path.glob("maskB_*.png"))
    
    print(f"Found {len(frame_files)} frames, {len(mask_files)} masks")
    
    if len(frame_files) == 0:
        raise ValueError(f"No .jpg files found in {clip_path}")
    if len(mask_files) == 0:
        raise ValueError(f"No maskB_*.png files found in {mask_path}")
    
    # Match frames and masks by their numeric part
    # Frame: 000001.jpg -> 000001
    # Mask:  maskB_000001.png -> 000001
    
    frame_dict = {}
    for f in frame_files:
        num = f.stem  # e.g., "000001"
        frame_dict[num] = f
    
    mask_dict = {}
    for m in mask_files:
        # Extract number from maskB_XXXXXX.png
        num = m.stem.replace("maskB_", "")  # e.g., "000001"
        mask_dict[num] = m
    
    # Find matching pairs
    common_nums = sorted(set(frame_dict.keys()) & set(mask_dict.keys()))
    print(f"Matched {len(common_nums)} frame-mask pairs")
    
    if len(common_nums) == 0:
        raise ValueError("No matching frame-mask pairs found!")
    
    if max_frames is not None:
        common_nums = common_nums[:max_frames]
    
    # Load frames
    frames = []
    for num in common_nums:
        img = cv2.imread(str(frame_dict[num]))
        if img is None:
            raise ValueError(f"Failed to load: {frame_dict[num]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)
    
    frames = np.stack(frames, axis=0)
    
    # Load masks
    masks = []
    for num in common_nums:
        mask = cv2.imread(str(mask_dict[num]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load: {mask_dict[num]}")
        mask = (mask > 127).astype(np.uint8)
        masks.append(mask)
    
    masks = np.stack(masks, axis=0)
    
    print(f"Loaded frames shape: {frames.shape}")
    print(f"Loaded masks shape: {masks.shape}")
    
    return frames, masks


def run_on_clip(
    clip_folder: str,
    output_folder: str = None,
    grid_size: int = 20,
    device: str = "cuda",
    max_frames: int = None,
    show_plots: bool = True
):
    """
    Run detection on a clip folder.
    """
    print("\n" + "=" * 70)
    print("TYMPANIC MEMBRANE DEFORMATION DETECTION")
    print("=" * 70)
    print(f"\nClip: {clip_folder}")
    
    # Load data
    print("\n--- Loading Data ---")
    frames, masks = load_your_data(clip_folder, max_frames)
    
    # Create detector
    detector = TympanicDeformationDetector(
        grid_size=grid_size,
        device=device,
        verbose=True
    )
    
    # Run detection using array input
    print("\n--- Running Detection ---")
    result = detector.detect_from_arrays(frames, masks)
    
    # Print results
    print("\n" + result.summary())
    
    # Save or show results
    if output_folder:
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save plots
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # Temporal profile
        fig = plt.figure(figsize=(14, 10))
        plot_temporal_profile(result, save_path=output_path / "temporal_profile.png")
        plt.close()
        
        # Tracking quality
        fig = plt.figure(figsize=(12, 8))
        plot_tracking_quality(result, save_path=output_path / "tracking_quality.png")
        plt.close()
        
        # Key frames
        for frame_idx, name in [
            (result.final_result.deformation_start, "deformation_start"),
            (result.final_result.peak_start, "peak_start")
        ]:
            if frame_idx is not None:
                fig = plt.figure(figsize=(15, 5))
                plot_residual_field(result, frame_idx, 
                                   save_path=output_path / f"residual_{name}_frame{frame_idx}.png")
                plt.close()
        
        # Save data
        np.save(output_path / "states.npy", result.final_result.smoothed_states)
        with open(output_path / "summary.txt", "w", encoding="utf-8") as f:
            f.write(result.summary())
        
        # Save summary video
        video_path = output_path / "detection_video.mp4"
        create_summary_video(result, video_path, fps=15)
        
        print(f"\nResults saved to: {output_path}")
    
    elif show_plots:
        print("\nShowing visualizations...")
        plot_temporal_profile(result)
        plot_tracking_quality(result)
        
        if result.final_result.peak_start is not None:
            plot_residual_field(result, result.final_result.peak_start)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run deformation detection on clip folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on a single clip
  python run_on_clip.py --clip /path/to/clip_0001
  
  # Save results
  python run_on_clip.py --clip /path/to/clip_0001 --output ./results/clip_0001
  
  # Quick test with limited frames
  python run_on_clip.py --clip /path/to/clip_0001 --max_frames 50
  
  # Use CPU
  python run_on_clip.py --clip /path/to/clip_0001 --device cpu
        """
    )
    
    parser.add_argument("--clip", type=str, required=True,
                        help="Path to clip_XXXX folder")
    parser.add_argument("--output", type=str, default=None,
                        help="Output folder for results")
    parser.add_argument("--grid_size", type=int, default=20,
                        help="Grid size (default: 20)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu (default: cuda)")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Maximum frames to process")
    parser.add_argument("--no_plot", action="store_true",
                        help="Disable interactive plots")
    
    args = parser.parse_args()
    
    run_on_clip(
        clip_folder=args.clip,
        output_folder=args.output,
        grid_size=args.grid_size,
        device=args.device,
        max_frames=args.max_frames,
        show_plots=not args.no_plot
    )


if __name__ == "__main__":
    main()
