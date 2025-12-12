"""
Module 1: Preprocessing

Load video frames and masks, sample grid points within the tympanic membrane region.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Union
from dataclasses import dataclass
import cv2


@dataclass
class PreprocessingResult:
    """Result of preprocessing step."""
    frames: np.ndarray  # [T, H, W, C]
    masks: np.ndarray  # [T, H, W] binary masks
    query_points: np.ndarray  # [N, 3] (frame_idx, x, y)
    grid_shape: Tuple[int, int]  # (rows, cols)
    point_indices_in_mask: np.ndarray  # [N] indices of valid points
    neighbor_indices: List[np.ndarray]  # neighbor indices for each point


def load_frames_from_folder(
    folder_path: Union[str, Path],
    pattern: str = "*.png",
    max_frames: Optional[int] = None
) -> np.ndarray:
    """
    Load video frames from a folder of images.
    
    Args:
        folder_path: Path to folder containing frame images
        pattern: Glob pattern for image files
        max_frames: Maximum number of frames to load
        
    Returns:
        frames: [T, H, W, C] numpy array of frames
    """
    folder = Path(folder_path)
    image_files = sorted(folder.glob(pattern))
    
    if max_frames is not None:
        image_files = image_files[:max_frames]
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {folder_path} with pattern {pattern}")
    
    frames = []
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)
    
    return np.stack(frames, axis=0)


def load_masks_from_folder(
    folder_path: Union[str, Path],
    pattern: str = "*.png",
    max_frames: Optional[int] = None
) -> np.ndarray:
    """
    Load binary masks from a folder.
    
    Args:
        folder_path: Path to folder containing mask images
        pattern: Glob pattern for mask files
        max_frames: Maximum number of masks to load
        
    Returns:
        masks: [T, H, W] numpy array of binary masks
    """
    folder = Path(folder_path)
    mask_files = sorted(folder.glob(pattern))
    
    if max_frames is not None:
        mask_files = mask_files[:max_frames]
    
    if len(mask_files) == 0:
        raise ValueError(f"No masks found in {folder_path} with pattern {pattern}")
    
    masks = []
    for mask_path in mask_files:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")
        # Binarize
        mask = (mask > 127).astype(np.uint8)
        masks.append(mask)
    
    return np.stack(masks, axis=0)


def sample_grid_points_in_mask(
    mask: np.ndarray,
    grid_size: int = 20,
    margin: int = 5
) -> Tuple[np.ndarray, Tuple[int, int], np.ndarray]:
    """
    Sample uniform grid points within a binary mask.
    
    Args:
        mask: [H, W] binary mask
        grid_size: Number of points per dimension (grid_size x grid_size)
        margin: Margin from bounding box edges
        
    Returns:
        points: [N, 2] (x, y) coordinates of sampled points
        grid_shape: (rows, cols) shape of the grid
        valid_mask: [grid_size*grid_size] boolean mask of points inside the mask
    """
    H, W = mask.shape
    
    # Find bounding box of mask
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("Empty mask provided")
    
    x_min, x_max = xs.min() + margin, xs.max() - margin
    y_min, y_max = ys.min() + margin, ys.max() - margin
    
    # Create grid within bounding box
    x_coords = np.linspace(x_min, x_max, grid_size)
    y_coords = np.linspace(y_min, y_max, grid_size)
    
    xx, yy = np.meshgrid(x_coords, y_coords)
    grid_points = np.stack([xx.ravel(), yy.ravel()], axis=1)  # [grid_size^2, 2]
    
    # Filter points that fall within the mask
    valid_mask = np.zeros(len(grid_points), dtype=bool)
    for i, (x, y) in enumerate(grid_points):
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < W and 0 <= yi < H and mask[yi, xi] > 0:
            valid_mask[i] = True
    
    return grid_points, (grid_size, grid_size), valid_mask


def build_grid_neighbors(
    grid_shape: Tuple[int, int],
    valid_mask: np.ndarray,
    connectivity: int = 8
) -> List[np.ndarray]:
    """
    Build neighbor indices for grid points.
    
    Args:
        grid_shape: (rows, cols) shape of the grid
        valid_mask: [N] boolean mask of valid points
        connectivity: 4 or 8 connectivity
        
    Returns:
        neighbors: List of arrays, each containing neighbor indices for a point
    """
    rows, cols = grid_shape
    n_points = rows * cols
    
    if connectivity == 8:
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    else:  # 4-connectivity
        offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    
    neighbors = []
    for idx in range(n_points):
        r, c = idx // cols, idx % cols
        neighbor_list = []
        
        for dr, dc in offsets:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                neighbor_idx = nr * cols + nc
                # Only include if both current and neighbor are valid
                if valid_mask[idx] and valid_mask[neighbor_idx]:
                    neighbor_list.append(neighbor_idx)
        
        neighbors.append(np.array(neighbor_list, dtype=np.int32))
    
    return neighbors


def preprocess(
    frames_folder: Union[str, Path],
    masks_folder: Union[str, Path],
    frame_pattern: str = "*.png",
    mask_pattern: str = "*.png",
    grid_size: int = 20,
    max_frames: Optional[int] = None
) -> PreprocessingResult:
    """
    Complete preprocessing pipeline.
    
    Args:
        frames_folder: Path to folder containing video frames
        masks_folder: Path to folder containing binary masks
        frame_pattern: Glob pattern for frame files
        mask_pattern: Glob pattern for mask files
        grid_size: Number of grid points per dimension
        max_frames: Maximum number of frames to process
        
    Returns:
        PreprocessingResult containing all preprocessing outputs
    """
    # Load frames and masks
    frames = load_frames_from_folder(frames_folder, frame_pattern, max_frames)
    masks = load_masks_from_folder(masks_folder, mask_pattern, max_frames)
    
    if len(frames) != len(masks):
        raise ValueError(f"Number of frames ({len(frames)}) != number of masks ({len(masks)})")
    
    print(f"Loaded {len(frames)} frames, shape: {frames.shape}")
    
    # Sample grid points using the first frame's mask
    first_mask = masks[0]
    grid_points, grid_shape, valid_mask = sample_grid_points_in_mask(first_mask, grid_size)
    
    # Build query points for CoTracker: (frame_idx, x, y)
    # We query from frame 0
    query_points = np.zeros((len(grid_points), 3), dtype=np.float32)
    query_points[:, 0] = 0  # frame index
    query_points[:, 1:] = grid_points  # x, y coordinates
    
    # Build neighbor indices
    neighbors = build_grid_neighbors(grid_shape, valid_mask)
    
    n_valid = valid_mask.sum()
    print(f"Sampled {grid_size}x{grid_size}={len(grid_points)} grid points, "
          f"{n_valid} inside mask ({100*n_valid/len(grid_points):.1f}%)")
    
    return PreprocessingResult(
        frames=frames,
        masks=masks,
        query_points=query_points,
        grid_shape=grid_shape,
        point_indices_in_mask=valid_mask,
        neighbor_indices=neighbors
    )


if __name__ == "__main__":
    # Test with example paths
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=str, required=True, help="Path to frames folder")
    parser.add_argument("--masks", type=str, required=True, help="Path to masks folder")
    parser.add_argument("--grid_size", type=int, default=20)
    args = parser.parse_args()
    
    result = preprocess(args.frames, args.masks, grid_size=args.grid_size)
    print(f"Frames shape: {result.frames.shape}")
    print(f"Query points shape: {result.query_points.shape}")
    print(f"Grid shape: {result.grid_shape}")
