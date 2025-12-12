"""
Module 2: CoTracker Tracking

Wrapper for CoTracker to track points across video frames.
"""

import numpy as np
import torch
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class TrackingResult:
    """Result of CoTracker tracking."""
    tracks: np.ndarray  # [T, N, 2] point trajectories
    visibility: np.ndarray  # [T, N] visibility scores (0-1)


def run_cotracker(
    frames: np.ndarray,
    query_points: np.ndarray,
    model_name: str = "cotracker3_offline",
    device: str = "cuda"
) -> TrackingResult:
    """
    Run CoTracker on video frames.
    
    Args:
        frames: [T, H, W, C] video frames (uint8 RGB)
        query_points: [N, 3] query points (frame_idx, x, y)
        model_name: CoTracker model to use
        device: Device to run on ('cuda' or 'cpu')
        
    Returns:
        TrackingResult with tracks and visibility
    """
    # Check device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    
    # Load model
    print(f"Loading {model_name}...")
    cotracker = torch.hub.load("facebookresearch/co-tracker", model_name)
    cotracker = cotracker.to(device)
    cotracker.eval()
    
    # Prepare video tensor: [B, T, C, H, W]
    video = torch.from_numpy(frames).permute(0, 3, 1, 2).float()  # [T, C, H, W]
    video = video.unsqueeze(0).to(device)  # [1, T, C, H, W]
    
    # Prepare query points: [B, N, 3]
    queries = torch.from_numpy(query_points).float().unsqueeze(0).to(device)
    
    print(f"Video shape: {video.shape}, Queries shape: {queries.shape}")
    print("Running CoTracker...")
    
    # Run tracking
    with torch.no_grad():
        pred_tracks, pred_visibility = cotracker(video, queries=queries)
    
    # pred_tracks: [B, T, N, 2]
    # pred_visibility: [B, T, N, 1] or [B, T, N]
    
    tracks = pred_tracks[0].cpu().numpy()  # [T, N, 2]
    visibility = pred_visibility[0].cpu().numpy()  # [T, N, 1] or [T, N]
    
    if visibility.ndim == 3:
        visibility = visibility.squeeze(-1)  # [T, N]
    
    print(f"Tracking complete. Tracks shape: {tracks.shape}")
    
    return TrackingResult(tracks=tracks, visibility=visibility)


def run_cotracker_online(
    frames: np.ndarray,
    query_points: np.ndarray,
    device: str = "cuda",
    step_size: Optional[int] = None
) -> TrackingResult:
    """
    Run CoTracker in online mode for longer videos.
    
    Args:
        frames: [T, H, W, C] video frames
        query_points: [N, 3] query points
        device: Device to run on
        step_size: Processing step size (default: model's default)
        
    Returns:
        TrackingResult with tracks and visibility
    """
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    
    print("Loading cotracker3_online...")
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")
    cotracker = cotracker.to(device)
    cotracker.eval()
    
    if step_size is None:
        step_size = cotracker.step
    
    # Prepare video tensor
    video = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    video = video.unsqueeze(0).to(device)
    
    # Prepare queries
    queries = torch.from_numpy(query_points).float().unsqueeze(0).to(device)
    
    T = video.shape[1]
    
    print(f"Running online tracking with step={step_size}...")
    
    # Initialize
    with torch.no_grad():
        cotracker(video_chunk=video[:, :step_size*2], is_first_step=True, queries=queries)
        
        # Process chunks
        all_tracks = []
        all_visibility = []
        
        for ind in range(0, T - step_size, step_size):
            end_idx = min(ind + step_size * 2, T)
            pred_tracks, pred_visibility = cotracker(
                video_chunk=video[:, ind:end_idx]
            )
            all_tracks.append(pred_tracks)
            all_visibility.append(pred_visibility)
    
    # Use the last prediction which covers the full video
    tracks = all_tracks[-1][0].cpu().numpy()
    visibility = all_visibility[-1][0].cpu().numpy()
    
    if visibility.ndim == 3:
        visibility = visibility.squeeze(-1)
    
    print(f"Online tracking complete. Tracks shape: {tracks.shape}")
    
    return TrackingResult(tracks=tracks, visibility=visibility)


if __name__ == "__main__":
    # Simple test
    print("Tracking module loaded successfully")
