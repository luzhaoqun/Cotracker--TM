"""
Visualization Tools

Visualize tracking, residuals, and detection results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pathlib import Path
from typing import Optional, List, Union

from .pipeline import DetectionResult
from .classification import DeformationState


def plot_temporal_profile(
    result: DetectionResult,
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (14, 10)
):
    """
    Plot temporal profile of deformation detection.
    
    Args:
        result: Detection result
        save_path: Optional path to save figure
        figsize: Figure size
    """
    features = result.features
    final = result.final_result
    
    T = len(features)
    frames = np.arange(T)
    
    # Extract feature arrays
    median_residuals = np.array([f.median_residual for f in features])
    inlier_ratios = np.array([f.inlier_ratio for f in features])
    center_vs_edge = np.array([f.center_vs_edge_ratio for f in features])
    
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    
    # 1. Median residual
    ax = axes[0]
    ax.plot(frames, median_residuals, 'b-', linewidth=2, label='Median Residual')
    ax.fill_between(frames, 0, median_residuals, alpha=0.3)
    ax.set_ylabel('Residual (px)')
    ax.legend(loc='upper right')
    ax.set_title('Deformation Detection - Temporal Profile')
    ax.grid(True, alpha=0.3)
    
    # 2. Inlier ratio
    ax = axes[1]
    ax.plot(frames, inlier_ratios, 'g-', linewidth=2, label='Inlier Ratio')
    ax.axhline(0.9, color='r', linestyle='--', alpha=0.5, label='Threshold')
    ax.set_ylabel('Inlier Ratio')
    ax.set_ylim(0, 1.1)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # 3. Center vs Edge ratio
    ax = axes[2]
    ax.plot(frames, center_vs_edge, 'm-', linewidth=2, label='Center/Edge Ratio')
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Center/Edge')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 4. State classification
    ax = axes[3]
    
    # Color map for states
    state_colors = {
        DeformationState.STATIC: 'green',
        DeformationState.DEFORMING: 'orange',
        DeformationState.PEAK: 'red'
    }
    state_labels = {
        DeformationState.STATIC: 'Static',
        DeformationState.DEFORMING: 'Deforming',
        DeformationState.PEAK: 'Peak'
    }
    
    # Plot raw states (transparent)
    for state in [0, 1, 2]:
        mask = final.raw_states == state
        ax.fill_between(frames, 0, 1, where=mask, 
                       color=state_colors[state], alpha=0.2)
    
    # Plot smoothed states (solid)
    for state in [0, 1, 2]:
        mask = final.smoothed_states == state
        ax.fill_between(frames, 0, 0.5, where=mask,
                       color=state_colors[state], alpha=0.8,
                       label=state_labels[state])
    
    # Mark key events
    if final.deformation_start is not None:
        ax.axvline(final.deformation_start, color='orange', linewidth=2, linestyle='-')
        ax.text(final.deformation_start, 0.7, 'Start', rotation=90, va='bottom')
    
    if final.peak_start is not None:
        ax.axvline(final.peak_start, color='red', linewidth=2, linestyle='-')
        ax.text(final.peak_start, 0.7, 'Peak', rotation=90, va='bottom')
    
    ax.set_ylabel('State')
    ax.set_xlabel('Frame')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, T-1)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved temporal profile to {save_path}")
    
    plt.show()


def plot_residual_field(
    result: DetectionResult,
    frame_idx: int,
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (15, 5)
):
    """
    Plot residual field for a specific frame.
    
    Args:
        result: Detection result
        frame_idx: Frame index to visualize
        save_path: Optional path to save figure
        figsize: Figure size
    """
    homography = result.homography
    grid_shape = result.preprocessing.grid_shape
    
    if frame_idx >= len(homography.frame_results):
        print(f"Frame {frame_idx} out of range")
        return
    
    frame_result = homography.frame_results[frame_idx]
    residual = frame_result.residual
    residual_mag = frame_result.residual_magnitude
    inliers = frame_result.inliers
    
    rows, cols = grid_shape
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # 1. Residual magnitude heatmap
    ax = axes[0]
    try:
        mag_grid = residual_mag.reshape(rows, cols)
        im = ax.imshow(mag_grid, cmap='hot', origin='lower')
        plt.colorbar(im, ax=ax, label='Residual (px)')
    except ValueError:
        ax.text(0.5, 0.5, 'Cannot reshape', ha='center', va='center')
    ax.set_title(f'Frame {frame_idx}: Residual Magnitude')
    
    # 2. Residual vector field
    ax = axes[1]
    try:
        # Subsample for cleaner visualization
        step = max(1, rows // 15)
        y_grid, x_grid = np.mgrid[0:rows:step, 0:cols:step]
        residual_grid = residual.reshape(rows, cols, 2)
        
        u = residual_grid[::step, ::step, 0]
        v = residual_grid[::step, ::step, 1]
        mag = np.sqrt(u**2 + v**2)
        
        ax.quiver(x_grid, y_grid, u, v, mag, cmap='coolwarm', scale=50)
        ax.set_xlim(-0.5, cols-0.5)
        ax.set_ylim(-0.5, rows-0.5)
        ax.invert_yaxis()
    except ValueError:
        ax.text(0.5, 0.5, 'Cannot reshape', ha='center', va='center')
    ax.set_title('Residual Vector Field')
    ax.set_aspect('equal')
    
    # 3. Inlier/Outlier map
    ax = axes[2]
    try:
        inlier_grid = inliers.reshape(rows, cols).astype(float)
        ax.imshow(inlier_grid, cmap='RdYlGn', origin='lower', vmin=0, vmax=1)
    except ValueError:
        ax.text(0.5, 0.5, 'Cannot reshape', ha='center', va='center')
    ax.set_title('Inliers (Green) / Outliers (Red)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved residual field to {save_path}")
    
    plt.show()


def plot_tracking_quality(
    result: DetectionResult,
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (12, 8)
):
    """
    Plot tracking quality statistics.
    
    Args:
        result: Detection result
        save_path: Optional path to save figure
        figsize: Figure size
    """
    qc = result.quality_control
    tracking = result.tracking
    
    T, N = qc.weights.shape
    frames = np.arange(T)
    
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    # 1. Average visibility
    ax = axes[0]
    avg_visibility = tracking.visibility.mean(axis=1)
    ax.plot(frames, avg_visibility, 'b-', linewidth=2)
    ax.axhline(0.5, color='r', linestyle='--', alpha=0.5)
    ax.set_ylabel('Avg Visibility')
    ax.set_ylim(0, 1)
    ax.set_title('Tracking Quality Over Time')
    ax.grid(True, alpha=0.3)
    
    # 2. Anomaly rate
    ax = axes[1]
    temporal_anomaly_rate = qc.temporal_anomaly.mean(axis=1)
    spatial_anomaly_rate = qc.spatial_anomaly.mean(axis=1)
    
    ax.plot(frames, temporal_anomaly_rate, 'r-', linewidth=2, label='Temporal')
    ax.plot(frames, spatial_anomaly_rate, 'orange', linewidth=2, label='Spatial')
    ax.set_ylabel('Anomaly Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Average weight
    ax = axes[2]
    avg_weight = qc.weights.mean(axis=1)
    ax.plot(frames, avg_weight, 'g-', linewidth=2)
    ax.set_ylabel('Avg Weight')
    ax.set_xlabel('Frame')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved tracking quality plot to {save_path}")
    
    plt.show()


def create_summary_video(
    result: DetectionResult,
    output_path: Union[str, Path],
    fps: int = 10
):
    """
    Create summary video with overlaid tracking and states.
    
    Args:
        result: Detection result
        output_path: Output video path
        fps: Frames per second
    """
    try:
        import cv2
    except ImportError:
        print("OpenCV required for video export")
        return
    
    frames = result.preprocessing.frames
    tracks = result.tracking.tracks
    states = result.final_result.smoothed_states
    
    T, H, W, C = frames.shape
    
    # State colors (BGR for OpenCV)
    state_colors = {
        0: (0, 255, 0),    # Green - Static
        1: (0, 165, 255),  # Orange - Deforming
        2: (0, 0, 255)     # Red - Peak
    }
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))
    
    for t in range(min(T, len(states))):
        frame = frames[t].copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Draw tracking points
        points = tracks[t]
        for i, (x, y) in enumerate(points):
            if 0 <= x < W and 0 <= y < H:
                cv2.circle(frame, (int(x), int(y)), 2, (255, 255, 0), -1)
        
        # Draw state indicator
        state = states[t]
        color = state_colors.get(state, (255, 255, 255))
        cv2.rectangle(frame, (10, 10), (60, 40), color, -1)
        
        state_text = ['STATIC', 'DEFORM', 'PEAK'][state]
        cv2.putText(frame, state_text, (70, 35), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, color, 2)
        
        cv2.putText(frame, f'Frame: {t}', (W-120, 35), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"Saved summary video to {output_path}")


if __name__ == "__main__":
    print("Visualization module loaded successfully")
