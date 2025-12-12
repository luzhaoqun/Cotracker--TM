"""
Module 3: Quality Control

Filter unreliable tracking points using visibility, temporal smoothness,
and spatial consistency checks.
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class QualityControlResult:
    """Result of quality control step."""
    weights: np.ndarray  # [T, N] reliability weights (0-1)
    temporal_anomaly: np.ndarray  # [T, N] temporal anomaly flags
    spatial_anomaly: np.ndarray  # [T, N] spatial anomaly flags
    repaired_tracks: np.ndarray  # [T, N, 2] tracks with anomalies repaired


class TrackQualityController:
    """
    Multi-layer quality control for tracking points.
    """
    
    def __init__(
        self,
        visibility_threshold: float = 0.5,
        max_velocity: float = 15.0,
        max_acceleration: float = 10.0,
        spatial_std_threshold: float = 2.5,
        neighbor_indices: Optional[List[np.ndarray]] = None
    ):
        """
        Args:
            visibility_threshold: Minimum visibility score to consider reliable
            max_velocity: Maximum allowed velocity (pixels/frame)
            max_acceleration: Maximum allowed acceleration
            spatial_std_threshold: Threshold for spatial consistency (in std units)
            neighbor_indices: Pre-computed neighbor indices for spatial check
        """
        self.visibility_threshold = visibility_threshold
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.spatial_std_threshold = spatial_std_threshold
        self.neighbor_indices = neighbor_indices
    
    def process(
        self,
        tracks: np.ndarray,
        visibility: np.ndarray
    ) -> QualityControlResult:
        """
        Run complete quality control pipeline.
        
        Args:
            tracks: [T, N, 2] point trajectories
            visibility: [T, N] visibility scores
            
        Returns:
            QualityControlResult
        """
        T, N = tracks.shape[:2]
        
        # Layer 1: Visibility check
        low_vis_mask = visibility < self.visibility_threshold
        
        # Layer 2: Temporal smoothness check
        temporal_anomaly = self._detect_temporal_anomaly(tracks)
        
        # Layer 3: Spatial consistency check
        if self.neighbor_indices is not None:
            spatial_anomaly = self._detect_spatial_anomaly(tracks)
        else:
            spatial_anomaly = np.zeros((T, N), dtype=bool)
        
        # Combine all anomalies
        combined_anomaly = low_vis_mask | temporal_anomaly | spatial_anomaly
        
        # Compute reliability weights
        weights = self._compute_weights(visibility, temporal_anomaly, spatial_anomaly)
        
        # Repair tracks (interpolate anomalous points)
        repaired_tracks = self._repair_tracks(tracks, combined_anomaly)
        
        return QualityControlResult(
            weights=weights,
            temporal_anomaly=temporal_anomaly,
            spatial_anomaly=spatial_anomaly,
            repaired_tracks=repaired_tracks
        )
    
    def _detect_temporal_anomaly(self, tracks: np.ndarray) -> np.ndarray:
        """
        Detect points with unrealistic motion (sudden jumps).
        """
        T, N = tracks.shape[:2]
        anomaly = np.zeros((T, N), dtype=bool)
        
        # Compute velocity
        velocity = np.zeros((T, N, 2))
        velocity[1:] = tracks[1:] - tracks[:-1]
        speed = np.linalg.norm(velocity, axis=2)
        
        # Check velocity threshold
        velocity_anomaly = speed > self.max_velocity
        anomaly |= velocity_anomaly
        
        # Compute acceleration
        acceleration = np.zeros((T, N, 2))
        acceleration[1:] = velocity[1:] - velocity[:-1]
        accel_magnitude = np.linalg.norm(acceleration, axis=2)
        
        # Check acceleration threshold
        accel_anomaly = accel_magnitude > self.max_acceleration
        anomaly |= accel_anomaly
        
        return anomaly
    
    def _detect_spatial_anomaly(self, tracks: np.ndarray) -> np.ndarray:
        """
        Detect points that move inconsistently with their neighbors.
        """
        T, N = tracks.shape[:2]
        anomaly = np.zeros((T, N), dtype=bool)
        
        if self.neighbor_indices is None:
            return anomaly
        
        for t in range(1, T):
            displacement = tracks[t] - tracks[t-1]  # [N, 2]
            
            for n in range(N):
                neighbors = self.neighbor_indices[n]
                if len(neighbors) < 2:
                    continue
                
                # Neighbor displacements
                neighbor_displacements = displacement[neighbors]
                neighbor_mean = neighbor_displacements.mean(axis=0)
                neighbor_std = neighbor_displacements.std(axis=0) + 1e-6
                
                # Check if current point deviates too much
                deviation = np.abs(displacement[n] - neighbor_mean)
                if (deviation > self.spatial_std_threshold * neighbor_std + 1.0).any():
                    anomaly[t, n] = True
        
        return anomaly
    
    def _compute_weights(
        self,
        visibility: np.ndarray,
        temporal_anomaly: np.ndarray,
        spatial_anomaly: np.ndarray
    ) -> np.ndarray:
        """
        Compute reliability weights combining all quality signals.
        """
        # Start with visibility as base weight (ensure float64 type)
        weights = visibility.astype(np.float64).copy()
        
        # Reduce weight for temporal anomalies
        weights[temporal_anomaly] *= 0.2
        
        # Reduce weight for spatial anomalies
        weights[spatial_anomaly] *= 0.3
        
        # Clip to [0, 1]
        weights = np.clip(weights, 0.0, 1.0)
        
        return weights
    
    def _repair_tracks(
        self,
        tracks: np.ndarray,
        anomaly_mask: np.ndarray,
        window_size: int = 5
    ) -> np.ndarray:
        """
        Repair anomalous points using linear interpolation.
        """
        T, N = tracks.shape[:2]
        repaired = tracks.copy()
        
        for n in range(N):
            # Find anomalous frames for this point
            anomaly_frames = np.where(anomaly_mask[:, n])[0]
            
            for t in anomaly_frames:
                # Find valid neighbors in time
                start = max(0, t - window_size)
                end = min(T, t + window_size + 1)
                
                valid_mask = ~anomaly_mask[start:end, n]
                valid_times = np.arange(start, end)[valid_mask]
                
                if len(valid_times) >= 2:
                    valid_positions = tracks[valid_times, n]
                    # Linear interpolation
                    repaired[t, n, 0] = np.interp(t, valid_times, valid_positions[:, 0])
                    repaired[t, n, 1] = np.interp(t, valid_times, valid_positions[:, 1])
        
        return repaired


def estimate_tracking_noise(
    tracks: np.ndarray,
    visibility: np.ndarray,
    n_static_frames: int = 10
) -> float:
    """
    Estimate tracking noise level from static frames at the beginning.
    
    Args:
        tracks: [T, N, 2] trajectories
        visibility: [T, N] visibility scores
        n_static_frames: Number of frames to use (assumed static)
        
    Returns:
        Estimated noise level (std of displacements in static region)
    """
    n_frames = min(n_static_frames, tracks.shape[0])
    
    if n_frames < 2:
        return 1.0  # Default
    
    displacements = []
    for t in range(1, n_frames):
        disp = tracks[t] - tracks[t-1]
        high_vis = visibility[t] > 0.5
        if high_vis.sum() > 0:
            displacements.append(np.linalg.norm(disp[high_vis], axis=1))
    
    if len(displacements) == 0:
        return 1.0
    
    all_displacements = np.concatenate(displacements)
    noise_level = np.std(all_displacements)
    
    return max(noise_level, 0.1)  # Minimum 0.1 pixel


if __name__ == "__main__":
    print("Quality control module loaded successfully")
