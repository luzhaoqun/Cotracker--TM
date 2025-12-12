"""
Module 5: Feature Extraction

Extract discriminative features from residual fields for deformation classification.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .homography_analysis import HomographyAnalysisResult, FrameHomographyResult


@dataclass
class FrameFeatures:
    """Features extracted for a single frame."""
    frame_idx: int
    
    # Residual magnitude statistics (robust)
    median_residual: float
    mean_residual: float
    max_residual: float
    std_residual: float
    
    # Inlier-based features
    inlier_ratio: float
    
    # Spatial distribution features
    center_residual: float  # Average residual in center region
    edge_residual: float  # Average residual at edges
    center_vs_edge_ratio: float  # Ratio indicating bulging pattern
    
    # Divergence (positive = outward expansion)
    divergence: float
    
    # High residual area ratio
    high_residual_ratio: float


class FeatureExtractor:
    """
    Extract deformation features from homography residuals.
    """
    
    def __init__(
        self,
        grid_shape: Tuple[int, int],
        valid_mask: Optional[np.ndarray] = None,
        high_residual_threshold: float = 2.0
    ):
        """
        Args:
            grid_shape: (rows, cols) shape of point grid
            valid_mask: [N] boolean mask of valid points
            high_residual_threshold: Threshold for "high residual" classification
        """
        self.grid_shape = grid_shape
        self.valid_mask = valid_mask
        self.high_residual_threshold = high_residual_threshold
        
        # Pre-compute center and edge masks
        self._build_region_masks()
    
    def _build_region_masks(self):
        """Build masks for center and edge regions."""
        rows, cols = self.grid_shape
        n_points = rows * cols
        
        self.center_mask = np.zeros(n_points, dtype=bool)
        self.edge_mask = np.zeros(n_points, dtype=bool)
        
        # Define center as middle 50% of grid
        center_r_start, center_r_end = rows // 4, rows * 3 // 4
        center_c_start, center_c_end = cols // 4, cols * 3 // 4
        
        for idx in range(n_points):
            r, c = idx // cols, idx % cols
            
            # Edge: first/last 2 rows/cols
            is_edge = (r < 2 or r >= rows - 2 or c < 2 or c >= cols - 2)
            
            # Center: middle region
            is_center = (center_r_start <= r < center_r_end and 
                        center_c_start <= c < center_c_end)
            
            self.edge_mask[idx] = is_edge
            self.center_mask[idx] = is_center
        
        # Apply valid mask if provided
        if self.valid_mask is not None:
            self.center_mask &= self.valid_mask
            self.edge_mask &= self.valid_mask
    
    def extract_frame_features(
        self,
        frame_result: FrameHomographyResult
    ) -> FrameFeatures:
        """
        Extract features for a single frame.
        """
        residual = frame_result.residual  # [N, 2]
        residual_mag = frame_result.residual_magnitude  # [N]
        
        # Apply valid mask
        if self.valid_mask is not None:
            valid_residual_mag = residual_mag[self.valid_mask]
            valid_residual = residual[self.valid_mask]
        else:
            valid_residual_mag = residual_mag
            valid_residual = residual
        
        # Basic statistics (using robust measures)
        median_residual = np.median(valid_residual_mag)
        mean_residual = np.mean(valid_residual_mag)
        max_residual = np.max(valid_residual_mag)
        std_residual = np.std(valid_residual_mag)
        
        # Inlier ratio
        inlier_ratio = frame_result.inlier_ratio
        
        # Center vs edge analysis
        center_points = self.center_mask.sum()
        edge_points = self.edge_mask.sum()
        
        if center_points > 0:
            center_residual = residual_mag[self.center_mask].mean()
        else:
            center_residual = median_residual
        
        if edge_points > 0:
            edge_residual = residual_mag[self.edge_mask].mean()
        else:
            edge_residual = median_residual
        
        # Ratio (high ratio = center bulging pattern)
        center_vs_edge_ratio = center_residual / (edge_residual + 0.1)
        
        # Divergence
        divergence = self._compute_divergence(residual)
        
        # High residual ratio
        threshold = self.high_residual_threshold
        high_residual_ratio = (valid_residual_mag > threshold).mean()
        
        return FrameFeatures(
            frame_idx=frame_result.frame_idx,
            median_residual=median_residual,
            mean_residual=mean_residual,
            max_residual=max_residual,
            std_residual=std_residual,
            inlier_ratio=inlier_ratio,
            center_residual=center_residual,
            edge_residual=edge_residual,
            center_vs_edge_ratio=center_vs_edge_ratio,
            divergence=divergence,
            high_residual_ratio=high_residual_ratio
        )
    
    def _compute_divergence(self, residual: np.ndarray) -> float:
        """
        Compute divergence of the residual field.
        Positive divergence indicates outward expansion (bulging).
        """
        rows, cols = self.grid_shape
        
        # Reshape to grid
        try:
            field = residual.reshape(rows, cols, 2)
        except ValueError:
            return 0.0
        
        # Compute gradients
        du_dx = np.gradient(field[:, :, 0], axis=1)
        dv_dy = np.gradient(field[:, :, 1], axis=0)
        
        divergence = du_dx + dv_dy
        
        # Apply valid mask if available
        if self.valid_mask is not None:
            valid_div = divergence.ravel()[self.valid_mask]
            return float(np.mean(valid_div))
        else:
            return float(np.mean(divergence))
    
    def extract_all_features(
        self,
        homography_result: HomographyAnalysisResult
    ) -> List[FrameFeatures]:
        """
        Extract features for all frames.
        """
        features = []
        for frame_result in homography_result.frame_results:
            feat = self.extract_frame_features(frame_result)
            features.append(feat)
        return features


def features_to_array(features: List[FrameFeatures]) -> np.ndarray:
    """
    Convert feature list to numpy array for classification.
    
    Returns:
        array: [T, n_features] feature array
    """
    n_features = 10
    array = np.zeros((len(features), n_features))
    
    for i, f in enumerate(features):
        array[i] = [
            f.median_residual,
            f.mean_residual,
            f.max_residual,
            f.std_residual,
            f.inlier_ratio,
            f.center_residual,
            f.edge_residual,
            f.center_vs_edge_ratio,
            f.divergence,
            f.high_residual_ratio
        ]
    
    return array


if __name__ == "__main__":
    print("Feature extraction module loaded successfully")
