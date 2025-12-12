"""
Module 4: Homography Analysis

Core module for separating camera motion from deformation using
perspective model fitting and residual analysis.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class FrameHomographyResult:
    """Homography analysis result for a single frame."""
    frame_idx: int
    homography: Optional[np.ndarray]  # [3, 3] homography matrix
    predicted_points: np.ndarray  # [N, 2] predicted positions
    residual: np.ndarray  # [N, 2] residual vectors
    residual_magnitude: np.ndarray  # [N] residual magnitudes
    inliers: np.ndarray  # [N] boolean inlier mask
    inlier_ratio: float


@dataclass
class HomographyAnalysisResult:
    """Complete homography analysis result."""
    frame_results: List[FrameHomographyResult]
    reference_frame: int
    ransac_threshold: float


def apply_homography(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Apply homography transformation to points.
    
    Args:
        points: [N, 2] points
        H: [3, 3] homography matrix
        
    Returns:
        transformed: [N, 2] transformed points
    """
    N = len(points)
    points_homo = np.hstack([points, np.ones((N, 1))])  # [N, 3]
    transformed_homo = (H @ points_homo.T).T  # [N, 3]
    transformed = transformed_homo[:, :2] / (transformed_homo[:, 2:3] + 1e-8)
    return transformed


def estimate_homography_robust(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    weights: Optional[np.ndarray] = None,
    ransac_threshold: float = 3.0,
    method: str = "ransac"
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Robustly estimate homography matrix.
    
    Args:
        src_points: [N, 2] source points
        dst_points: [N, 2] destination points
        weights: [N] optional reliability weights
        ransac_threshold: RANSAC reprojection threshold
        method: 'ransac' or 'lmeds'
        
    Returns:
        H: [3, 3] homography matrix (or None if failed)
        inliers: [N] boolean inlier mask
    """
    # Filter by weights if provided
    if weights is not None:
        reliable_mask = weights > 0.3
        if reliable_mask.sum() < 4:
            reliable_mask = np.ones(len(src_points), dtype=bool)
    else:
        reliable_mask = np.ones(len(src_points), dtype=bool)
    
    src_reliable = src_points[reliable_mask].astype(np.float32)
    dst_reliable = dst_points[reliable_mask].astype(np.float32)
    
    if len(src_reliable) < 4:
        return None, np.zeros(len(src_points), dtype=bool)
    
    # Choose method
    if method == "lmeds":
        cv_method = cv2.LMEDS
    else:
        cv_method = cv2.RANSAC
    
    H, mask = cv2.findHomography(
        src_reliable,
        dst_reliable,
        cv_method,
        ransacReprojThreshold=ransac_threshold,
        maxIters=2000,
        confidence=0.995
    )
    
    # Map back to full point set
    full_inliers = np.zeros(len(src_points), dtype=bool)
    if mask is not None:
        full_inliers[reliable_mask] = mask.ravel().astype(bool)
    
    return H, full_inliers


def estimate_homography_two_stage(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    weights: Optional[np.ndarray] = None,
    initial_threshold: float = 5.0,
    final_threshold: float = 2.0
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Two-stage homography estimation for robustness against large deformations.
    
    Stage 1: Loose threshold to get initial estimate
    Stage 2: Use low-residual points for refined estimate
    """
    # Stage 1: Initial loose fit
    H1, inliers1 = estimate_homography_robust(
        src_points, dst_points, weights, initial_threshold
    )
    
    if H1 is None:
        return None, np.zeros(len(src_points), dtype=bool)
    
    # Compute residuals
    predicted = apply_homography(src_points, H1)
    residuals = np.linalg.norm(dst_points - predicted, axis=1)
    
    # Stage 2: Use only low-residual points
    percentile_threshold = np.percentile(residuals, 50)
    low_residual_mask = residuals < percentile_threshold
    
    if low_residual_mask.sum() < 4:
        return H1, inliers1
    
    H2, inliers2 = cv2.findHomography(
        src_points[low_residual_mask].astype(np.float32),
        dst_points[low_residual_mask].astype(np.float32),
        cv2.RANSAC,
        ransacReprojThreshold=final_threshold
    )
    
    if H2 is None:
        return H1, inliers1
    
    # Map inliers back
    full_inliers = np.zeros(len(src_points), dtype=bool)
    if inliers2 is not None:
        full_inliers[low_residual_mask] = inliers2.ravel().astype(bool)
    
    return H2, full_inliers


class HomographyAnalyzer:
    """
    Analyze deformation by fitting homography and computing residuals.
    """
    
    def __init__(
        self,
        ransac_threshold: float = 3.0,
        use_two_stage: bool = True,
        reference_frame: int = 0
    ):
        """
        Args:
            ransac_threshold: RANSAC reprojection threshold
            use_two_stage: Use two-stage estimation for robustness
            reference_frame: Frame index to use as reference
        """
        self.ransac_threshold = ransac_threshold
        self.use_two_stage = use_two_stage
        self.reference_frame = reference_frame
    
    def analyze(
        self,
        tracks: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> HomographyAnalysisResult:
        """
        Analyze all frames relative to reference frame.
        
        Args:
            tracks: [T, N, 2] point trajectories
            weights: [T, N] optional reliability weights
            
        Returns:
            HomographyAnalysisResult
        """
        T, N = tracks.shape[:2]
        ref_points = tracks[self.reference_frame]
        
        frame_results = []
        
        for t in range(T):
            if t == self.reference_frame:
                # Reference frame has zero residual
                result = FrameHomographyResult(
                    frame_idx=t,
                    homography=np.eye(3),
                    predicted_points=ref_points.copy(),
                    residual=np.zeros((N, 2)),
                    residual_magnitude=np.zeros(N),
                    inliers=np.ones(N, dtype=bool),
                    inlier_ratio=1.0
                )
            else:
                cur_points = tracks[t]
                cur_weights = weights[t] if weights is not None else None
                
                # Estimate homography
                if self.use_two_stage:
                    H, inliers = estimate_homography_two_stage(
                        ref_points, cur_points, cur_weights
                    )
                else:
                    H, inliers = estimate_homography_robust(
                        ref_points, cur_points, cur_weights, self.ransac_threshold
                    )
                
                if H is not None:
                    predicted = apply_homography(ref_points, H)
                    residual = cur_points - predicted
                    residual_magnitude = np.linalg.norm(residual, axis=1)
                    inlier_ratio = inliers.sum() / max(1, len(inliers))
                else:
                    predicted = ref_points.copy()
                    residual = cur_points - ref_points
                    residual_magnitude = np.linalg.norm(residual, axis=1)
                    inlier_ratio = 0.0
                
                result = FrameHomographyResult(
                    frame_idx=t,
                    homography=H,
                    predicted_points=predicted,
                    residual=residual,
                    residual_magnitude=residual_magnitude,
                    inliers=inliers,
                    inlier_ratio=inlier_ratio
                )
            
            frame_results.append(result)
        
        return HomographyAnalysisResult(
            frame_results=frame_results,
            reference_frame=self.reference_frame,
            ransac_threshold=self.ransac_threshold
        )
    
    def analyze_frame_to_frame(
        self,
        tracks: np.ndarray,
        weights: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None
    ) -> HomographyAnalysisResult:
        """
        Analyze consecutive frames (detect ongoing changes).
        
        Args:
            tracks: [T, N, 2] point trajectories
            weights: [T, N] optional reliability weights
            masks: [T, H, W] optional per-frame binary masks for filtering
            
        Returns:
            HomographyAnalysisResult (relative to previous frame)
        """
        T, N = tracks.shape[:2]
        
        frame_results = []
        
        # First frame has no previous
        frame_results.append(FrameHomographyResult(
            frame_idx=0,
            homography=np.eye(3),
            predicted_points=tracks[0].copy(),
            residual=np.zeros((N, 2)),
            residual_magnitude=np.zeros(N),
            inliers=np.ones(N, dtype=bool),
            inlier_ratio=1.0
        ))
        
        for t in range(1, T):
            prev_points = tracks[t-1]
            cur_points = tracks[t]
            cur_weights = weights[t] if weights is not None else None
            
            # Apply mask filtering if masks provided
            if masks is not None and len(masks) > t:
                mask = masks[t]
                H_img, W_img = mask.shape
                mask_valid = np.ones(N, dtype=bool)
                
                for i, (x, y) in enumerate(cur_points):
                    xi, yi = int(round(x)), int(round(y))
                    # Check if point is within image bounds and inside mask
                    if not (0 <= xi < W_img and 0 <= yi < H_img and mask[yi, xi] > 0):
                        mask_valid[i] = False
                
                # Combine with existing weights
                if cur_weights is not None:
                    cur_weights = cur_weights * mask_valid.astype(np.float64)
                else:
                    cur_weights = mask_valid.astype(np.float64)
            
            # Estimate homography from previous to current
            H, inliers = estimate_homography_robust(
                prev_points, cur_points, cur_weights, self.ransac_threshold
            )
            
            if H is not None:
                predicted = apply_homography(prev_points, H)
                residual = cur_points - predicted
                residual_magnitude = np.linalg.norm(residual, axis=1)
                inlier_ratio = inliers.sum() / max(1, len(inliers))
            else:
                predicted = prev_points.copy()
                residual = cur_points - prev_points
                residual_magnitude = np.linalg.norm(residual, axis=1)
                inlier_ratio = 0.0
            
            frame_results.append(FrameHomographyResult(
                frame_idx=t,
                homography=H,
                predicted_points=predicted,
                residual=residual,
                residual_magnitude=residual_magnitude,
                inliers=inliers,
                inlier_ratio=inlier_ratio
            ))
        
        return HomographyAnalysisResult(
            frame_results=frame_results,
            reference_frame=-1,  # Indicates frame-to-frame
            ransac_threshold=self.ransac_threshold
        )


if __name__ == "__main__":
    print("Homography analysis module loaded successfully")
