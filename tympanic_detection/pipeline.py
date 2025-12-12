"""
Main Pipeline

Complete tympanic membrane deformation detection pipeline.
"""

import numpy as np
from pathlib import Path
from typing import Union, Optional
from dataclasses import dataclass

from .preprocessing import preprocess, PreprocessingResult
from .tracking import run_cotracker, TrackingResult
from .quality_control import TrackQualityController, QualityControlResult, estimate_tracking_noise
from .homography_analysis import HomographyAnalyzer, HomographyAnalysisResult
from .feature_extraction import FeatureExtractor, FrameFeatures
from .classification import RuleBasedClassifier, AdaptiveThresholdClassifier, ClassificationResult
from .postprocessing import postprocess, PostProcessedResult


@dataclass
class DetectionResult:
    """Complete detection result."""
    # Preprocessing
    preprocessing: PreprocessingResult
    
    # Tracking
    tracking: TrackingResult
    
    # Quality control
    quality_control: QualityControlResult
    
    # Homography analysis
    homography: HomographyAnalysisResult
    
    # Features
    features: list  # List[FrameFeatures]
    
    # Classification
    classification: ClassificationResult
    
    # Post-processing
    final_result: PostProcessedResult
    
    def summary(self) -> str:
        """Return human-readable summary."""
        r = self.final_result
        lines = [
            "=== Tympanic Membrane Deformation Detection ===",
            f"Total frames: {len(r.smoothed_states)}",
            f"Detection confidence: {r.confidence:.2%}",
            "",
            "State distribution:",
            f"  - Static frames: {(r.smoothed_states == 0).sum()}",
            f"  - Deforming frames: {(r.smoothed_states == 1).sum()}",
            f"  - Peak frames: {(r.smoothed_states == 2).sum()}",
            "",
            "Key events:",
        ]
        
        if r.deformation_start is not None:
            lines.append(f"  - Deformation starts: frame {r.deformation_start}")
        else:
            lines.append("  - Deformation starts: not detected")
        
        if r.peak_start is not None:
            lines.append(f"  - Peak reached: frame {r.peak_start}")
        else:
            lines.append("  - Peak reached: not detected")
        
        if r.peak_end is not None:
            lines.append(f"  - Peak ends: frame {r.peak_end}")
        
        return "\n".join(lines)


class TympanicDeformationDetector:
    """
    Complete pipeline for tympanic membrane deformation detection.
    """
    
    def __init__(
        self,
        grid_size: int = 20,
        device: str = "cuda",
        ransac_threshold: Optional[float] = None,
        use_adaptive_classifier: bool = True,
        use_frame_to_frame: bool = True,
        verbose: bool = True
    ):
        """
        Args:
            grid_size: Size of sampling grid
            device: Device for CoTracker ('cuda' or 'cpu')
            ransac_threshold: RANSAC threshold (None for adaptive)
            use_adaptive_classifier: Use adaptive threshold classifier
            use_frame_to_frame: Use frame-to-frame analysis (True) or reference frame (False)
            verbose: Print progress messages
        """
        self.grid_size = grid_size
        self.device = device
        self.ransac_threshold = ransac_threshold
        self.use_adaptive_classifier = use_adaptive_classifier
        self.use_frame_to_frame = use_frame_to_frame
        self.verbose = verbose
    
    def detect(
        self,
        frames_folder: Union[str, Path],
        masks_folder: Union[str, Path],
        frame_pattern: str = "*.png",
        mask_pattern: str = "*.png",
        max_frames: Optional[int] = None
    ) -> DetectionResult:
        """
        Run complete detection pipeline.
        
        Args:
            frames_folder: Path to video frames
            masks_folder: Path to mask images
            frame_pattern: Pattern for frame files
            mask_pattern: Pattern for mask files
            max_frames: Maximum frames to process
            
        Returns:
            DetectionResult with all intermediate and final results
        """
        # Step 1: Preprocessing
        self._log("Step 1/7: Preprocessing...")
        prep_result = preprocess(
            frames_folder, masks_folder,
            frame_pattern, mask_pattern,
            self.grid_size, max_frames
        )
        
        # Step 2: CoTracker tracking
        self._log("Step 2/7: Running CoTracker...")
        tracking_result = run_cotracker(
            prep_result.frames,
            prep_result.query_points,
            device=self.device
        )
        
        # Step 3: Quality control
        self._log("Step 3/7: Quality control...")
        qc = TrackQualityController(
            neighbor_indices=prep_result.neighbor_indices
        )
        qc_result = qc.process(tracking_result.tracks, tracking_result.visibility)
        
        # Estimate noise for RANSAC threshold
        if self.ransac_threshold is None:
            noise = estimate_tracking_noise(tracking_result.tracks, tracking_result.visibility)
            ransac_threshold = max(2.0, noise * 2.5)
            self._log(f"  Estimated tracking noise: {noise:.2f}px, RANSAC threshold: {ransac_threshold:.2f}px")
        else:
            ransac_threshold = self.ransac_threshold
        
        # Step 4: Homography analysis
        self._log("Step 4/7: Homography analysis...")
        analyzer = HomographyAnalyzer(
            ransac_threshold=ransac_threshold,
            use_two_stage=True
        )
        if self.use_frame_to_frame:
            homography_result = analyzer.analyze_frame_to_frame(
                qc_result.repaired_tracks,
                qc_result.weights,
                prep_result.masks  # Pass masks for per-frame filtering
            )
        else:
            homography_result = analyzer.analyze(
                qc_result.repaired_tracks,
                qc_result.weights
            )
        
        # Step 5: Feature extraction
        self._log("Step 5/7: Feature extraction...")
        extractor = FeatureExtractor(
            grid_shape=prep_result.grid_shape,
            valid_mask=prep_result.point_indices_in_mask
        )
        features = extractor.extract_all_features(homography_result)
        
        # Step 6: Classification
        self._log("Step 6/7: Classification...")
        if self.use_adaptive_classifier:
            classifier = AdaptiveThresholdClassifier()
        else:
            classifier = RuleBasedClassifier()
        classification_result = classifier.classify(features)
        
        # Step 7: Post-processing
        self._log("Step 7/7: Post-processing...")
        final_result = postprocess(classification_result)
        
        self._log("Detection complete!")
        
        return DetectionResult(
            preprocessing=prep_result,
            tracking=tracking_result,
            quality_control=qc_result,
            homography=homography_result,
            features=features,
            classification=classification_result,
            final_result=final_result
        )
    
    def detect_from_arrays(
        self,
        frames: np.ndarray,
        masks: np.ndarray
    ) -> DetectionResult:
        """
        Run detection on pre-loaded arrays.
        
        Args:
            frames: [T, H, W, C] video frames
            masks: [T, H, W] binary masks
            
        Returns:
            DetectionResult
        """
        from .preprocessing import (
            sample_grid_points_in_mask, 
            build_grid_neighbors,
            PreprocessingResult
        )
        
        # Create preprocessing result from arrays
        self._log("Step 1/7: Preprocessing from arrays...")
        grid_points, grid_shape, valid_mask = sample_grid_points_in_mask(
            masks[0], self.grid_size
        )
        
        query_points = np.zeros((len(grid_points), 3), dtype=np.float32)
        query_points[:, 0] = 0
        query_points[:, 1:] = grid_points
        
        neighbors = build_grid_neighbors(grid_shape, valid_mask)
        
        prep_result = PreprocessingResult(
            frames=frames,
            masks=masks,
            query_points=query_points,
            grid_shape=grid_shape,
            point_indices_in_mask=valid_mask,
            neighbor_indices=neighbors
        )
        
        # Continue with rest of pipeline
        self._log("Step 2/7: Running CoTracker...")
        tracking_result = run_cotracker(
            frames, query_points, device=self.device
        )
        
        # Steps 3-7 same as before...
        self._log("Step 3/7: Quality control...")
        qc = TrackQualityController(neighbor_indices=neighbors)
        qc_result = qc.process(tracking_result.tracks, tracking_result.visibility)
        
        if self.ransac_threshold is None:
            noise = estimate_tracking_noise(tracking_result.tracks, tracking_result.visibility)
            ransac_threshold = max(2.0, noise * 2.5)
        else:
            ransac_threshold = self.ransac_threshold
        
        self._log("Step 4/7: Homography analysis...")
        analyzer = HomographyAnalyzer(ransac_threshold=ransac_threshold, use_two_stage=True)
        if self.use_frame_to_frame:
            homography_result = analyzer.analyze_frame_to_frame(
                qc_result.repaired_tracks, 
                qc_result.weights,
                masks  # Pass masks for per-frame filtering
            )
        else:
            homography_result = analyzer.analyze(qc_result.repaired_tracks, qc_result.weights)
        
        self._log("Step 5/7: Feature extraction...")
        extractor = FeatureExtractor(grid_shape=grid_shape, valid_mask=valid_mask)
        features = extractor.extract_all_features(homography_result)
        
        self._log("Step 6/7: Classification...")
        classifier = AdaptiveThresholdClassifier() if self.use_adaptive_classifier else RuleBasedClassifier()
        classification_result = classifier.classify(features)
        
        self._log("Step 7/7: Post-processing...")
        final_result = postprocess(classification_result)
        
        self._log("Detection complete!")
        
        return DetectionResult(
            preprocessing=prep_result,
            tracking=tracking_result,
            quality_control=qc_result,
            homography=homography_result,
            features=features,
            classification=classification_result,
            final_result=final_result
        )
    
    def _log(self, message: str):
        """Print log message if verbose."""
        if self.verbose:
            print(message)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Tympanic membrane deformation detection")
    parser.add_argument("--frames", type=str, required=True, help="Path to frames folder")
    parser.add_argument("--masks", type=str, required=True, help="Path to masks folder")
    parser.add_argument("--grid_size", type=int, default=20, help="Grid size")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--max_frames", type=int, default=None, help="Max frames to process")
    args = parser.parse_args()
    
    detector = TympanicDeformationDetector(
        grid_size=args.grid_size,
        device=args.device
    )
    
    result = detector.detect(
        args.frames,
        args.masks,
        max_frames=args.max_frames
    )
    
    print(result.summary())
