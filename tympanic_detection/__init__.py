"""
Tympanic Membrane Deformation Detection System

Automatically detect tympanic membrane deformation during Valsalva maneuver
using CoTracker point tracking and homography-based residual analysis.

Usage:
    from tympanic_detection import TympanicDeformationDetector
    
    detector = TympanicDeformationDetector(grid_size=20, device="cuda")
    result = detector.detect(frames_folder, masks_folder)
    print(result.summary())
"""

from .pipeline import TympanicDeformationDetector, DetectionResult
from .preprocessing import preprocess, PreprocessingResult
from .tracking import run_cotracker, TrackingResult
from .quality_control import TrackQualityController, QualityControlResult
from .homography_analysis import HomographyAnalyzer, HomographyAnalysisResult
from .feature_extraction import FeatureExtractor, FrameFeatures
from .classification import (
    RuleBasedClassifier, 
    AdaptiveThresholdClassifier, 
    ClassificationResult,
    DeformationState
)
from .postprocessing import postprocess, PostProcessedResult
from .visualization import (
    plot_temporal_profile,
    plot_residual_field,
    plot_tracking_quality,
    create_summary_video
)

__version__ = "0.1.0"
__all__ = [
    # Main detector
    "TympanicDeformationDetector",
    "DetectionResult",
    
    # Preprocessing
    "preprocess",
    "PreprocessingResult",
    
    # Tracking
    "run_cotracker",
    "TrackingResult",
    
    # Quality control
    "TrackQualityController",
    "QualityControlResult",
    
    # Homography analysis
    "HomographyAnalyzer",
    "HomographyAnalysisResult",
    
    # Features
    "FeatureExtractor",
    "FrameFeatures",
    
    # Classification
    "RuleBasedClassifier",
    "AdaptiveThresholdClassifier",
    "ClassificationResult",
    "DeformationState",
    
    # Post-processing
    "postprocess",
    "PostProcessedResult",
    
    # Visualization
    "plot_temporal_profile",
    "plot_residual_field",
    "plot_tracking_quality",
    "create_summary_video",
]
