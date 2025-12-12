"""
Module 6: State Classification

Classify each frame into deformation states:
- State 0: Static (no deformation)
- State 1: Deforming (ongoing deformation)
- State 2: Peak (deformation plateau)
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import IntEnum

from .feature_extraction import FrameFeatures


class DeformationState(IntEnum):
    """Deformation state enumeration."""
    STATIC = 0
    DEFORMING = 1
    PEAK = 2


@dataclass
class ClassificationResult:
    """Classification result for the video."""
    states: np.ndarray  # [T] state labels
    confidences: np.ndarray  # [T] confidence scores
    deformation_start: Optional[int]  # Frame where deformation starts
    peak_start: Optional[int]  # Frame where peak starts
    peak_end: Optional[int]  # Frame where peak ends


class RuleBasedClassifier:
    """
    Rule-based state classifier using thresholds.
    """
    
    def __init__(
        self,
        static_threshold: float = 0.5,
        deformation_threshold: float = 1.0,
        trend_window: int = 3,
        inlier_ratio_threshold: float = 0.9
    ):
        """
        Args:
            static_threshold: Residual below this is considered static
            deformation_threshold: Residual above this indicates deformation
            trend_window: Window size for trend detection
            inlier_ratio_threshold: Inlier ratio below this indicates deformation
        """
        self.static_threshold = static_threshold
        self.deformation_threshold = deformation_threshold
        self.trend_window = trend_window
        self.inlier_ratio_threshold = inlier_ratio_threshold
    
    def classify(self, features: List[FrameFeatures]) -> ClassificationResult:
        """
        Classify frames into states.
        """
        T = len(features)
        states = np.zeros(T, dtype=np.int32)
        confidences = np.zeros(T)
        
        # Extract key features
        residuals = np.array([f.median_residual for f in features])
        inlier_ratios = np.array([f.inlier_ratio for f in features])
        center_vs_edge = np.array([f.center_vs_edge_ratio for f in features])
        
        # Normalize residuals relative to baseline (first few frames)
        baseline = np.median(residuals[:5]) if T >= 5 else residuals[0]
        relative_residuals = residuals - baseline
        
        for t in range(T):
            residual = residuals[t]
            relative = relative_residuals[t]
            inlier = inlier_ratios[t]
            
            # Compute trend (is residual increasing?)
            if t >= self.trend_window:
                trend = relative_residuals[t] - relative_residuals[t - self.trend_window]
            else:
                trend = 0
            
            # Classification logic
            if residual < self.static_threshold and inlier > self.inlier_ratio_threshold:
                # Low residual, high inlier ratio → static
                states[t] = DeformationState.STATIC
                confidences[t] = 1.0 - residual / self.static_threshold
            
            elif residual >= self.deformation_threshold or inlier < self.inlier_ratio_threshold:
                # High residual or low inlier ratio
                if trend > 0.1:
                    # Increasing trend → still deforming
                    states[t] = DeformationState.DEFORMING
                    confidences[t] = min(1.0, residual / self.deformation_threshold)
                else:
                    # Stable high residual → peak
                    states[t] = DeformationState.PEAK
                    confidences[t] = min(1.0, residual / self.deformation_threshold)
            
            else:
                # Intermediate region
                if trend > 0.05:
                    states[t] = DeformationState.DEFORMING
                else:
                    states[t] = DeformationState.STATIC
                confidences[t] = 0.5
        
        # Find key transitions
        deformation_start = self._find_transition(states, 
                                                   DeformationState.STATIC, 
                                                   DeformationState.DEFORMING)
        peak_start = self._find_transition(states,
                                           DeformationState.DEFORMING,
                                           DeformationState.PEAK)
        peak_end = self._find_last_state(states, DeformationState.PEAK)
        
        return ClassificationResult(
            states=states,
            confidences=confidences,
            deformation_start=deformation_start,
            peak_start=peak_start,
            peak_end=peak_end
        )
    
    def _find_transition(self, states: np.ndarray, 
                         from_state: int, to_state: int) -> Optional[int]:
        """Find first frame where transition occurs."""
        for t in range(1, len(states)):
            if states[t-1] == from_state and states[t] == to_state:
                return t
        return None
    
    def _find_last_state(self, states: np.ndarray, state: int) -> Optional[int]:
        """Find last frame of a given state."""
        indices = np.where(states == state)[0]
        if len(indices) > 0:
            return int(indices[-1])
        return None


class AdaptiveThresholdClassifier:
    """
    Classifier with adaptive thresholds based on video statistics.
    """
    
    def __init__(
        self,
        static_percentile: float = 25,
        deformation_percentile: float = 75,
        trend_window: int = 5
    ):
        """
        Args:
            static_percentile: Percentile for static threshold
            deformation_percentile: Percentile for deformation threshold
            trend_window: Window for trend analysis
        """
        self.static_percentile = static_percentile
        self.deformation_percentile = deformation_percentile
        self.trend_window = trend_window
    
    def classify(self, features: List[FrameFeatures]) -> ClassificationResult:
        """
        Classify with adaptive thresholds.
        """
        T = len(features)
        residuals = np.array([f.median_residual for f in features])
        inlier_ratios = np.array([f.inlier_ratio for f in features])
        
        # Adaptive thresholds
        static_threshold = np.percentile(residuals, self.static_percentile)
        deformation_threshold = np.percentile(residuals, self.deformation_percentile)
        
        # Use rule-based classifier with adaptive thresholds
        classifier = RuleBasedClassifier(
            static_threshold=static_threshold,
            deformation_threshold=deformation_threshold,
            trend_window=self.trend_window
        )
        
        return classifier.classify(features)


if __name__ == "__main__":
    print("Classification module loaded successfully")
