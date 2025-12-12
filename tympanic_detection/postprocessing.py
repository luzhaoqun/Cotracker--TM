"""
Module 7: Post-processing

Smooth and refine classification results.
"""

import numpy as np
from scipy import ndimage
from typing import Optional
from dataclasses import dataclass

from .classification import ClassificationResult, DeformationState


@dataclass
class PostProcessedResult:
    """Post-processed classification result."""
    raw_states: np.ndarray  # Original states
    smoothed_states: np.ndarray  # Smoothed states
    deformation_start: Optional[int]
    peak_start: Optional[int]
    peak_end: Optional[int]
    confidence: float  # Overall detection confidence


def median_filter_states(states: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Apply median filter to state sequence.
    """
    return ndimage.median_filter(states, size=kernel_size, mode='nearest')


def enforce_state_transitions(states: np.ndarray) -> np.ndarray:
    """
    Enforce valid state transitions:
    - Static → Deforming → Peak is allowed
    - Peak → Deforming → Static is allowed
    - Direct Static ↔ Peak transitions are corrected
    """
    result = states.copy()
    T = len(states)
    
    for t in range(1, T):
        prev, curr = result[t-1], result[t]
        
        # Invalid: Static (0) → Peak (2) directly
        if prev == DeformationState.STATIC and curr == DeformationState.PEAK:
            result[t] = DeformationState.DEFORMING
        
        # Invalid: Peak (2) → Static (0) directly
        if prev == DeformationState.PEAK and curr == DeformationState.STATIC:
            result[t] = DeformationState.DEFORMING
    
    return result


def enforce_minimum_duration(
    states: np.ndarray, 
    min_duration: int = 3
) -> np.ndarray:
    """
    Enforce minimum duration for each state segment.
    Short segments are merged with neighbors.
    """
    result = states.copy()
    T = len(states)
    
    # Find segments
    segments = []
    start = 0
    for t in range(1, T):
        if states[t] != states[t-1]:
            segments.append((start, t, states[start]))
            start = t
    segments.append((start, T, states[start]))
    
    # Merge short segments
    for i, (start, end, state) in enumerate(segments):
        duration = end - start
        if duration < min_duration:
            # Merge with longer neighbor
            if i > 0:
                prev_start, prev_end, prev_state = segments[i-1]
                result[start:end] = prev_state
            elif i < len(segments) - 1:
                next_start, next_end, next_state = segments[i+1]
                result[start:end] = next_state
    
    return result


def postprocess(
    classification_result: ClassificationResult,
    median_kernel: int = 5,
    min_state_duration: int = 3
) -> PostProcessedResult:
    """
    Complete post-processing pipeline.
    
    Args:
        classification_result: Raw classification result
        median_kernel: Kernel size for median filter
        min_state_duration: Minimum frames for each state
        
    Returns:
        PostProcessedResult
    """
    raw_states = classification_result.states
    
    # Step 1: Median filter
    smoothed = median_filter_states(raw_states, median_kernel)
    
    # Step 2: Enforce valid transitions
    smoothed = enforce_state_transitions(smoothed)
    
    # Step 3: Enforce minimum duration
    smoothed = enforce_minimum_duration(smoothed, min_state_duration)
    
    # Re-find transitions after smoothing
    deformation_start = None
    peak_start = None
    peak_end = None
    
    for t in range(1, len(smoothed)):
        if smoothed[t-1] == DeformationState.STATIC and smoothed[t] == DeformationState.DEFORMING:
            if deformation_start is None:
                deformation_start = t
        if smoothed[t-1] == DeformationState.DEFORMING and smoothed[t] == DeformationState.PEAK:
            if peak_start is None:
                peak_start = t
    
    # Find peak end
    peak_indices = np.where(smoothed == DeformationState.PEAK)[0]
    if len(peak_indices) > 0:
        peak_end = int(peak_indices[-1])
    
    # Calculate overall confidence
    # Based on consistency between raw and smoothed, and state progression
    changes = (raw_states != smoothed).sum()
    consistency = 1.0 - changes / len(raw_states)
    
    has_deformation = (smoothed == DeformationState.DEFORMING).any()
    has_peak = (smoothed == DeformationState.PEAK).any()
    
    if has_deformation or has_peak:
        confidence = consistency * 0.7 + 0.3
    else:
        confidence = consistency * 0.5  # Lower confidence if no deformation detected
    
    return PostProcessedResult(
        raw_states=raw_states,
        smoothed_states=smoothed,
        deformation_start=deformation_start,
        peak_start=peak_start,
        peak_end=peak_end,
        confidence=confidence
    )


if __name__ == "__main__":
    print("Post-processing module loaded successfully")
