"""
Point Distance Analysis

Detect tympanic membrane deformation by measuring changes in 
distances between neighboring tracking points.

This approach is camera-motion invariant because relative distances
are preserved under perspective transformations.
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from dataclasses import dataclass

# Add project root to path
_project_root = Path(__file__).parent.parent.absolute()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


@dataclass
class DistanceAnalysisResult:
    """Result of point distance analysis."""
    tracks: np.ndarray  # [T, N, 2] point tracks
    reference_distances: np.ndarray  # [N] average neighbor distance at frame 0
    frame_distances: np.ndarray  # [T, N] average neighbor distance per frame
    distance_ratios: np.ndarray  # [T, N] ratio relative to reference
    neighbor_indices: List[List[int]]  # neighbor list for each point
    

def build_grid_neighbors_with_radius(
    grid_shape: Tuple[int, int],
    valid_mask: np.ndarray,
    radius: int = 1
) -> List[List[int]]:
    """
    Build neighbor indices for grid points with configurable radius.
    
    Args:
        grid_shape: (rows, cols) shape of the grid
        valid_mask: [N] boolean mask of valid points
        radius: Neighborhood radius (1 = 8 neighbors, 2 = 24 neighbors, etc.)
        
    Returns:
        neighbors: List of neighbor index lists for each point
    """
    rows, cols = grid_shape
    n_points = rows * cols
    
    neighbors = []
    for idx in range(n_points):
        r, c = idx // cols, idx % cols
        neighbor_list = []
        
        # Search within radius
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr == 0 and dc == 0:
                    continue  # Skip self
                
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    neighbor_idx = nr * cols + nc
                    # Only include if both current and neighbor are valid
                    if valid_mask[idx] and valid_mask[neighbor_idx]:
                        neighbor_list.append(neighbor_idx)
        
        neighbors.append(neighbor_list)
    
    return neighbors


def compute_neighbor_distances(
    points: np.ndarray,
    neighbor_indices: List[List[int]]
) -> np.ndarray:
    """
    Compute average distance to neighbors for each point.
    
    Args:
        points: [N, 2] point coordinates
        neighbor_indices: List of neighbor lists
        
    Returns:
        avg_distances: [N] average distance to neighbors
    """
    N = len(points)
    avg_distances = np.zeros(N)
    
    for i in range(N):
        neighbors = neighbor_indices[i]
        if len(neighbors) == 0:
            avg_distances[i] = np.nan
            continue
        
        # Compute distance to each neighbor
        distances = []
        for j in neighbors:
            dist = np.linalg.norm(points[i] - points[j])
            distances.append(dist)
        
        avg_distances[i] = np.mean(distances)
    
    return avg_distances


def analyze_distance_changes(
    tracks: np.ndarray,
    neighbor_indices: List[List[int]],
    reference_frame: int = 0
) -> DistanceAnalysisResult:
    """
    Analyze point distance changes across all frames.
    
    Args:
        tracks: [T, N, 2] point trajectories
        neighbor_indices: Neighbor lists from build_grid_neighbors_with_radius
        reference_frame: Frame to use as reference (default: 0)
        
    Returns:
        DistanceAnalysisResult
    """
    T, N = tracks.shape[:2]
    
    # Compute reference distances
    ref_points = tracks[reference_frame]
    reference_distances = compute_neighbor_distances(ref_points, neighbor_indices)
    
    # Compute distances for all frames
    frame_distances = np.zeros((T, N))
    for t in range(T):
        frame_distances[t] = compute_neighbor_distances(tracks[t], neighbor_indices)
    
    # Compute distance ratios
    # Avoid division by zero
    safe_ref = reference_distances.copy()
    safe_ref[safe_ref < 0.1] = 0.1
    distance_ratios = frame_distances / safe_ref[None, :]
    
    return DistanceAnalysisResult(
        tracks=tracks,
        reference_distances=reference_distances,
        frame_distances=frame_distances,
        distance_ratios=distance_ratios,
        neighbor_indices=neighbor_indices
    )


def extract_distance_features(result: DistanceAnalysisResult) -> dict:
    """
    Extract summary features from distance analysis.
    
    Returns:
        Dictionary of per-frame features
    """
    ratios = result.distance_ratios  # [T, N]
    T = ratios.shape[0]
    
    features = {
        'median_ratio': np.zeros(T),
        'mean_ratio': np.zeros(T),
        'max_ratio': np.zeros(T),
        'min_ratio': np.zeros(T),
        'std_ratio': np.zeros(T),
        'expanding_ratio': np.zeros(T),  # Fraction of points with ratio > 1.05
        'contracting_ratio': np.zeros(T),  # Fraction of points with ratio < 0.95
    }
    
    for t in range(T):
        valid = ~np.isnan(ratios[t])
        if valid.sum() == 0:
            continue
        
        r = ratios[t, valid]
        features['median_ratio'][t] = np.median(r)
        features['mean_ratio'][t] = np.mean(r)
        features['max_ratio'][t] = np.max(r)
        features['min_ratio'][t] = np.min(r)
        features['std_ratio'][t] = np.std(r)
        features['expanding_ratio'][t] = (r > 1.05).mean()
        features['contracting_ratio'][t] = (r < 0.95).mean()
    
    return features


# ===== Feature 2: Divergence (位移场散度) =====

def compute_divergence(
    tracks: np.ndarray,
    grid_shape: Tuple[int, int],
    reference_frame: int = 0
) -> np.ndarray:
    """
    Compute divergence of displacement field.
    
    Divergence > 0 → outward expansion (bulging)
    Divergence < 0 → inward contraction
    Divergence ≈ 0 → no deformation
    
    Args:
        tracks: [T, N, 2] point trajectories
        grid_shape: (rows, cols) shape of the grid
        reference_frame: Reference frame index
        
    Returns:
        divergence: [T] divergence value per frame
    """
    T, N = tracks.shape[:2]
    rows, cols = grid_shape
    
    ref_points = tracks[reference_frame]  # [N, 2]
    divergence = np.zeros(T)
    
    for t in range(T):
        # Displacement from reference
        displacement = tracks[t] - ref_points  # [N, 2]
        
        try:
            # Reshape to grid
            dx = displacement[:, 0].reshape(rows, cols)
            dy = displacement[:, 1].reshape(rows, cols)
            
            # Compute gradients (partial derivatives)
            # du/dx: gradient of x-displacement along x-axis
            du_dx = np.gradient(dx, axis=1)
            # dv/dy: gradient of y-displacement along y-axis
            dv_dy = np.gradient(dy, axis=0)
            
            # Divergence = du/dx + dv/dy
            div_field = du_dx + dv_dy
            
            # Take mean divergence (excluding edges which have gradient artifacts)
            inner = div_field[1:-1, 1:-1]
            divergence[t] = np.mean(inner)
            
        except ValueError:
            divergence[t] = 0.0
    
    return divergence


# ===== Feature 3: Triangle Mesh Area (三角网格面积) =====

def compute_triangle_areas(
    tracks: np.ndarray,
    valid_mask: np.ndarray,
    reference_frame: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute triangle mesh area changes using Delaunay triangulation.
    
    Args:
        tracks: [T, N, 2] point trajectories
        valid_mask: [N] boolean mask of valid points
        reference_frame: Reference frame index
        
    Returns:
        area_ratios: [T] mean area ratio per frame
        max_area_ratios: [T] max area ratio per frame
        triangles: Delaunay triangulation (for visualization)
    """
    from scipy.spatial import Delaunay
    
    T = tracks.shape[0]
    
    # Get valid points at reference frame
    ref_points = tracks[reference_frame][valid_mask]
    
    # Create Delaunay triangulation on reference points
    try:
        tri = Delaunay(ref_points)
    except Exception as e:
        print(f"Delaunay triangulation failed: {e}")
        return np.ones(T), np.ones(T), None
    
    # Compute reference triangle areas
    def compute_areas(points, simplices):
        """Compute area of each triangle."""
        areas = np.zeros(len(simplices))
        for i, simplex in enumerate(simplices):
            # Get vertices
            p0, p1, p2 = points[simplex]
            # Area = 0.5 * |cross product|
            v1 = p1 - p0
            v2 = p2 - p0
            areas[i] = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
        return areas
    
    ref_areas = compute_areas(ref_points, tri.simplices)
    
    # Compute area ratios for each frame
    area_ratios = np.zeros(T)
    max_area_ratios = np.zeros(T)
    
    for t in range(T):
        cur_points = tracks[t][valid_mask]
        cur_areas = compute_areas(cur_points, tri.simplices)
        
        # Avoid division by zero
        safe_ref = ref_areas.copy()
        safe_ref[safe_ref < 0.1] = 0.1
        
        ratios = cur_areas / safe_ref
        area_ratios[t] = np.mean(ratios)
        max_area_ratios[t] = np.max(ratios)
    
    return area_ratios, max_area_ratios, tri


def extract_all_features(
    tracks: np.ndarray,
    neighbor_indices: List[List[int]],
    grid_shape: Tuple[int, int],
    valid_mask: np.ndarray,
    reference_frame: int = 0
) -> dict:
    """
    Extract all three types of features.
    
    Returns:
        Dictionary containing all features
    """
    # Feature 1: Distance ratios
    result = analyze_distance_changes(tracks, neighbor_indices, reference_frame)
    features = extract_distance_features(result)
    
    # Feature 2: Divergence
    divergence = compute_divergence(tracks, grid_shape, reference_frame)
    features['divergence'] = divergence
    
    # Feature 3: Triangle areas
    area_ratios, max_area_ratios, triangles = compute_triangle_areas(
        tracks, valid_mask, reference_frame
    )
    features['area_ratio'] = area_ratios
    features['max_area_ratio'] = max_area_ratios
    
    return features, result, triangles


def build_feature_matrix(features: dict) -> np.ndarray:
    """
    Build feature matrix from features dictionary.
    
    Args:
        features: Dictionary of 1D feature arrays
        
    Returns:
        X: [T, n_features] feature matrix
    """
    # Select features to include
    feature_names = [
        'median_ratio', 'mean_ratio', 'max_ratio', 'min_ratio', 'std_ratio',
        'expanding_ratio', 'contracting_ratio',
        'divergence',
        'area_ratio', 'max_area_ratio'
    ]
    
    # Stack features
    feature_list = []
    for name in feature_names:
        if name in features:
            feature_list.append(features[name])
        else:
            # print(f"Warning: Feature {name} not found")
            pass
    
    if not feature_list:
        raise ValueError("No features found to build matrix")
        
    X = np.stack(feature_list, axis=1)
    
    # Handle NaNs
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    return X


# ===== Change Point Detection (变化点检测) =====

def detect_change_points_simple(
    signal: np.ndarray,
    n_changes: int = 2,
    min_segment_length: int = 3
) -> List[int]:
    """
    Simple change point detection using cumulative sum (CUSUM) method.
    
    No external dependencies required.
    
    Args:
        signal: 1D signal to analyze
        n_changes: Expected number of change points
        min_segment_length: Minimum length between change points
        
    Returns:
        List of change point indices
    """
    T = len(signal)
    if T < 2 * min_segment_length:
        return []
    
    # Normalize signal
    signal_norm = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    
    # Compute cost of splitting at each point
    costs = np.zeros(T)
    
    for t in range(min_segment_length, T - min_segment_length):
        left = signal_norm[:t]
        right = signal_norm[t:]
        
        # Cost = sum of squared residuals from segment means
        cost_left = np.sum((left - np.mean(left)) ** 2)
        cost_right = np.sum((right - np.mean(right)) ** 2)
        costs[t] = cost_left + cost_right
    
    # Find optimal split points (minimum total cost)
    change_points = []
    
    for _ in range(n_changes):
        # Mask out already found points and their neighborhoods
        masked_costs = costs.copy()
        for cp in change_points:
            start = max(0, cp - min_segment_length)
            end = min(T, cp + min_segment_length)
            masked_costs[start:end] = np.inf
        
        # Find minimum cost point
        valid_range = masked_costs[min_segment_length:-min_segment_length]
        if len(valid_range) == 0:
            break
        
        best_idx = np.argmin(masked_costs)
        if masked_costs[best_idx] < np.inf:
            change_points.append(best_idx)
    
    return sorted(change_points)


def detect_change_points_ruptures(
    signal: np.ndarray,
    n_changes: int = 2,
    model: str = "rbf",
    min_size: int = 3
) -> List[int]:
    """
    Change point detection using ruptures library.
    
    Args:
        signal: 1D signal to analyze
        n_changes: Expected number of change points
        model: Cost model ('l2', 'rbf', 'linear', 'normal')
        min_size: Minimum segment size
        
    Returns:
        List of change point indices
    """
    try:
        import ruptures as rpt
    except ImportError:
        print("ruptures not installed. Install with: pip install ruptures")
        print("Falling back to simple method...")
        return detect_change_points_simple(signal, n_changes, min_size)
    
    # Reshape for ruptures
    signal_2d = signal.reshape(-1, 1)
    
    # Use Pelt algorithm (optimal for detecting multiple change points)
    algo = rpt.Pelt(model=model, min_size=min_size).fit(signal_2d)
    
    # Predict change points
    try:
        # n_bkps = number of breakpoints (change points)
        result = algo.predict(n_bkps=n_changes)
        # Remove last element (which is len(signal))
        change_points = [x for x in result if x < len(signal)]
    except Exception as e:
        print(f"ruptures failed: {e}")
        change_points = detect_change_points_simple(signal, n_changes, min_size)
    
    return change_points


def detect_change_points_derivative(
    signal: np.ndarray,
    threshold_percentile: float = 90,
    smooth_window: int = 3
) -> List[int]:
    """
    Detect change points using signal derivative.
    
    Args:
        signal: 1D signal to analyze  
        threshold_percentile: Percentile for significant changes
        smooth_window: Smoothing window size
        
    Returns:
        List of change point indices
    """
    # Smooth the signal
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        smoothed = np.convolve(signal, kernel, mode='same')
    else:
        smoothed = signal
    
    # Compute derivative
    derivative = np.abs(np.gradient(smoothed))
    
    # Find peaks in derivative (significant changes)
    threshold = np.percentile(derivative, threshold_percentile)
    peaks = np.where(derivative > threshold)[0]
    
    # Cluster nearby peaks
    if len(peaks) == 0:
        return []
    
    change_points = [peaks[0]]
    min_gap = max(3, len(signal) // 10)
    
    for p in peaks[1:]:
        if p - change_points[-1] >= min_gap:
            change_points.append(p)
    
    return change_points


def classify_with_change_points(
    features: dict,
    n_expected_changes: int = 2,
    method: str = "combined"
) -> Tuple[np.ndarray, List[int], dict]:
    """
    Classify deformation states using change point detection.
    
    Expected pattern: Static → Deforming → Peak
    So we expect 2 change points.
    
    Args:
        features: Dictionary of per-frame features
        n_expected_changes: Expected number of change points
        method: Detection method ('simple', 'ruptures', 'derivative', 'combined')
        
    Returns:
        states: [T] state labels (0=static, 1=deforming, 2=peak)
        change_points: List of detected change point indices
        detection_info: Additional information about detection
    """
    T = len(features['median_ratio'])
    
    # Combine multiple features into a single signal
    # Higher value = more deformation
    combined_signal = (
        features['median_ratio'] - 1.0 +  # Distance ratio (centered at 0)
        features['area_ratio'] - 1.0 +     # Area ratio (centered at 0)
        features['divergence'] * 0.1       # Divergence (scaled)
    )
    
    # Detect change points based on method
    if method == "simple":
        change_points = detect_change_points_simple(combined_signal, n_expected_changes)
    elif method == "ruptures":
        change_points = detect_change_points_ruptures(combined_signal, n_expected_changes)
    elif method == "derivative":
        change_points = detect_change_points_derivative(combined_signal)
    elif method == "combined":
        # Use multiple methods and find consensus
        cp1 = detect_change_points_simple(combined_signal, n_expected_changes)
        cp2 = detect_change_points_derivative(combined_signal)
        
        # Try ruptures if available
        try:
            import ruptures
            cp3 = detect_change_points_ruptures(combined_signal, n_expected_changes)
        except ImportError:
            cp3 = []
        
        # Combine: find points that appear in at least 2 methods
        all_cps = cp1 + cp2 + cp3
        if len(all_cps) == 0:
            change_points = []
        else:
            # Cluster nearby points
            all_cps = sorted(all_cps)
            clusters = []
            current_cluster = [all_cps[0]]
            
            for cp in all_cps[1:]:
                if cp - current_cluster[-1] < max(3, T // 10):
                    current_cluster.append(cp)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [cp]
            clusters.append(current_cluster)
            
            # Take median of each cluster
            change_points = [int(np.median(c)) for c in clusters if len(c) >= 1]
            change_points = sorted(change_points)[:n_expected_changes]
    else:
        change_points = detect_change_points_simple(combined_signal, n_expected_changes)
    
    # Classify frames based on change points
    states = np.zeros(T, dtype=np.int32)
    
    if len(change_points) == 0:
        # No change detected - check if there's any deformation
        if np.max(combined_signal) > 0.05:
            states[:] = 1  # All deforming
        else:
            states[:] = 0  # All static
    elif len(change_points) == 1:
        # One change point
        cp = change_points[0]
        # Determine direction
        before_mean = np.mean(combined_signal[:cp])
        after_mean = np.mean(combined_signal[cp:])
        
        if after_mean > before_mean:
            # Increasing: static → deforming/peak
            states[:cp] = 0
            states[cp:] = 1
        else:
            # Decreasing: peak → recovering
            states[:cp] = 2
            states[cp:] = 1
    else:
        # Two or more change points: static → deforming → peak
        cp1, cp2 = change_points[0], change_points[1]
        states[:cp1] = 0        # Static
        states[cp1:cp2] = 1     # Deforming
        states[cp2:] = 2        # Peak
    
    detection_info = {
        'combined_signal': combined_signal,
        'method': method,
        'n_changes_found': len(change_points)
    }
    
    return states, change_points, detection_info


def plot_change_point_detection(
    features: dict,
    states: np.ndarray,
    change_points: List[int],
    detection_info: dict,
    save_path: Optional[str] = None
):
    """
    Visualize change point detection results.
    """
    T = len(states)
    frames = np.arange(T)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # 1. Combined signal with change points
    ax = axes[0]
    signal = detection_info['combined_signal']
    ax.plot(frames, signal, 'b-', linewidth=2, label='Combined Signal')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    for i, cp in enumerate(change_points):
        ax.axvline(cp, color='red', linewidth=2, linestyle='-', 
                   label=f'Change Point {i+1}' if i == 0 else None)
        ax.annotate(f'CP{i+1}: frame {cp}', (cp, ax.get_ylim()[1]), 
                   xytext=(5, -5), textcoords='offset points', fontsize=10)
    
    ax.set_ylabel('Deformation Signal')
    ax.set_title(f'Change Point Detection (Method: {detection_info["method"]})')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 2. Individual features
    ax = axes[1]
    ax.plot(frames, features['median_ratio'], 'b-', linewidth=2, label='Distance Ratio')
    ax.plot(frames, features['area_ratio'], 'm-', linewidth=2, label='Area Ratio')
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    for cp in change_points:
        ax.axvline(cp, color='red', linewidth=1.5, linestyle='--', alpha=0.7)
    ax.set_ylabel('Ratio')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 3. State classification
    ax = axes[2]
    
    state_colors = {0: 'green', 1: 'orange', 2: 'red'}
    state_names = {0: 'Static', 1: 'Deforming', 2: 'Peak'}
    
    for state in [0, 1, 2]:
        mask = states == state
        ax.fill_between(frames, 0, 1, where=mask, 
                       color=state_colors[state], alpha=0.6,
                       label=state_names[state])
    
    for cp in change_points:
        ax.axvline(cp, color='black', linewidth=2, linestyle='-')
    
    ax.set_ylabel('State')
    ax.set_xlabel('Frame')
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def plot_distance_analysis(
    result: DistanceAnalysisResult,
    features: dict,
    grid_shape: Tuple[int, int],
    save_path: Optional[str] = None
):
    """
    Visualize distance analysis results.
    """
    T = result.distance_ratios.shape[0]
    frames = np.arange(T)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Plot 1: Distance ratio statistics
    ax = axes[0]
    ax.plot(frames, features['median_ratio'], 'b-', linewidth=2, label='Median Ratio')
    ax.fill_between(frames, 
                    features['median_ratio'] - features['std_ratio'],
                    features['median_ratio'] + features['std_ratio'],
                    alpha=0.3, label='±1 Std')
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.7, label='No Change')
    ax.set_ylabel('Distance Ratio')
    ax.set_title('Point Distance Analysis - Temporal Profile')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Expanding/Contracting points
    ax = axes[1]
    ax.plot(frames, features['expanding_ratio'] * 100, 'r-', linewidth=2, label='Expanding (ratio > 1.05)')
    ax.plot(frames, features['contracting_ratio'] * 100, 'b-', linewidth=2, label='Contracting (ratio < 0.95)')
    ax.set_ylabel('Percentage (%)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Max ratio
    ax = axes[2]
    ax.plot(frames, features['max_ratio'], 'orange', linewidth=2, label='Max Ratio')
    ax.plot(frames, features['min_ratio'], 'purple', linewidth=2, label='Min Ratio')
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.7)
    ax.set_ylabel('Ratio')
    ax.set_xlabel('Frame')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def plot_distance_heatmap(
    result: DistanceAnalysisResult,
    grid_shape: Tuple[int, int],
    frame_idx: int,
    save_path: Optional[str] = None
):
    """
    Plot spatial heatmap of distance ratios for a specific frame.
    """
    rows, cols = grid_shape
    ratios = result.distance_ratios[frame_idx]
    
    # Reshape to grid
    try:
        ratio_grid = ratios.reshape(rows, cols)
    except ValueError:
        print("Cannot reshape ratios to grid")
        return
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    im = ax.imshow(ratio_grid, cmap='RdBu_r', origin='lower', 
                   vmin=0.8, vmax=1.2)
    plt.colorbar(im, ax=ax, label='Distance Ratio')
    
    ax.set_title(f'Frame {frame_idx}: Distance Ratio Heatmap\n(Red = Expanding, Blue = Contracting)')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


# ===== Main Script =====

def load_clip_data(clip_folder: str, max_frames: int = None):
    """Load frames and masks from clip folder."""
    clip_path = Path(clip_folder)
    mask_path = clip_path / "masks"
    
    # Get frame files
    # Get frame files - support both jpg and png
    frame_files = sorted(list(clip_path.glob("*.jpg")) + list(clip_path.glob("*.png")))
    
    # Check for masks in maskB subdir or root
    if mask_path.exists():
        mask_files = sorted(mask_path.glob("*.png"))
    else:
        # Try finding masks in root with maskB_ prefix
        mask_files = sorted(clip_path.glob("maskB_*.png"))
    
    print(f"Found {len(frame_files)} frames, {len(mask_files)} masks in {clip_folder}")
    
    if len(frame_files) == 0:
        raise ValueError(f"No frames found in {clip_folder}. Checked for .jpg and .png")
    
    # Match by number
    frame_dict = {f.stem: f for f in frame_files}
    mask_dict = {m.stem.replace("maskB_", ""): m for m in mask_files}
    
    common = sorted(set(frame_dict.keys()) & set(mask_dict.keys()))
    if len(common) == 0:
        print("Available frames:", list(frame_dict.keys())[:5], "...")
        print("Available masks:", list(mask_dict.keys())[:5], "...")
        raise ValueError(f"No matching frame/mask pairs found. Frames: {len(frame_files)}, Masks: {len(mask_files)}")

    if max_frames:
        common = common[:max_frames]
    
    print(f"Matched {len(common)} pairs")
    
    frames = []
    masks = []
    for num in common:
        img = cv2.imread(str(frame_dict[num]))
        if img is None:
            print(f"Warning: Could not read frame {frame_dict[num]}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)
        
        mask = cv2.imread(str(mask_dict[num]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: Could not read mask {mask_dict[num]}")
            continue
        mask = (mask > 127).astype(np.uint8)
        masks.append(mask)
    
    if len(frames) == 0:
        raise ValueError("No valid frames could be loaded.")

    return np.stack(frames), np.stack(masks)


def sample_grid_in_mask(mask: np.ndarray, grid_size: int = 20, margin: int = 5):
    """Sample grid points within mask."""
    ys, xs = np.where(mask > 0)
    x_min, x_max = xs.min() + margin, xs.max() - margin
    y_min, y_max = ys.min() + margin, ys.max() - margin
    
    x_coords = np.linspace(x_min, x_max, grid_size)
    y_coords = np.linspace(y_min, y_max, grid_size)
    
    xx, yy = np.meshgrid(x_coords, y_coords)
    points = np.stack([xx.ravel(), yy.ravel()], axis=1)
    
    # Check validity
    H, W = mask.shape
    valid = np.zeros(len(points), dtype=bool)
    for i, (x, y) in enumerate(points):
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < W and 0 <= yi < H and mask[yi, xi] > 0:
            valid[i] = True
    
    return points, (grid_size, grid_size), valid


def run_cotracker(frames: np.ndarray, query_points: np.ndarray, device: str = "cuda", model = None):
    """Run CoTracker on frames."""
    import torch
    
    if model is None:
        print("Loading CoTracker...")
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
        model = model.to(device)
    else:
        # Check if model is on correct device
        pass
    
    # Prepare video
    video = torch.from_numpy(frames).permute(0, 3, 1, 2).float()[None].to(device)
    queries = torch.from_numpy(query_points).float()[None].to(device)
    
    # Run tracking
    print("Running CoTracker...")
    with torch.no_grad():
        pred_tracks, pred_visibility = model(video, queries=queries)
    
    tracks = pred_tracks[0].cpu().numpy()
    visibility = pred_visibility[0].cpu().numpy()
    
    print(f"Tracking complete. Tracks shape: {tracks.shape}")
    return tracks, visibility


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Point Distance Analysis")
    parser.add_argument("--clip", type=str, required=True, help="Path to clip folder")
    parser.add_argument("--grid_size", type=int, default=20, help="Grid size")
    parser.add_argument("--radius", type=int, default=2, help="Neighbor radius (1=8, 2=24, 3=48 neighbors)")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--max_frames", type=int, default=None, help="Max frames")
    parser.add_argument("--output", type=str, default=None, help="Output folder")
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("POINT DISTANCE ANALYSIS")
    print("=" * 60)
    print(f"Clip: {args.clip}")
    print(f"Grid: {args.grid_size}x{args.grid_size}")
    print(f"Neighbor radius: {args.radius} ({(2*args.radius+1)**2 - 1} max neighbors)")
    
    # Load data
    print("\n--- Loading Data ---")
    frames, masks = load_clip_data(args.clip, args.max_frames)
    
    # Sample grid
    print("\n--- Sampling Grid ---")
    points, grid_shape, valid_mask = sample_grid_in_mask(masks[0], args.grid_size)
    print(f"Grid shape: {grid_shape}, Valid points: {valid_mask.sum()}")
    
    # Build neighbors with radius
    neighbors = build_grid_neighbors_with_radius(grid_shape, valid_mask, radius=args.radius)
    avg_neighbors = np.mean([len(n) for n in neighbors if len(n) > 0])
    print(f"Average neighbors per point: {avg_neighbors:.1f}")
    
    # Run CoTracker
    print("\n--- Running CoTracker ---")
    query_points = np.zeros((len(points), 3), dtype=np.float32)
    query_points[:, 0] = 0
    query_points[:, 1:] = points
    tracks, visibility = run_cotracker(frames, query_points, args.device)
    
    # Extract ALL features (distance, divergence, triangle area)
    print("\n--- Extracting All Features ---")
    features, distance_result, triangles = extract_all_features(
        tracks, neighbors, grid_shape, valid_mask
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total frames: {len(frames)}")
    
    print(f"\n1. Distance Ratio Features:")
    print(f"   - Overall median: {np.median(features['median_ratio']):.4f}")
    print(f"   - Max observed: {np.max(features['max_ratio']):.4f}")
    print(f"   - Max expanding %: {np.max(features['expanding_ratio'])*100:.1f}%")
    
    print(f"\n2. Divergence Features:")
    print(f"   - Max divergence: {np.max(features['divergence']):.4f}")
    print(f"   - Min divergence: {np.min(features['divergence']):.4f}")
    max_div_frame = np.argmax(features['divergence'])
    print(f"   - Peak divergence at frame: {max_div_frame}")
    
    print(f"\n3. Triangle Area Features:")
    print(f"   - Max mean area ratio: {np.max(features['area_ratio']):.4f}")
    print(f"   - Max area ratio: {np.max(features['max_area_ratio']):.4f}")
    max_area_frame = np.argmax(features['max_area_ratio'])
    print(f"   - Peak area change at frame: {max_area_frame}")
    
    # Find peak frame (using max distance ratio)
    peak_frame = np.argmax(features['max_ratio'])
    print(f"\n   Overall peak deformation at frame: {peak_frame}")
    
    # Change Point Detection
    print("\n--- Change Point Detection ---")
    states, change_points, detection_info = classify_with_change_points(
        features, n_expected_changes=2, method="combined"
    )
    
    print(f"   Detected change points: {change_points}")
    print(f"   State distribution:")
    print(f"     - Static (0): {(states == 0).sum()} frames")
    print(f"     - Deforming (1): {(states == 1).sum()} frames")
    print(f"     - Peak (2): {(states == 2).sum()} frames")
    
    # Visualize
    print("\n--- Visualizations ---")
    
    if args.output:
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Plot all features together
        plot_all_features(features, save_path=str(output_path / "all_features.png"))
        
        # Change point detection results
        plot_change_point_detection(features, states, change_points, detection_info,
                                   save_path=str(output_path / "change_points.png"))
        
        # Distance analysis
        plot_distance_analysis(distance_result, features, grid_shape, 
                              save_path=str(output_path / "distance_temporal.png"))
        plot_distance_heatmap(distance_result, grid_shape, peak_frame,
                             save_path=str(output_path / f"distance_heatmap_frame{peak_frame}.png"))
        
        # Save data
        np.savez(output_path / "all_features_data.npz",
                 tracks=tracks,
                 distance_ratios=distance_result.distance_ratios,
                 states=states,
                 change_points=np.array(change_points),
                 **features)
        print(f"\nResults saved to: {output_path}")
    else:
        plot_all_features(features)
        plot_change_point_detection(features, states, change_points, detection_info)
        plot_distance_analysis(distance_result, features, grid_shape)
        plot_distance_heatmap(distance_result, grid_shape, peak_frame)


def plot_all_features(features: dict, save_path: Optional[str] = None):
    """
    Plot all three feature types together.
    """
    T = len(features['median_ratio'])
    frames = np.arange(T)
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    
    # 1. Distance ratio
    ax = axes[0]
    ax.plot(frames, features['median_ratio'], 'b-', linewidth=2, label='Median Distance Ratio')
    ax.fill_between(frames, 
                    features['median_ratio'] - features['std_ratio'],
                    features['median_ratio'] + features['std_ratio'],
                    alpha=0.3)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.7)
    ax.set_ylabel('Distance Ratio')
    ax.set_title('Multi-Feature Analysis for Deformation Detection')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 2. Divergence
    ax = axes[1]
    ax.plot(frames, features['divergence'], 'g-', linewidth=2, label='Divergence')
    ax.axhline(0.0, color='gray', linestyle='--', alpha=0.7)
    ax.fill_between(frames, 0, features['divergence'], 
                    where=features['divergence'] > 0, color='red', alpha=0.3, label='Expanding')
    ax.fill_between(frames, 0, features['divergence'], 
                    where=features['divergence'] < 0, color='blue', alpha=0.3, label='Contracting')
    ax.set_ylabel('Divergence')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 3. Triangle area ratio
    ax = axes[2]
    ax.plot(frames, features['area_ratio'], 'm-', linewidth=2, label='Mean Area Ratio')
    ax.plot(frames, features['max_area_ratio'], 'orange', linewidth=2, alpha=0.7, label='Max Area Ratio')
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.7)
    ax.set_ylabel('Triangle Area Ratio')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 4. Expanding points percentage
    ax = axes[3]
    ax.plot(frames, features['expanding_ratio'] * 100, 'r-', linewidth=2, label='Expanding Points %')
    ax.plot(frames, features['contracting_ratio'] * 100, 'b-', linewidth=2, label='Contracting Points %')
    ax.set_ylabel('Percentage (%)')
    ax.set_xlabel('Frame')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    main()
