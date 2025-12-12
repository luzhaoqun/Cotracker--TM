"""
Clustering-based Deformation Classification

Unsupervised approach: cluster all frames into 3 groups
to automatically discover Static / Deforming / Peak patterns.
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Add project root to path
_project_root = Path(__file__).parent.parent.absolute()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Import from point_distance_analysis
from point_distance_analysis import (
    load_clip_data,
    sample_grid_in_mask,
    build_grid_neighbors_with_radius,
    run_cotracker,
    extract_all_features,
    DistanceAnalysisResult
)


@dataclass
class ClusteringResult:
    """Result of clustering-based classification."""
    labels: np.ndarray  # [T] cluster labels
    ordered_labels: np.ndarray  # [T] ordered by deformation level (0=static, 2=peak)
    cluster_centers: np.ndarray  # [K, n_features] cluster centers
    deformation_order: List[int]  # Mapping from cluster to deformation level
    confidence: np.ndarray  # [T] classification confidence


def build_feature_matrix(features: dict) -> np.ndarray:
    """
    Build feature matrix for clustering.
    
    Args:
        features: Dictionary of per-frame features
        
    Returns:
        X: [T, n_features] feature matrix
    """
    T = len(features['median_ratio'])
    
    # Select features for clustering
    feature_list = [
        features['median_ratio'] - 1.0,      # Distance ratio (centered)
        features['max_ratio'] - 1.0,          # Max distance ratio
        features['std_ratio'],                 # Variation in ratios
        features['expanding_ratio'],           # Fraction expanding
        features['divergence'],                # Displacement divergence
        features['area_ratio'] - 1.0,          # Triangle area ratio
        features['max_area_ratio'] - 1.0,      # Max area ratio
    ]
    
    X = np.column_stack(feature_list)
    return X


def cluster_kmeans(X: np.ndarray, n_clusters: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    K-Means clustering.
    """
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    
    return labels, centers


def cluster_gmm(X: np.ndarray, n_clusters: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Gaussian Mixture Model clustering.
    Returns labels, centers, and probabilities.
    """
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit GMM
    gmm = GaussianMixture(n_components=n_clusters, random_state=42, n_init=5)
    gmm.fit(X_scaled)
    
    labels = gmm.predict(X_scaled)
    probs = gmm.predict_proba(X_scaled)
    centers = scaler.inverse_transform(gmm.means_)
    
    return labels, centers, probs


def order_clusters_by_deformation(
    labels: np.ndarray,
    cluster_centers: np.ndarray,
    features: dict
) -> Tuple[np.ndarray, List[int]]:
    """
    Order clusters by deformation level.
    
    Assumption: Static has lowest feature values, Peak has highest.
    
    Returns:
        ordered_labels: Labels remapped to 0=static, 1=deforming, 2=peak
        order: Original cluster ID for each deformation level
    """
    n_clusters = len(cluster_centers)
    
    # Use first feature column (median_ratio - 1.0) to order clusters
    # Lower value = less deformation
    deformation_level = cluster_centers[:, 0]  # median_ratio - 1.0
    
    # Sort clusters by deformation level
    order = np.argsort(deformation_level)  # [static, deforming, peak]
    
    # Create mapping
    label_map = {old: new for new, old in enumerate(order)}
    ordered_labels = np.array([label_map[l] for l in labels])
    
    return ordered_labels, order.tolist()


def temporal_smoothing(
    labels: np.ndarray,
    window_size: int = 3
) -> np.ndarray:
    """
    Apply temporal smoothing to remove noise in labels.
    """
    T = len(labels)
    smoothed = labels.copy()
    
    for t in range(T):
        start = max(0, t - window_size // 2)
        end = min(T, t + window_size // 2 + 1)
        window = labels[start:end]
        
        # Majority vote
        counts = np.bincount(window, minlength=3)
        smoothed[t] = np.argmax(counts)
    
    return smoothed


def enforce_state_order(labels: np.ndarray) -> np.ndarray:
    """
    Enforce temporal ordering: Static → Deforming → Peak.
    
    Once a higher state is reached, don't go back to lower states.
    """
    T = len(labels)
    result = labels.copy()
    max_state_seen = 0
    
    for t in range(T):
        if result[t] > max_state_seen:
            max_state_seen = result[t]
        elif result[t] < max_state_seen:
            # Don't allow going back
            result[t] = max_state_seen
    
    return result


def classify_with_clustering(
    features: dict,
    method: str = "gmm",
    n_clusters: int = 3,
    smooth: bool = True,
    enforce_order: bool = False
) -> ClusteringResult:
    """
    Classify frames using clustering.
    
    Args:
        features: Dictionary of per-frame features
        method: 'kmeans' or 'gmm'
        n_clusters: Number of clusters
        smooth: Apply temporal smoothing
        enforce_order: Enforce monotonic state progression
        
    Returns:
        ClusteringResult
    """
    # Build feature matrix
    X = build_feature_matrix(features)
    
    # Cluster
    if method == "kmeans":
        labels, centers = cluster_kmeans(X, n_clusters)
        confidence = np.ones(len(labels))  # K-Means doesn't give probabilities
    else:  # GMM
        labels, centers, probs = cluster_gmm(X, n_clusters)
        confidence = np.max(probs, axis=1)
    
    # Order clusters by deformation level
    ordered_labels, deformation_order = order_clusters_by_deformation(
        labels, centers, features
    )
    
    # Apply smoothing
    if smooth:
        ordered_labels = temporal_smoothing(ordered_labels)
    
    # Enforce state ordering (optional)
    if enforce_order:
        ordered_labels = enforce_state_order(ordered_labels)
    
    return ClusteringResult(
        labels=labels,
        ordered_labels=ordered_labels,
        cluster_centers=centers,
        deformation_order=deformation_order,
        confidence=confidence
    )


def plot_clustering_result(
    features: dict,
    result: ClusteringResult,
    save_path: Optional[str] = None
):
    """
    Visualize clustering results.
    """
    T = len(result.ordered_labels)
    frames = np.arange(T)
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    
    # 1. Features with cluster coloring
    ax = axes[0]
    colors = ['green', 'orange', 'red']
    for i in range(3):
        mask = result.ordered_labels == i
        ax.scatter(frames[mask], features['median_ratio'][mask], 
                   c=colors[i], s=50, alpha=0.7)
    ax.plot(frames, features['median_ratio'], 'gray', alpha=0.5, linewidth=1)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Distance Ratio')
    ax.set_title('Clustering-based Classification')
    ax.grid(True, alpha=0.3)
    
    # 2. Multiple features
    ax = axes[1]
    ax.plot(frames, features['median_ratio'], 'b-', linewidth=2, label='Distance Ratio')
    ax.plot(frames, features['area_ratio'], 'm-', linewidth=2, label='Area Ratio')
    ax.plot(frames, features['divergence'] + 1, 'g-', linewidth=2, label='Divergence+1')
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Feature Value')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 3. Confidence
    ax = axes[2]
    ax.fill_between(frames, 0, result.confidence, alpha=0.5, color='blue')
    ax.plot(frames, result.confidence, 'b-', linewidth=2)
    ax.set_ylabel('Confidence')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    
    # 4. State classification
    ax = axes[3]
    state_colors = {0: 'green', 1: 'orange', 2: 'red'}
    state_names = {0: 'Static', 1: 'Deforming', 2: 'Peak'}
    
    for state in [0, 1, 2]:
        mask = result.ordered_labels == state
        ax.fill_between(frames, 0, 1, where=mask, 
                       color=state_colors[state], alpha=0.6,
                       label=state_names[state])
    
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


def plot_feature_space(
    features: dict,
    result: ClusteringResult,
    save_path: Optional[str] = None
):
    """
    Plot 2D projection of feature space with clusters.
    """
    X = build_feature_matrix(features)
    
    # PCA for visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(StandardScaler().fit_transform(X))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['green', 'orange', 'red']
    state_names = ['Static', 'Deforming', 'Peak']
    
    for i in range(3):
        mask = result.ordered_labels == i
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                   c=colors[i], s=80, alpha=0.7, label=state_names[i])
        
        # Add frame numbers
        for j in np.where(mask)[0]:
            ax.annotate(str(j), (X_2d[j, 0], X_2d[j, 1]), fontsize=8, alpha=0.7)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('Feature Space Clustering (PCA Projection)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Clustering-based Classification")
    parser.add_argument("--clip", type=str, required=True, help="Path to clip folder")
    parser.add_argument("--grid_size", type=int, default=20, help="Grid size")
    parser.add_argument("--radius", type=int, default=2, help="Neighbor radius")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--max_frames", type=int, default=None, help="Max frames")
    parser.add_argument("--output", type=str, default=None, help="Output folder")
    parser.add_argument("--method", type=str, default="gmm", 
                       choices=["kmeans", "gmm"], help="Clustering method")
    parser.add_argument("--n_clusters", type=int, default=3, help="Number of clusters")
    parser.add_argument("--no_smooth", action="store_true", help="Disable smoothing")
    parser.add_argument("--enforce_order", action="store_true", 
                       help="Enforce monotonic state progression")
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("CLUSTERING-BASED CLASSIFICATION")
    print("=" * 60)
    print(f"Clip: {args.clip}")
    print(f"Method: {args.method.upper()}")
    print(f"Clusters: {args.n_clusters}")
    
    # Load data
    print("\n--- Loading Data ---")
    frames, masks = load_clip_data(args.clip, args.max_frames)
    
    # Sample grid
    print("\n--- Sampling Grid ---")
    points, grid_shape, valid_mask = sample_grid_in_mask(masks[0], args.grid_size)
    print(f"Grid shape: {grid_shape}, Valid points: {valid_mask.sum()}")
    
    # Build neighbors
    neighbors = build_grid_neighbors_with_radius(grid_shape, valid_mask, radius=args.radius)
    
    # Run CoTracker
    print("\n--- Running CoTracker ---")
    query_points = np.zeros((len(points), 3), dtype=np.float32)
    query_points[:, 0] = 0
    query_points[:, 1:] = points
    tracks, visibility = run_cotracker(frames, query_points, args.device)
    
    # Extract features
    print("\n--- Extracting Features ---")
    features, distance_result, triangles = extract_all_features(
        tracks, neighbors, grid_shape, valid_mask
    )
    
    # Clustering
    print("\n--- Clustering ---")
    result = classify_with_clustering(
        features,
        method=args.method,
        n_clusters=args.n_clusters,
        smooth=not args.no_smooth,
        enforce_order=args.enforce_order
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total frames: {len(frames)}")
    print(f"\nCluster assignments (original):")
    for i, cnt in enumerate(np.bincount(result.labels, minlength=3)):
        print(f"   Cluster {i}: {cnt} frames")
    
    print(f"\nState distribution (ordered):")
    state_names = ['Static', 'Deforming', 'Peak']
    for i, cnt in enumerate(np.bincount(result.ordered_labels, minlength=3)):
        print(f"   {state_names[i]}: {cnt} frames")
    
    print(f"\nDeformation order: {result.deformation_order}")
    print(f"Average confidence: {result.confidence.mean():.2%}")
    
    # Find state transitions
    transitions = []
    for t in range(1, len(result.ordered_labels)):
        if result.ordered_labels[t] != result.ordered_labels[t-1]:
            transitions.append((t, result.ordered_labels[t-1], result.ordered_labels[t]))
    
    print(f"\nState transitions:")
    for t, from_s, to_s in transitions:
        print(f"   Frame {t}: {state_names[from_s]} → {state_names[to_s]}")
    
    # Visualize
    print("\n--- Visualizations ---")
    
    if args.output:
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        plot_clustering_result(features, result, 
                              save_path=str(output_path / "clustering_result.png"))
        plot_feature_space(features, result,
                          save_path=str(output_path / "feature_space.png"))
        
        # Save data
        np.savez(output_path / "clustering_data.npz",
                 labels=result.labels,
                 ordered_labels=result.ordered_labels,
                 confidence=result.confidence,
                 deformation_order=result.deformation_order,
                 **features)
        print(f"\nResults saved to: {output_path}")
    else:
        plot_clustering_result(features, result)
        plot_feature_space(features, result)


if __name__ == "__main__":
    main()
