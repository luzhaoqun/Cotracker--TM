"""
Supervised Learning Classification

Train classifiers on labeled data to detect tympanic membrane deformation.
Supports: Random Forest, SVM, MLP (Neural Network)
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict
import pickle
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Add project root to path
_project_root = Path(__file__).parent.parent.absolute()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from point_distance_analysis import (
    load_clip_data,
    sample_grid_in_mask,
    build_grid_neighbors_with_radius,
    run_cotracker,
    extract_all_features,
    build_feature_matrix
)


def load_labels(label_file: str) -> Dict[str, List[Tuple[str, int]]]:
    """
    Load labels from text file.
    
    Format: clip_XXXX/XXXXXX 0/1
    
    Returns:
        Dictionary mapping clip_name to list of (frame_name, label)
    """
    labels_by_clip = {}
    
    with open(label_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) != 2:
                continue
            
            path, label = parts
            label = int(label)
            
            # Parse path: clip_XXXX/XXXXXX
            if '/' in path:
                clip_name, frame_name = path.split('/')
            else:
                continue
            
            if clip_name not in labels_by_clip:
                labels_by_clip[clip_name] = []
            
            labels_by_clip[clip_name].append((frame_name, label))
    
    # Sort frames within each clip
    for clip_name in labels_by_clip:
        labels_by_clip[clip_name].sort(key=lambda x: x[0])
    
    return labels_by_clip


def convert_binary_to_states(binary_labels: np.ndarray) -> np.ndarray:
    """
    Convert binary labels (0=no change, 1=changing) to 3-state labels.
    
    0 → 0 → 0 → 1 → 1 → 1 → 0 → 0 → 0
       Static  | Deforming |   Peak   
       
    Returns:
        states: 0=static, 1=deforming, 2=peak
    """
    T = len(binary_labels)
    states = np.zeros(T, dtype=np.int32)
    
    # Find first 1 (deformation start)
    first_one = -1
    for i, label in enumerate(binary_labels):
        if label == 1:
            first_one = i
            break
    
    # Find last 1 (deformation end)
    last_one = -1
    for i in range(T - 1, -1, -1):
        if binary_labels[i] == 1:
            last_one = i
            break
    
    if first_one == -1:
        # No deformation detected
        return states  # All static
    
    # Assign states
    states[:first_one] = 0          # Static
    states[first_one:last_one+1] = 1  # Deforming
    states[last_one+1:] = 2           # Peak
    
    return states


def extract_features_for_clip(
    clip_folder: str,
    grid_size: int = 20,
    radius: int = 2,
    device: str = "cuda",
    max_frames: Optional[int] = None,
    model = None
) -> Tuple[np.ndarray, dict]:
    """
    Extract features for a single clip.
    
    Returns:
        X: [T, n_features] feature matrix
        features: Dictionary of raw features
    """
    # Load data
    frames, masks = load_clip_data(clip_folder, max_frames)
    
    # Sample grid
    points, grid_shape, valid_mask = sample_grid_in_mask(masks[0], grid_size)
    
    # Build neighbors
    neighbors = build_grid_neighbors_with_radius(grid_shape, valid_mask, radius=radius)
    
    # Run CoTracker
    query_points = np.zeros((len(points), 3), dtype=np.float32)
    query_points[:, 0] = 0
    query_points[:, 1:] = points
    tracks, visibility = run_cotracker(frames, query_points, device, model=model)
    
    # Extract features
    features, distance_result, triangles = extract_all_features(
        tracks, neighbors, grid_shape, valid_mask
    )
    
    # Build feature matrix
    X = build_feature_matrix(features)
    
    return X, features


def prepare_training_data(
    clips_folder: str,
    labels_file: str,
    grid_size: int = 20,
    radius: int = 2,
    device: str = "cuda",
    use_binary: bool = False
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare training data from multiple clips.
    
    Args:
        clips_folder: Folder containing clip_XXXX subfolders
        labels_file: Path to labels.txt
        grid_size: Grid size for tracking
        radius: Neighbor radius
        device: Device for CoTracker
        use_binary: If True, use binary labels (0/1), else convert to 3 states
        
    Returns:
        X: [N, n_features] feature matrix for all frames
        y: [N] labels
        clip_info: List of (clip_name, start_idx, end_idx) for each clip
    """
    # Load labels
    labels_by_clip = load_labels(labels_file)
    print(f"Found labels for {len(labels_by_clip)} clips")
    
    all_X = []
    all_y = []
    clip_info = []
    
    clips_path = Path(clips_folder)
    
    # Load CoTracker model once
    import torch
    print("Loading CoTracker model...")
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
    model = model.to(device)
    
    for clip_name, frame_labels in labels_by_clip.items():
        clip_path = clips_path / clip_name
        
        if not clip_path.exists():
            print(f"  Skipping {clip_name}: folder not found")
            continue
        
        print(f"\nProcessing {clip_name}...")
        
        try:
            # Extract features (pass pre-loaded model)
            X, features = extract_features_for_clip(
                str(clip_path), grid_size, radius, device, model=model
            )
            
            # Get labels (binary)
            binary_labels = np.array([label for _, label in frame_labels])
            
            # Ensure lengths match
            min_len = min(len(X), len(binary_labels))
            X = X[:min_len]
            binary_labels = binary_labels[:min_len]
            
            # Convert to 3 states if needed
            if use_binary:
                y = binary_labels
            else:
                y = convert_binary_to_states(binary_labels)
            
            # Record
            start_idx = len(all_y)
            all_X.append(X)
            all_y.append(y)
            clip_info.append((clip_name, start_idx, start_idx + len(y)))
            
            print(f"  Frames: {len(y)}, States: {np.bincount(y, minlength=3)}")
            
        except Exception as e:
            print(f"  Error processing {clip_name}: {e}")
            continue
    
    if len(all_X) == 0:
        raise ValueError("No valid clips found!")
    
    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    
    print(f"\nTotal: {len(y)} frames from {len(clip_info)} clips")
    print(f"Class distribution: {np.bincount(y)}")
    
    return X, y, clip_info


def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "rf",
    test_size: float = 0.2,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None
) -> Tuple[object, StandardScaler, dict]:
    """
    Train a classifier.
    
    Args:
        X: Feature matrix (training data)
        y: Labels (training labels)
        model_type: 'rf' (Random Forest), 'svm', 'mlp'
        test_size: Fraction for test set (only used if X_val is None)
        X_val: Optional validation feature matrix
        y_val: Optional validation labels
        
    Returns:
        model: Trained classifier
        scaler: Fitted StandardScaler
        metrics: Dictionary of evaluation metrics
    """
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X)
    
    # Use provided validation set or split from training data
    if X_val is not None and y_val is not None:
        print("Using provided validation set (separate folders)")
        X_test = scaler.transform(X_val)
        y_train, y_test = y, y_val
        X_train = X_train_scaled
    else:
        print(f"Splitting training data (test_size={test_size})")
        X_train, X_test, y_train, y_test = train_test_split(
            X_train_scaled, y, test_size=test_size, random_state=42, stratify=y
        )
    
    print(f"Train samples: {len(y_train)}, Val samples: {len(y_test)}")
    
    # Create model
    if model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
    elif model_type == "svm":
        model = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            class_weight='balanced',
            probability=True
        )
    elif model_type == "mlp":
        model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            random_state=42,
            early_stopping=True
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train
    print(f"\nTraining {model_type.upper()}...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    
    # Determine target names based on unique labels
    unique_labels = sorted(set(y_train) | set(y_test))
    if len(unique_labels) == 2:
        target_names = ['No Change', 'Changing']
    else:
        target_names = ['Static', 'Deforming', 'Peak']
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, 
                                                        labels=unique_labels,
                                                        target_names=target_names[:len(unique_labels)]),
        'confusion_matrix': confusion_matrix(y_test, y_pred, labels=unique_labels),
        'cv_scores': cross_val_score(model, X_train, y_train, cv=min(5, len(set(y_train))))
    }
    
    print(f"\nValidation Accuracy: {metrics['accuracy']:.2%}")
    print(f"Cross-validation (on train): {metrics['cv_scores'].mean():.2%} ± {metrics['cv_scores'].std():.2%}")
    print(f"\n{metrics['classification_report']}")
    
    return model, scaler, metrics


def save_model(
    model: object,
    scaler: StandardScaler,
    metrics: dict,
    output_path: str
):
    """Save trained model and scaler."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model
    with open(output_path / "model.pkl", 'wb') as f:
        pickle.dump(model, f)
    
    # Save scaler
    with open(output_path / "scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save metrics (convert numpy to lists for JSON)
    metrics_json = {
        'accuracy': float(metrics['accuracy']),
        'cv_scores': metrics['cv_scores'].tolist(),
        'confusion_matrix': metrics['confusion_matrix'].tolist(),
        'classification_report': metrics['classification_report']
    }
    with open(output_path / "metrics.json", 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    print(f"\nModel saved to: {output_path}")


def load_model(model_path: str) -> Tuple[object, StandardScaler]:
    """Load trained model and scaler."""
    model_path = Path(model_path)
    
    with open(model_path / "model.pkl", 'rb') as f:
        model = pickle.load(f)
    
    with open(model_path / "scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    
    return model, scaler


def predict_clip(
    clip_folder: str,
    model: object,
    scaler: StandardScaler,
    grid_size: int = 20,
    radius: int = 2,
    device: str = "cuda"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict states for a clip using trained model.
    
    Returns:
        predictions: [T] predicted states
        probabilities: [T, 3] class probabilities
    """
    # Extract features
    X, features = extract_features_for_clip(clip_folder, grid_size, radius, device)
    
    # Scale
    X_scaled = scaler.transform(X)
    
    # Predict
    predictions = model.predict(X_scaled)
    
    # Get probabilities if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X_scaled)
    else:
        probabilities = np.zeros((len(predictions), 3))
        probabilities[np.arange(len(predictions)), predictions] = 1.0
    
    return predictions, probabilities


def plot_training_results(
    metrics: dict,
    save_path: Optional[str] = None
):
    """Plot training results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Confusion matrix
    ax = axes[0]
    cm = metrics['confusion_matrix']
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(['Static', 'Deforming', 'Peak'])
    ax.set_yticklabels(['Static', 'Deforming', 'Peak'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    
    # Add values
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                   color='white' if cm[i, j] > cm.max()/2 else 'black')
    
    plt.colorbar(im, ax=ax)
    
    # Cross-validation scores
    ax = axes[1]
    cv_scores = metrics['cv_scores']
    ax.bar(range(len(cv_scores)), cv_scores, color='steelblue')
    ax.axhline(cv_scores.mean(), color='red', linestyle='--', 
               label=f'Mean: {cv_scores.mean():.2%}')
    ax.set_xlabel('Fold')
    ax.set_ylabel('Accuracy')
    ax.set_title('Cross-Validation Scores')
    ax.legend()
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Supervised Learning Classification")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a classifier')
    train_parser.add_argument("--clips", type=str, required=True,
                              help="Folder containing clip_XXXX subfolders")
    train_parser.add_argument("--labels", type=str, required=True,
                              help="Path to labels.txt")
    train_parser.add_argument("--output", type=str, required=True,
                              help="Output folder for model")
    train_parser.add_argument("--model", type=str, default="rf",
                              choices=["rf", "svm", "mlp"],
                              help="Model type: rf (Random Forest), svm, mlp")
    train_parser.add_argument("--grid_size", type=int, default=20)
    train_parser.add_argument("--radius", type=int, default=2)
    train_parser.add_argument("--device", type=str, default="cuda")
    train_parser.add_argument("--binary", action="store_true",
                              help="Use binary labels (0/1) instead of 3 states")
    train_parser.add_argument("--val_clips", type=str, default=None,
                              help="Optional: Folder containing validation clip_XXXX subfolders")
    train_parser.add_argument("--val_labels", type=str, default=None,
                              help="Optional: Path to validation labels.txt")
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict on a clip')
    predict_parser.add_argument("--clip", type=str, required=True,
                                help="Path to clip folder")
    predict_parser.add_argument("--model_path", type=str, required=True,
                                help="Path to trained model folder")
    predict_parser.add_argument("--output", type=str, default=None,
                                help="Output folder for results")
    predict_parser.add_argument("--grid_size", type=int, default=20)
    predict_parser.add_argument("--radius", type=int, default=2)
    predict_parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    if args.command == 'train':
        print("\n" + "=" * 60)
        print("SUPERVISED LEARNING - TRAINING")
        print("=" * 60)
        
        # Prepare training data
        print("\n--- Preparing Training Data ---")
        X_train, y_train, clip_info = prepare_training_data(
            args.clips, args.labels,
            args.grid_size, args.radius, args.device,
            use_binary=args.binary
        )
        
        # Prepare validation data if provided
        X_val, y_val = None, None
        if args.val_clips and args.val_labels:
            print("\n--- Preparing Validation Data ---")
            X_val, y_val, val_clip_info = prepare_training_data(
                args.val_clips, args.val_labels,
                args.grid_size, args.radius, args.device,
                use_binary=args.binary
            )
        
        # Train
        model, scaler, metrics = train_classifier(
            X_train, y_train, args.model,
            X_val=X_val, y_val=y_val
        )
        
        # Save
        save_model(model, scaler, metrics, args.output)
        
        # Plot
        output_path = Path(args.output)
        plot_training_results(metrics, save_path=str(output_path / "training_results.png"))
        
    elif args.command == 'predict':
        print("\n" + "=" * 60)
        print("SUPERVISED LEARNING - PREDICTION")
        print("=" * 60)
        
        # Load model
        print("\n--- Loading Model ---")
        model, scaler = load_model(args.model_path)
        
        # Predict
        print("\n--- Predicting ---")
        predictions, probabilities = predict_clip(
            args.clip, model, scaler,
            args.grid_size, args.radius, args.device
        )
        
        # Print results
        state_names = ['Static', 'Deforming', 'Peak']
        print(f"\nPredictions:")
        for i, cnt in enumerate(np.bincount(predictions, minlength=3)):
            print(f"  {state_names[i]}: {cnt} frames")
        
        # Find transitions
        transitions = []
        for t in range(1, len(predictions)):
            if predictions[t] != predictions[t-1]:
                transitions.append((t, predictions[t-1], predictions[t]))
        
        print(f"\nState transitions:")
        for t, from_s, to_s in transitions:
            print(f"  Frame {t}: {state_names[from_s]} → {state_names[to_s]}")
        
        # Save if output specified
        if args.output:
            output_path = Path(args.output)
            output_path.mkdir(parents=True, exist_ok=True)
            
            np.savez(output_path / "predictions.npz",
                     predictions=predictions,
                     probabilities=probabilities)
            print(f"\nResults saved to: {output_path}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
