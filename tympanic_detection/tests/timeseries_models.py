"""
Time-Series Models for Deformation Classification

Compare multiple temporal models:
1. HMM (Hidden Markov Model)
2. 1D-CNN (Convolutional Neural Network)
3. LSTM/GRU (Recurrent Neural Network)
4. Transformer (Self-Attention)
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict
import pickle
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
_project_root = Path(__file__).parent.parent.absolute()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Import from existing scripts
from supervised_classification import (
    load_labels,
    convert_binary_to_states,
    extract_features_for_clip,
    build_feature_matrix
)
from point_distance_analysis import (
    load_clip_data,
    sample_grid_in_mask,
    build_grid_neighbors_with_radius,
    run_cotracker,
    extract_all_features
)


# ===== Data Preparation =====

def prepare_sequence_data(
    clips_folder: str,
    labels_file: str,
    grid_size: int = 20,
    radius: int = 2,
    device: str = "cuda",
    max_len: int = 50
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Prepare sequence data for time-series models.
    
    Returns:
        X: [N_clips, max_len, n_features] padded feature sequences
        y: [N_clips, max_len] padded state labels
        lengths: [N_clips] actual lengths of each sequence
        clip_names: List of clip names
    """
    labels_by_clip = load_labels(labels_file)
    print(f"Found labels for {len(labels_by_clip)} clips")
    
    all_X = []
    all_y = []
    all_lengths = []
    clip_names = []
    
    clips_path = Path(clips_folder)
    n_features = 10  # Number of features from build_feature_matrix
    
    # Load CoTracker model once
    import torch
    print("Loading CoTracker model...")
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
    model = model.to(device)
    
    for clip_name, frame_labels in labels_by_clip.items():
        clip_path = clips_path / clip_name
        
        if not clip_path.exists():
            continue
        
        print(f"Processing {clip_name}...", end=" ")
        
        try:
            # Extract features (pass pre-loaded model)
            X, features = extract_features_for_clip(
                str(clip_path), grid_size, radius, device, model=model
            )
            
            # Get labels
            binary_labels = np.array([label for _, label in frame_labels])
            min_len = min(len(X), len(binary_labels))
            X = X[:min_len]
            binary_labels = binary_labels[:min_len]
            
            # Convert to 3 states
            y = convert_binary_to_states(binary_labels)
            
            # Pad or truncate
            actual_len = len(X)
            if actual_len > max_len:
                X = X[:max_len]
                y = y[:max_len]
                actual_len = max_len
            elif actual_len < max_len:
                X_padded = np.zeros((max_len, n_features))
                y_padded = np.zeros(max_len, dtype=np.int32)
                X_padded[:actual_len] = X
                y_padded[:actual_len] = y
                X = X_padded
                y = y_padded
            
            all_X.append(X)
            all_y.append(y)
            all_lengths.append(actual_len)
            clip_names.append(clip_name)
            
            print(f"{actual_len} frames")
            
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    X = np.stack(all_X)  # [N, max_len, n_features]
    y = np.stack(all_y)  # [N, max_len]
    lengths = np.array(all_lengths)
    
    print(f"\nTotal: {len(clip_names)} clips, shape: {X.shape}")
    
    return X, y, lengths, clip_names


# ===== Model 1: HMM (Hidden Markov Model) =====

class HMMModel:
    """Hidden Markov Model for sequence classification."""
    
    def __init__(self, n_states: int = 3):
        self.n_states = n_states
        self.transition_probs = None
        self.emission_means = None
        self.emission_stds = None
        self.initial_probs = None
        self.scaler = StandardScaler()
    
    def fit(self, X: np.ndarray, y: np.ndarray, lengths: np.ndarray):
        """
        Fit HMM parameters from labeled data.
        """
        # Flatten and scale features
        X_flat = X.reshape(-1, X.shape[-1])
        self.scaler.fit(X_flat[X_flat.sum(axis=1) != 0])  # Exclude padding
        
        # Estimate initial probabilities
        self.initial_probs = np.zeros(self.n_states)
        for i, length in enumerate(lengths):
            if length > 0:
                self.initial_probs[y[i, 0]] += 1
        self.initial_probs = self.initial_probs / self.initial_probs.sum()
        
        # Estimate transition probabilities
        self.transition_probs = np.zeros((self.n_states, self.n_states))
        for i, length in enumerate(lengths):
            for t in range(1, length):
                from_state = y[i, t-1]
                to_state = y[i, t]
                self.transition_probs[from_state, to_state] += 1
        
        # Normalize
        for s in range(self.n_states):
            row_sum = self.transition_probs[s].sum()
            if row_sum > 0:
                self.transition_probs[s] /= row_sum
            else:
                self.transition_probs[s] = 1.0 / self.n_states
        
        # Estimate emission parameters (Gaussian)
        self.emission_means = []
        self.emission_stds = []
        
        for s in range(self.n_states):
            state_features = []
            for i, length in enumerate(lengths):
                mask = y[i, :length] == s
                if mask.sum() > 0:
                    state_features.append(X[i, :length][mask])
            
            if len(state_features) > 0:
                state_features = np.vstack(state_features)
                self.emission_means.append(state_features.mean(axis=0))
                self.emission_stds.append(state_features.std(axis=0) + 1e-6)
            else:
                self.emission_means.append(np.zeros(X.shape[-1]))
                self.emission_stds.append(np.ones(X.shape[-1]))
        
        self.emission_means = np.array(self.emission_means)
        self.emission_stds = np.array(self.emission_stds)
    
    def predict(self, X: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        """
        Predict states using Viterbi algorithm.
        """
        N = len(X)
        max_len = X.shape[1]
        predictions = np.zeros((N, max_len), dtype=np.int32)
        
        for i in range(N):
            length = lengths[i]
            if length == 0:
                continue
            
            seq = X[i, :length]
            
            # Emission log-probabilities (Gaussian)
            log_emission = np.zeros((length, self.n_states))
            for s in range(self.n_states):
                diff = seq - self.emission_means[s]
                log_prob = -0.5 * np.sum((diff / self.emission_stds[s])**2, axis=1)
                log_emission[:, s] = log_prob
            
            # Viterbi algorithm
            log_delta = np.zeros((length, self.n_states))
            psi = np.zeros((length, self.n_states), dtype=np.int32)
            
            log_delta[0] = np.log(self.initial_probs + 1e-10) + log_emission[0]
            
            for t in range(1, length):
                for s in range(self.n_states):
                    trans = log_delta[t-1] + np.log(self.transition_probs[:, s] + 1e-10)
                    psi[t, s] = np.argmax(trans)
                    log_delta[t, s] = trans[psi[t, s]] + log_emission[t, s]
            
            # Backtrack
            path = np.zeros(length, dtype=np.int32)
            path[-1] = np.argmax(log_delta[-1])
            for t in range(length-2, -1, -1):
                path[t] = psi[t+1, path[t+1]]
            
            predictions[i, :length] = path
        
        return predictions


# ===== Model 2-4: Deep Learning Models =====

def create_cnn_model(n_features: int, max_len: int, n_classes: int = 3):
    """Create 1D-CNN model."""
    import torch
    import torch.nn as nn
    
    class CNN1D(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(n_features, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
            self.fc = nn.Linear(64, n_classes)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.3)
        
        def forward(self, x):
            # x: [batch, seq_len, features] -> [batch, features, seq_len]
            x = x.permute(0, 2, 1)
            x = self.relu(self.conv1(x))
            x = self.dropout(x)
            x = self.relu(self.conv2(x))
            x = self.dropout(x)
            x = self.relu(self.conv3(x))
            # [batch, 64, seq_len] -> [batch, seq_len, 64]
            x = x.permute(0, 2, 1)
            x = self.fc(x)  # [batch, seq_len, n_classes]
            return x
    
    return CNN1D()


def create_lstm_model(n_features: int, max_len: int, n_classes: int = 3):
    """Create LSTM model."""
    import torch
    import torch.nn as nn
    
    class LSTMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(n_features, 64, num_layers=2, 
                               batch_first=True, dropout=0.3, bidirectional=True)
            self.fc = nn.Linear(128, n_classes)  # 64*2 for bidirectional
        
        def forward(self, x):
            lstm_out, _ = self.lstm(x)  # [batch, seq_len, 128]
            out = self.fc(lstm_out)  # [batch, seq_len, n_classes]
            return out
    
    return LSTMModel()


def create_transformer_model(n_features: int, max_len: int, n_classes: int = 3):
    """Create lightweight Transformer model."""
    import torch
    import torch.nn as nn
    
    class TransformerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Linear(n_features, 32)
            self.pos_encoding = nn.Parameter(torch.randn(1, max_len, 32) * 0.1)
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=32, nhead=4, dim_feedforward=64, 
                dropout=0.3, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.fc = nn.Linear(32, n_classes)
        
        def forward(self, x):
            x = self.embedding(x)  # [batch, seq_len, 32]
            x = x + self.pos_encoding[:, :x.size(1), :]
            x = self.transformer(x)
            x = self.fc(x)  # [batch, seq_len, n_classes]
            return x
    
    return TransformerModel()


def train_deep_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    lengths_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    lengths_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 0.001,
    device: str = "cuda"
):
    """Train a PyTorch model."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    
    model = model.to(device)
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)
    
    # Create mask for valid positions (not padding)
    mask_train = torch.zeros_like(y_train_t, dtype=torch.bool)
    mask_val = torch.zeros_like(y_val_t, dtype=torch.bool)
    for i, length in enumerate(lengths_train):
        mask_train[i, :length] = True
    for i, length in enumerate(lengths_val):
        mask_val[i, :length] = True
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    best_val_acc = 0
    best_model_state = None
    patience = 10
    patience_counter = 0
    
    train_losses = []
    val_accs = []
    
    for epoch in range(epochs):
        model.train()
        
        # Training
        optimizer.zero_grad()
        outputs = model(X_train_t)  # [batch, seq_len, n_classes]
        
        # Flatten for loss calculation
        outputs_flat = outputs.view(-1, 3)
        y_flat = y_train_t.view(-1)
        mask_flat = mask_train.view(-1)
        
        loss = criterion(outputs_flat, y_flat)
        loss = (loss * mask_flat.float()).sum() / mask_flat.sum()
        
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # Validation
        model.eval()
        with torch.no_grad():
            outputs_val = model(X_val_t)
            preds = outputs_val.argmax(dim=-1)
            
            correct = ((preds == y_val_t) & mask_val).sum().item()
            total = mask_val.sum().item()
            val_acc = correct / total
            val_accs.append(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: Loss={loss.item():.4f}, Val Acc={val_acc:.2%}")
        
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_accs


def predict_deep_model(model, X: np.ndarray, lengths: np.ndarray, device: str = "cuda"):
    """Predict with PyTorch model."""
    import torch
    
    model = model.to(device)
    model.eval()
    
    X_t = torch.FloatTensor(X).to(device)
    
    with torch.no_grad():
        outputs = model(X_t)
        preds = outputs.argmax(dim=-1).cpu().numpy()
    
    return preds


# ===== Evaluation =====

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, lengths: np.ndarray) -> dict:
    """Evaluate predictions, excluding padding."""
    # Flatten while excluding padding
    y_true_flat = []
    y_pred_flat = []
    
    for i, length in enumerate(lengths):
        y_true_flat.extend(y_true[i, :length])
        y_pred_flat.extend(y_pred[i, :length])
    
    y_true_flat = np.array(y_true_flat)
    y_pred_flat = np.array(y_pred_flat)
    
    return {
        'accuracy': accuracy_score(y_true_flat, y_pred_flat),
        'report': classification_report(y_true_flat, y_pred_flat,
                                        target_names=['Static', 'Deforming', 'Peak']),
        'confusion_matrix': confusion_matrix(y_true_flat, y_pred_flat)
    }


def plot_comparison(results: Dict[str, dict], save_path: Optional[str] = None):
    """Plot comparison of all models."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Accuracy comparison
    ax = axes[0, 0]
    models = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in models]
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    bars = ax.bar(models, accuracies, color=colors[:len(models)])
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Comparison - Accuracy')
    ax.set_ylim(0, 1)
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{acc:.1%}', ha='center', va='bottom', fontsize=12)
    
    # 2-5. Confusion matrices
    for idx, (model_name, result) in enumerate(results.items()):
        if idx >= 3:
            break
        ax = axes[(idx+1)//2, (idx+1)%2]
        cm = result['confusion_matrix']
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(['Static', 'Deform', 'Peak'])
        ax.set_yticklabels(['Static', 'Deform', 'Peak'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'{model_name} Confusion Matrix')
        
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                       color='white' if cm[i, j] > cm.max()/2 else 'black')
    
    # If 4 models, use the last subplot
    if len(results) >= 4:
        ax = axes[1, 1]
        model_name = list(results.keys())[3]
        result = results[model_name]
        cm = result['confusion_matrix']
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(['Static', 'Deform', 'Peak'])
        ax.set_yticklabels(['Static', 'Deform', 'Peak'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'{model_name} Confusion Matrix')
        
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                       color='white' if cm[i, j] > cm.max()/2 else 'black')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


# ===== Main =====

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare Time-Series Models")
    parser.add_argument("--clips", type=str, required=True,
                        help="Folder containing clip_XXXX subfolders")
    parser.add_argument("--labels", type=str, required=True,
                        help="Path to labels.txt")
    parser.add_argument("--output", type=str, default="./model_comparison",
                        help="Output folder")
    parser.add_argument("--grid_size", type=int, default=20)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_len", type=int, default=50,
                        help="Maximum sequence length")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs for deep models")
    parser.add_argument("--models", type=str, nargs='+',
                        default=["hmm", "cnn", "lstm", "transformer"],
                        choices=["hmm", "cnn", "lstm", "transformer"],
                        help="Models to compare")
    parser.add_argument("--val_clips", type=str, default=None,
                        help="Optional: Folder containing validation clip_XXXX subfolders")
    parser.add_argument("--val_labels", type=str, default=None,
                        help="Optional: Path to validation labels.txt")
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("TIME-SERIES MODEL COMPARISON")
    print("=" * 60)
    
    # Prepare training data
    print("\n--- Preparing Training Data ---")
    X_train, y_train, lengths_train, clip_names_train = prepare_sequence_data(
        args.clips, args.labels,
        args.grid_size, args.radius, args.device, args.max_len
    )
    
    # Prepare validation/test data
    if args.val_clips and args.val_labels:
        print("\n--- Preparing Validation Data (from separate folder) ---")
        X_test, y_test, lengths_test, clip_names_test = prepare_sequence_data(
            args.val_clips, args.val_labels,
            args.grid_size, args.radius, args.device, args.max_len
        )
        print(f"\nTrain: {len(X_train)} clips, Test: {len(X_test)} clips (separate folders)")
    else:
        # Split data randomly
        print("\n--- Splitting Data (random 80/20) ---")
        X = X_train
        y = y_train
        lengths = lengths_train
        indices = np.arange(len(X))
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        lengths_train, lengths_test = lengths[train_idx], lengths[test_idx]
        print(f"\nTrain: {len(X_train)} clips, Test: {len(X_test)} clips (random split)")
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    # Fit scaler on non-padding elements
    all_train_features = []
    for i, length in enumerate(lengths_train):
        all_train_features.append(X_train[i, :length])
    all_train_features = np.vstack(all_train_features)
    scaler.fit(all_train_features)
    
    # Transform
    for i, length in enumerate(lengths_train):
        X_train_scaled[i, :length] = scaler.transform(X_train[i, :length])
    for i, length in enumerate(lengths_test):
        X_test_scaled[i, :length] = scaler.transform(X_test[i, :length])
    
    results = {}
    n_features = X_train.shape[-1]
    
    # Train and evaluate each model
    for model_name in args.models:
        print(f"\n{'='*40}")
        print(f"Training {model_name.upper()}")
        print('='*40)
        
        if model_name == "hmm":
            model = HMMModel(n_states=3)
            model.fit(X_train, y_train, lengths_train)
            preds = model.predict(X_test, lengths_test)
            
        else:
            import torch
            
            if model_name == "cnn":
                model = create_cnn_model(n_features, args.max_len)
            elif model_name == "lstm":
                model = create_lstm_model(n_features, args.max_len)
            elif model_name == "transformer":
                model = create_transformer_model(n_features, args.max_len)
            
            # Split train into train/val
            train_sub_idx, val_idx = train_test_split(
                np.arange(len(X_train_scaled)), test_size=0.2, random_state=42
            )
            
            model, train_losses, val_accs = train_deep_model(
                model,
                X_train_scaled[train_sub_idx], y_train[train_sub_idx], lengths_train[train_sub_idx],
                X_train_scaled[val_idx], y_train[val_idx], lengths_train[val_idx],
                epochs=args.epochs,
                device=args.device
            )
            
            preds = predict_deep_model(model, X_test_scaled, lengths_test, args.device)
        
        # Evaluate
        metrics = evaluate_model(y_test, preds, lengths_test)
        results[model_name.upper()] = metrics
        
        print(f"\n{model_name.upper()} Results:")
        print(f"Accuracy: {metrics['accuracy']:.2%}")
        print(metrics['report'])
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nModel Accuracies:")
    for model_name, metrics in sorted(results.items(), key=lambda x: -x[1]['accuracy']):
        print(f"  {model_name}: {metrics['accuracy']:.2%}")
    
    # Save results
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    plot_comparison(results, save_path=str(output_path / "comparison.png"))
    
    # Save metrics
    metrics_json = {}
    for name, m in results.items():
        metrics_json[name] = {
            'accuracy': float(m['accuracy']),
            'confusion_matrix': m['confusion_matrix'].tolist()
        }
    with open(output_path / "results.json", 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    # Save scaler
    with open(output_path / "scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler to {output_path / 'scaler.pkl'}")
    
    # Save all trained models
    print("\n--- Saving Models ---")
    for model_name in args.models:
        model_save_path = output_path / f"{model_name}_model"
        model_save_path.mkdir(parents=True, exist_ok=True)
        
        if model_name == "hmm":
            # Re-train and save HMM (since we didn't store it in a dict)
            hmm_model = HMMModel(n_states=3)
            hmm_model.fit(X_train, y_train, lengths_train)
            with open(model_save_path / "model.pkl", 'wb') as f:
                pickle.dump(hmm_model, f)
            print(f"  Saved HMM model to {model_save_path}")
        else:
            import torch
            # Re-create and train model to save
            if model_name == "cnn":
                save_model = create_cnn_model(n_features, args.max_len)
            elif model_name == "lstm":
                save_model = create_lstm_model(n_features, args.max_len)
            elif model_name == "transformer":
                save_model = create_transformer_model(n_features, args.max_len)
            
            train_sub_idx, val_idx = train_test_split(
                np.arange(len(X_train_scaled)), test_size=0.2, random_state=42
            )
            save_model, _, _ = train_deep_model(
                save_model,
                X_train_scaled[train_sub_idx], y_train[train_sub_idx], lengths_train[train_sub_idx],
                X_train_scaled[val_idx], y_train[val_idx], lengths_train[val_idx],
                epochs=args.epochs,
                device=args.device
            )
            torch.save(save_model.state_dict(), model_save_path / "model.pt")
            # Also save model config
            config = {
                'n_features': n_features,
                'max_len': args.max_len,
                'n_classes': 3,
                'model_type': model_name
            }
            with open(model_save_path / "config.json", 'w') as f:
                json.dump(config, f, indent=2)
            print(f"  Saved {model_name.upper()} model to {model_save_path}")
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
