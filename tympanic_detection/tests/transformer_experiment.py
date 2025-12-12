"""
Transformer Experiment Script for Deformation Classification

This script allows experimenting with various anti-overfitting techniques:
1. Dropout rate control
2. Label smoothing
3. Weight decay (L2 regularization)
4. Model capacity (layers, d_model, heads)
5. Data augmentation (noise, frame masking)
6. Early stopping on external validation set
7. Gradient clipping

Usage:
    python transformer_experiment.py \
        --clips /path/to/train \
        --labels /path/to/train/labels.txt \
        --val_clips /path/to/val \
        --val_labels /path/to/val/labels.txt \
        --output ./transformer_exp \
        --dropout 0.5 \
        --label_smoothing 0.1 \
        --weight_decay 0.01 \
        --noise_std 0.1 \
        --d_model 32 \
        --n_layers 2 \
        --n_heads 4
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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
_project_root = Path(__file__).parent.parent.absolute()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Import from existing scripts
from supervised_classification import (
    load_labels,
    convert_binary_to_states,
    extract_features_for_clip,
)
from point_distance_analysis import build_feature_matrix


# ===== Configurable Transformer Model =====

class ConfigurableTransformer(nn.Module):
    """Transformer model with configurable architecture and regularization."""
    
    def __init__(
        self,
        n_features: int,
        max_len: int,
        n_classes: int = 3,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 64,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        
        # Input embedding
        self.embedding = nn.Linear(n_features, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.1)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output layer with dropout
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, n_classes)
        
        # Store config
        self.config = {
            'n_features': n_features,
            'max_len': max_len,
            'n_classes': n_classes,
            'd_model': d_model,
            'n_heads': n_heads,
            'n_layers': n_layers,
            'dim_feedforward': dim_feedforward,
            'dropout': dropout
        }
    
    def forward(self, x):
        # x: [batch, seq_len, features]
        x = self.embedding(x)  # [batch, seq_len, d_model]
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        x = self.dropout(x)
        x = self.fc(x)  # [batch, seq_len, n_classes]
        return x


# ===== Data Augmentation =====

class DataAugmentation:
    """Data augmentation for time-series features."""
    
    def __init__(
        self,
        noise_std: float = 0.0,
        frame_mask_prob: float = 0.0,
        time_shift_range: int = 0
    ):
        self.noise_std = noise_std
        self.frame_mask_prob = frame_mask_prob
        self.time_shift_range = time_shift_range
    
    def __call__(self, X: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        """Apply augmentation to batch of sequences."""
        X_aug = X.copy()
        
        for i in range(len(X)):
            length = lengths[i]
            
            # Add Gaussian noise
            if self.noise_std > 0:
                noise = np.random.randn(length, X.shape[-1]) * self.noise_std
                X_aug[i, :length] += noise
            
            # Random frame masking (set to zero)
            if self.frame_mask_prob > 0:
                mask = np.random.random(length) < self.frame_mask_prob
                X_aug[i, :length][mask] = 0
        
        return X_aug


# ===== Label Smoothing Loss =====

class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with label smoothing."""
    
    def __init__(self, smoothing: float = 0.0, reduction: str = 'none'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = pred.size(-1)
        
        # One-hot encoding with smoothing
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(-1, target.unsqueeze(-1), 1.0 - self.smoothing)
        
        log_probs = torch.log_softmax(pred, dim=-1)
        loss = -torch.sum(true_dist * log_probs, dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Focuses more on hard-to-classify samples (like 'Deforming' class).
    gamma=0 is equivalent to CrossEntropy, higher gamma focuses more on hard samples.
    """
    
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None, reduction: str = 'none'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # Class weights
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred: [N, C], target: [N]
        log_probs = torch.log_softmax(pred, dim=-1)
        probs = torch.exp(log_probs)
        
        # Get the probability of the correct class
        target_probs = probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        
        # Focal weight: (1 - p)^gamma
        focal_weight = (1 - target_probs) ** self.gamma
        
        # Cross entropy loss
        ce_loss = -log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        
        # Apply focal weight
        loss = focal_weight * ce_loss
        
        # Apply class weights if provided
        if self.alpha is not None:
            alpha_weight = self.alpha.gather(0, target)
            loss = alpha_weight * loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# ===== Data Preparation =====

def prepare_sequence_data(
    clips_folder: str,
    labels_file: str,
    grid_size: int = 20,
    radius: int = 2,
    device: str = "cuda",
    max_len: int = 50
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Prepare sequence data for time-series models."""
    
    labels_by_clip = load_labels(labels_file)
    print(f"Found labels for {len(labels_by_clip)} clips")
    
    all_X = []
    all_y = []
    all_lengths = []
    clip_names = []
    
    clips_path = Path(clips_folder)
    n_features = 10  # Number of features from build_feature_matrix
    
    # Load CoTracker model once (use local repository)
    print("Loading CoTracker model from local repository...")
    # Use the local co-tracker repository directly
    local_repo_path = Path(__file__).parent.parent.parent.absolute()  # /home/lzq/TM_project/co-tracker
    cotracker_model = torch.hub.load(str(local_repo_path), "cotracker3_offline", source='local', trust_repo=True)
    cotracker_model = cotracker_model.to(device)
    
    for clip_name, frame_labels in labels_by_clip.items():
        clip_path = clips_path / clip_name
        
        if not clip_path.exists():
            continue
        
        print(f"Processing {clip_name}...", end=" ")
        
        try:
            # Extract features (pass pre-loaded model)
            X, features = extract_features_for_clip(
                str(clip_path), grid_size, radius, device, model=cotracker_model
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
    
    X = np.stack(all_X)
    y = np.stack(all_y)
    lengths = np.array(all_lengths)
    
    print(f"\nTotal: {len(clip_names)} clips, shape: {X.shape}")
    
    return X, y, lengths, clip_names


# ===== Training =====

def train_transformer(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    lengths_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    lengths_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    label_smoothing: float = 0.0,
    focal_loss: bool = False,
    focal_gamma: float = 2.0,
    grad_clip: float = 1.0,
    augmentation: Optional[DataAugmentation] = None,
    device: str = "cuda",
    patience: int = 15
) -> Tuple[nn.Module, List[float], List[float], List[float]]:
    """Train transformer with configurable regularization."""
    
    model = model.to(device)
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)
    
    # Create masks for valid positions
    mask_train = torch.zeros_like(y_train_t, dtype=torch.bool)
    mask_val = torch.zeros_like(y_val_t, dtype=torch.bool)
    for i, length in enumerate(lengths_train):
        mask_train[i, :length] = True
    for i, length in enumerate(lengths_val):
        mask_val[i, :length] = True
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Loss function
    if focal_loss:
        # Class weights for imbalanced data (Static=0, Deforming=1, Peak=2)
        # Give higher weight to Deforming class which is harder to classify
        alpha = torch.tensor([1.0, 2.0, 1.0]).to(device)  # Deforming gets 2x weight
        criterion = FocalLoss(gamma=focal_gamma, alpha=alpha, reduction='none')
        loss_type = f"FocalLoss(gamma={focal_gamma})"
    elif label_smoothing > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing, reduction='none')
        loss_type = f"LabelSmoothing({label_smoothing})"
    else:
        criterion = nn.CrossEntropyLoss(reduction='none')
        loss_type = "CrossEntropy"
    
    best_val_acc = 0
    best_model_state = None
    patience_counter = 0
    
    train_losses = []
    train_accs = []
    val_accs = []
    
    print(f"\nTraining with: dropout={model.config['dropout']}, weight_decay={weight_decay}, "
          f"loss={loss_type}, patience={patience}")
    
    for epoch in range(epochs):
        model.train()
        
        # Apply data augmentation
        if augmentation is not None:
            X_aug = augmentation(X_train, lengths_train)
            X_train_t = torch.FloatTensor(X_aug).to(device)
        
        # Training step
        optimizer.zero_grad()
        outputs = model(X_train_t)
        
        # Flatten for loss
        outputs_flat = outputs.view(-1, 3)
        y_flat = y_train_t.view(-1)
        mask_flat = mask_train.view(-1)
        
        loss = criterion(outputs_flat, y_flat)
        loss = (loss * mask_flat.float()).sum() / mask_flat.sum()
        
        loss.backward()
        
        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # Training accuracy
        with torch.no_grad():
            train_preds = outputs.argmax(dim=-1)
            train_correct = ((train_preds == y_train_t) & mask_train).sum().item()
            train_total = mask_train.sum().item()
            train_acc = train_correct / train_total
            train_accs.append(train_acc)
        
        # Validation
        model.eval()
        with torch.no_grad():
            outputs_val = model(X_val_t)
            preds = outputs_val.argmax(dim=-1)
            
            correct = ((preds == y_val_t) & mask_val).sum().item()
            total = mask_val.sum().item()
            val_acc = correct / total
            val_accs.append(val_acc)
            
            # Update scheduler
            scheduler.step(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Loss={loss.item():.4f}, "
                  f"Train Acc={train_acc:.2%}, Val Acc={val_acc:.2%}")
        
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model = model.to(device)
    
    return model, train_losses, train_accs, val_accs


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, lengths: np.ndarray) -> dict:
    """Evaluate predictions, excluding padding."""
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


def predict(model: nn.Module, X: np.ndarray, lengths: np.ndarray, device: str = "cuda"):
    """Predict with trained model."""
    model = model.to(device)
    model.eval()
    
    X_t = torch.FloatTensor(X).to(device)
    
    with torch.no_grad():
        outputs = model(X_t)
        preds = outputs.argmax(dim=-1).cpu().numpy()
    
    return preds


def plot_training_curves(
    train_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: str
):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curve
    axes[0].plot(train_losses, label='Train Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[1].plot(train_accs, label='Train Acc')
    axes[1].plot(val_accs, label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training vs Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved training curves to {save_path}")
    plt.close()


# ===== Main =====

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Transformer Experiment with Anti-Overfitting Techniques",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Baseline (no regularization)
  python transformer_experiment.py --clips ... --output ./exp_baseline

  # High dropout
  python transformer_experiment.py --clips ... --output ./exp_dropout --dropout 0.5

  # Label smoothing
  python transformer_experiment.py --clips ... --output ./exp_smooth --label_smoothing 0.1

  # Weight decay
  python transformer_experiment.py --clips ... --output ./exp_wd --weight_decay 0.01

  # Data augmentation
  python transformer_experiment.py --clips ... --output ./exp_aug --noise_std 0.1 --frame_mask_prob 0.1

  # Smaller model
  python transformer_experiment.py --clips ... --output ./exp_small --d_model 16 --n_layers 1

  # Everything combined
  python transformer_experiment.py --clips ... --output ./exp_all \\
      --dropout 0.5 --label_smoothing 0.1 --weight_decay 0.01 \\
      --noise_std 0.1 --d_model 24 --n_layers 1
        """
    )
    
    # Data arguments
    parser.add_argument("--clips", type=str, required=True,
                        help="Folder containing training clip_XXXX subfolders")
    parser.add_argument("--labels", type=str, required=True,
                        help="Path to training labels.txt")
    parser.add_argument("--val_clips", type=str, required=True,
                        help="Folder containing validation clip_XXXX subfolders")
    parser.add_argument("--val_labels", type=str, required=True,
                        help="Path to validation labels.txt")
    parser.add_argument("--output", type=str, required=True,
                        help="Output folder for results")
    
    # Data processing
    parser.add_argument("--grid_size", type=int, default=20)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_len", type=int, default=50)
    
    # Model architecture
    parser.add_argument("--d_model", type=int, default=32,
                        help="Model dimension (default: 32)")
    parser.add_argument("--n_heads", type=int, default=4,
                        help="Number of attention heads (default: 4)")
    parser.add_argument("--n_layers", type=int, default=2,
                        help="Number of transformer layers (default: 2)")
    parser.add_argument("--dim_feedforward", type=int, default=64,
                        help="Feedforward dimension (default: 64)")
    
    # Regularization
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate (default: 0.3)")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight decay / L2 regularization (default: 0.0)")
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="Label smoothing factor (default: 0.0)")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping norm (default: 1.0)")
    parser.add_argument("--focal_loss", action="store_true",
                        help="Use Focal Loss for class imbalance (focuses on hard samples)")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Focal Loss gamma parameter (default: 2.0, higher = more focus on hard samples)")
    
    # Data augmentation
    parser.add_argument("--noise_std", type=float, default=0.0,
                        help="Gaussian noise std for augmentation (default: 0.0)")
    parser.add_argument("--frame_mask_prob", type=float, default=0.0,
                        help="Probability of masking a frame (default: 0.0)")
    
    # Training
    parser.add_argument("--epochs", type=int, default=100,
                        help="Maximum training epochs (default: 100)")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience (default: 15)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate (default: 0.001)")
    parser.add_argument("--batch_size", type=int, default=16)
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("TRANSFORMER EXPERIMENT")
    print("=" * 60)
    
    # Print configuration
    print("\n--- Configuration ---")
    print(f"Model: d_model={args.d_model}, n_heads={args.n_heads}, n_layers={args.n_layers}")
    print(f"Regularization: dropout={args.dropout}, weight_decay={args.weight_decay}, "
          f"label_smoothing={args.label_smoothing}")
    print(f"Augmentation: noise_std={args.noise_std}, frame_mask_prob={args.frame_mask_prob}")
    print(f"Training: epochs={args.epochs}, patience={args.patience}, lr={args.lr}")
    
    # Prepare training data
    print("\n--- Preparing Training Data ---")
    X_train, y_train, lengths_train, _ = prepare_sequence_data(
        args.clips, args.labels,
        args.grid_size, args.radius, args.device, args.max_len
    )
    
    # Prepare validation data
    print("\n--- Preparing Validation Data ---")
    X_val, y_val, lengths_val, _ = prepare_sequence_data(
        args.val_clips, args.val_labels,
        args.grid_size, args.radius, args.device, args.max_len
    )
    
    print(f"\nTrain: {len(X_train)} clips, Val: {len(X_val)} clips")
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    
    # Fit scaler on non-padding elements
    all_train_features = []
    for i, length in enumerate(lengths_train):
        all_train_features.append(X_train[i, :length])
    all_train_features = np.vstack(all_train_features)
    scaler.fit(all_train_features)
    
    # Transform
    for i, length in enumerate(lengths_train):
        X_train_scaled[i, :length] = scaler.transform(X_train[i, :length])
    for i, length in enumerate(lengths_val):
        X_val_scaled[i, :length] = scaler.transform(X_val[i, :length])
    
    n_features = X_train.shape[-1]
    
    # Create model
    print("\n--- Creating Model ---")
    model = ConfigurableTransformer(
        n_features=n_features,
        max_len=args.max_len,
        n_classes=3,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Create augmentation
    augmentation = None
    if args.noise_std > 0 or args.frame_mask_prob > 0:
        augmentation = DataAugmentation(
            noise_std=args.noise_std,
            frame_mask_prob=args.frame_mask_prob
        )
        print(f"Data augmentation enabled: noise_std={args.noise_std}, "
              f"frame_mask_prob={args.frame_mask_prob}")
    
    # Train
    print("\n--- Training ---")
    model, train_losses, train_accs, val_accs = train_transformer(
        model,
        X_train_scaled, y_train, lengths_train,
        X_val_scaled, y_val, lengths_val,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        focal_loss=args.focal_loss,
        focal_gamma=args.focal_gamma,
        grad_clip=args.grad_clip,
        augmentation=augmentation,
        device=args.device,
        patience=args.patience
    )
    
    # Evaluate on validation set
    print("\n--- Evaluation on Validation Set ---")
    preds = predict(model, X_val_scaled, lengths_val, args.device)
    metrics = evaluate_model(y_val, preds, lengths_val)
    
    print(f"\nValidation Accuracy: {metrics['accuracy']:.2%}")
    print(metrics['report'])
    
    # Calculate overfitting gap
    final_train_acc = train_accs[-1] if train_accs else 0
    final_val_acc = val_accs[-1] if val_accs else 0
    overfitting_gap = final_train_acc - metrics['accuracy']
    
    print(f"\n--- Overfitting Analysis ---")
    print(f"Final Training Accuracy: {final_train_acc:.2%}")
    print(f"Final Validation Accuracy (during training): {final_val_acc:.2%}")
    print(f"Test Accuracy (external val set): {metrics['accuracy']:.2%}")
    print(f"Overfitting Gap (Train - Test): {overfitting_gap:.2%}")
    
    # Save results
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), output_path / "model.pt")
    
    # Save config
    config = {
        **model.config,
        'weight_decay': args.weight_decay,
        'label_smoothing': args.label_smoothing,
        'focal_loss': args.focal_loss,
        'focal_gamma': args.focal_gamma,
        'noise_std': args.noise_std,
        'frame_mask_prob': args.frame_mask_prob,
        'epochs_trained': len(train_losses),
        'final_train_acc': final_train_acc,
        'final_val_acc': metrics['accuracy'],
        'overfitting_gap': overfitting_gap
    }
    with open(output_path / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save scaler
    with open(output_path / "scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save training curves
    plot_training_curves(train_losses, train_accs, val_accs,
                        str(output_path / "training_curves.png"))
    
    # Save metrics
    metrics_json = {
        'accuracy': float(metrics['accuracy']),
        'confusion_matrix': metrics['confusion_matrix'].tolist(),
        'overfitting_gap': overfitting_gap
    }
    with open(output_path / "metrics.json", 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    print("\nFiles saved:")
    print(f"  - model.pt (model weights)")
    print(f"  - config.json (experiment configuration)")
    print(f"  - scaler.pkl (feature scaler)")
    print(f"  - training_curves.png (loss and accuracy curves)")
    print(f"  - metrics.json (evaluation metrics)")


if __name__ == "__main__":
    main()
