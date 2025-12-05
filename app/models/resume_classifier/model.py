"""
Resume Classifier Model - Advanced Architecture
Sentence-BERT embeddings + Multi-Head Attention + Residual Connections
"""

import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Get best available device"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class ClassificationResult:
    """Classification result"""
    category: str
    confidence: float
    top_k: List[Tuple[str, float]]
    probabilities: Dict[str, float]


# ============================================================
# BUILDING BLOCKS
# ============================================================

class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention"""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        return self.out_proj(out)


class FeedForward(nn.Module):
    """Feed-Forward Network with GELU and Dropout"""
    
    def __init__(self, embed_dim: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Transformer Block with Pre-LayerNorm"""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        ff_dim: int = None,
        dropout: float = 0.1
    ):
        super().__init__()
        ff_dim = ff_dim or embed_dim * 4
        
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LayerNorm with residual
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x


class FeaturePyramid(nn.Module):
    """Multi-scale feature extraction"""
    
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.3):
        super().__init__()
        
        # Different scale projections
        self.scale1 = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.scale2 = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )
        
        self.scale3 = nn.Sequential(
            nn.Linear(input_dim, output_dim * 4),
            nn.LayerNorm(output_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 4, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 3, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.scale1(x)
        s2 = self.scale2(x)
        s3 = self.scale3(x)
        
        fused = torch.cat([s1, s2, s3], dim=-1)
        return self.fusion(fused)


# ============================================================
# MAIN MODEL
# ============================================================

class ResumeCategoryNetwork(nn.Module):
    """
    Advanced Resume Classification Network
    
    Architecture:
        SBERT Embeddings (384-dim)
            ↓
        Feature Pyramid (multi-scale extraction)
            ↓
        Transformer Blocks × N (self-attention + FFN)
            ↓
        Global Average + Max Pooling
            ↓
        Classification Head with Residual
            ↓
        Output (num_classes)
    
    Features:
        - Multi-head self-attention
        - Pre-LayerNorm transformer blocks
        - Feature pyramid for multi-scale patterns
        - Residual connections throughout
        - Mixup-ready for training
    """
    
    def __init__(
        self,
        input_dim: int = 384,
        num_classes: int = 24,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 3,
        ff_dim: int = 2048,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # Feature pyramid for initial projection
        self.feature_pyramid = FeaturePyramid(input_dim, embed_dim, dropout)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_ln = nn.LayerNorm(embed_dim)
        
        # Classification head with residual
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),  # *2 for avg+max pooling concat
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        # Residual projection for classification
        self.res_proj = nn.Linear(embed_dim * 2, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) or (batch, seq_len, input_dim)
        """
        # Handle 2D input (add seq dimension)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_dim)
        
        # Feature pyramid
        x = self.feature_pyramid(x)  # (batch, seq_len, embed_dim)
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        x = self.final_ln(x)
        
        # Pooling: concat avg and max
        avg_pool = x.mean(dim=1)
        max_pool = x.max(dim=1).values
        pooled = torch.cat([avg_pool, max_pool], dim=-1)  # (batch, embed_dim * 2)
        
        # Classification with residual
        logits = self.classifier(pooled) + 0.1 * self.res_proj(pooled)
        
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)


# ============================================================
# CLASSIFIER WRAPPER
# ============================================================

class ResumeClassifier:
    """
    Resume Classifier with Sentence-BERT embeddings
    
    Features:
        - Sentence-BERT for semantic embeddings
        - Advanced transformer-based classifier
        - Mixup augmentation during training
        - Label smoothing
        - Gradient accumulation
    """
    
    def __init__(
        self,
        sbert_model: str = "all-MiniLM-L6-v2",
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 3,
        ff_dim: int = 2048,
        dropout: float = 0.3,
        device: Optional[torch.device] = None
    ):
        self.sbert_model_name = sbert_model
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.device = device or get_device()
        
        # Will be initialized
        self.sbert: Optional[SentenceTransformer] = None
        self.model: Optional[ResumeCategoryNetwork] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.input_dim: int = 384
        self.num_classes: int = 0
        self.is_fitted = False
        
        logger.info(f"Using device: {self.device}")
    
    def _encode_texts(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """Encode texts using SBERT"""
        if self.sbert is None:
            logger.info(f"Loading SBERT model: {self.sbert_model_name}")
            self.sbert = SentenceTransformer(self.sbert_model_name)
        
        embeddings = self.sbert.encode(
            texts,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        return embeddings
    
    def fit(
        self,
        texts: List[str],
        labels: List[str],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[str]] = None,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-4,
        patience: int = 10,
        mixup_alpha: float = 0.2,
        label_smoothing: float = 0.1,
        gradient_accumulation: int = 2
    ) -> Dict[str, float]:
        """
        Train the classifier
        
        Args:
            texts: Training texts
            labels: Training labels
            val_texts: Validation texts (optional, will split if not provided)
            val_labels: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            lr: Learning rate
            patience: Early stopping patience
            mixup_alpha: Mixup augmentation alpha (0 to disable)
            label_smoothing: Label smoothing factor
            gradient_accumulation: Gradient accumulation steps
        """
        logger.info(f"Training Resume Classifier with {len(texts)} samples")
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(labels)
        self.num_classes = len(self.label_encoder.classes_)
        
        logger.info(f"Classes ({self.num_classes}): {list(self.label_encoder.classes_)}")
        
        # Encode texts with SBERT
        logger.info("Encoding texts with SBERT...")
        X = self._encode_texts(texts, show_progress=True)
        self.input_dim = X.shape[1]
        
        logger.info(f"Input dimension: {self.input_dim}")
        
        # Validation split
        if val_texts is None:
            from sklearn.model_selection import train_test_split
            X, val_X, y, val_y = train_test_split(
                X, y, test_size=0.1, stratify=y, random_state=42
            )
        else:
            val_X = self._encode_texts(val_texts)
            val_y = self.label_encoder.transform(val_labels)
        
        # Convert to tensors
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        val_X = torch.tensor(val_X, dtype=torch.float32).to(self.device)
        val_y = torch.tensor(val_y, dtype=torch.long).to(self.device)
        
        # Build model
        self.model = ResumeCategoryNetwork(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            ff_dim=self.ff_dim,
            dropout=self.dropout
        ).to(self.device)
        
        # Count parameters
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {num_params:,}")
        
        # Training setup
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Cosine annealing with warmup
        warmup_epochs = min(5, epochs // 10)
        
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Training loop
        best_val_acc = 0.0
        best_state = None
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            
            # Shuffle
            perm = torch.randperm(len(X))
            X_shuffled = X[perm]
            y_shuffled = y[perm]
            
            total_loss = 0.0
            n_batches = 0
            optimizer.zero_grad()
            
            for i in range(0, len(X), batch_size):
                batch_X = X_shuffled[i:i+batch_size].to(self.device)
                batch_y = y_shuffled[i:i+batch_size].to(self.device)
                
                # Mixup augmentation
                if mixup_alpha > 0 and len(batch_X) > 1:
                    lam = np.random.beta(mixup_alpha, mixup_alpha)
                    index = torch.randperm(len(batch_X)).to(self.device)
                    
                    mixed_X = lam * batch_X + (1 - lam) * batch_X[index]
                    
                    outputs = self.model(mixed_X)
                    loss = lam * criterion(outputs, batch_y) + (1 - lam) * criterion(outputs, batch_y[index])
                else:
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                
                # Gradient accumulation
                loss = loss / gradient_accumulation
                loss.backward()
                
                if (i // batch_size + 1) % gradient_accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                
                total_loss += loss.item() * gradient_accumulation
                n_batches += 1
            
            scheduler.step()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(val_X)
                val_preds = val_outputs.argmax(dim=1)
                val_acc = (val_preds == val_y).float().mean().item()
            
            avg_loss = total_loss / n_batches
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Val Acc: {val_acc:.4f} - LR: {scheduler.get_last_lr()[0]:.6f}")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        if best_state:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)
        
        self.is_fitted = True
        logger.info(f"Training complete. Best Val Accuracy: {best_val_acc:.4f}")
        
        return {"val_accuracy": best_val_acc, "epochs_trained": epoch + 1}
    
    def predict(
        self,
        text: str,
        top_k: int = 3
    ) -> ClassificationResult:
        """Predict category for a single resume"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Encode
        embedding = self._encode_texts([text])
        X = torch.tensor(embedding, dtype=torch.float32).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            probs = self.model.predict_proba(X)[0].cpu().numpy()
        
        # Get top-k
        top_indices = np.argsort(probs)[::-1][:top_k]
        top_k_results = [
            (self.label_encoder.classes_[i], float(probs[i]))
            for i in top_indices
        ]
        
        # All probabilities
        all_probs = {
            self.label_encoder.classes_[i]: float(probs[i])
            for i in range(len(probs))
        }
        
        return ClassificationResult(
            category=top_k_results[0][0],
            confidence=top_k_results[0][1],
            top_k=top_k_results,
            probabilities=all_probs
        )
    
    def predict_batch(
        self,
        texts: List[str],
        top_k: int = 3
    ) -> List[ClassificationResult]:
        """Predict categories for multiple resumes"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Encode
        embeddings = self._encode_texts(texts, show_progress=True)
        X = torch.tensor(embeddings, dtype=torch.float32).to(self.device)
        
        # Predict
        self.model.eval()
        results = []
        
        with torch.no_grad():
            probs = self.model.predict_proba(X).cpu().numpy()
        
        for prob in probs:
            top_indices = np.argsort(prob)[::-1][:top_k]
            top_k_results = [
                (self.label_encoder.classes_[i], float(prob[i]))
                for i in top_indices
            ]
            
            all_probs = {
                self.label_encoder.classes_[i]: float(prob[i])
                for i in range(len(prob))
            }
            
            results.append(ClassificationResult(
                category=top_k_results[0][0],
                confidence=top_k_results[0][1],
                top_k=top_k_results,
                probabilities=all_probs
            ))
        
        return results
    
    def save(self, path: Union[str, Path]) -> None:
        """Save model to directory"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save PyTorch model
        torch.save(self.model.state_dict(), path / "model.pt")
        
        # Save label encoder
        with open(path / "label_encoder.pkl", "wb") as f:
            pickle.dump(self.label_encoder, f)
        
        # Save config
        config = {
            "sbert_model": self.sbert_model_name,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout,
            "input_dim": self.input_dim,
            "num_classes": self.num_classes
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path], device: Optional[torch.device] = None) -> "ResumeClassifier":
        """Load model from directory"""
        path = Path(path)
        
        # Load config
        with open(path / "config.json", "r") as f:
            config = json.load(f)
        
        # Create instance
        classifier = cls(
            sbert_model=config["sbert_model"],
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            ff_dim=config["ff_dim"],
            dropout=config["dropout"],
            device=device
        )
        
        classifier.input_dim = config["input_dim"]
        classifier.num_classes = config["num_classes"]
        
        # Load label encoder
        with open(path / "label_encoder.pkl", "rb") as f:
            classifier.label_encoder = pickle.load(f)
        
        # Build and load model
        classifier.model = ResumeCategoryNetwork(
            input_dim=classifier.input_dim,
            num_classes=classifier.num_classes,
            embed_dim=classifier.embed_dim,
            num_heads=classifier.num_heads,
            num_layers=classifier.num_layers,
            ff_dim=classifier.ff_dim,
            dropout=classifier.dropout
        ).to(classifier.device)
        
        classifier.model.load_state_dict(
            torch.load(path / "model.pt", map_location=classifier.device)
        )
        classifier.model.eval()
        
        classifier.is_fitted = True
        logger.info(f"Model loaded from {path}")
        
        return classifier