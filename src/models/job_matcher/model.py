"""
Job Matcher Model
Siamese Network with shared encoder for resume-job matching
Trains on Resume.csv using contrastive learning
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Get best available device"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class ResumeEncoder(nn.Module):
    """
    Fast encoder for resumes and jobs
    
    Architecture:
        SBERT (frozen) -> [384] -> Linear(512) -> BN -> GELU -> Dropout
                                -> Linear(256) -> BN -> GELU -> Dropout  
                                -> Linear(128) -> L2 Normalize
    
    Output: 128-dim normalized embedding (fast cosine similarity)
    """
    
    def __init__(
        self,
        sbert_model: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 128,
        hidden_dims: List[int] = [512, 256],
        dropout: float = 0.3,
        freeze_sbert: bool = True
    ):
        super().__init__()
        
        self.sbert_model_name = sbert_model
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout
        self.freeze_sbert = freeze_sbert
        
        # SBERT encoder (384-dim for MiniLM)
        self.sbert = SentenceTransformer(sbert_model)
        self.sbert_dim = self.sbert.get_sentence_embedding_dimension()
        
        if freeze_sbert:
            for param in self.sbert.parameters():
                param.requires_grad = False
        
        # Projection head
        layers = []
        prev_dim = self.sbert_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, embedding_dim))
        
        self.projection = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Encode texts using SBERT (returns on CPU)"""
        with torch.no_grad():
            embeddings = self.sbert.encode(
                texts, 
                convert_to_tensor=True,
                show_progress_bar=False
            )
        return embeddings
    
    def forward(self, sbert_embeddings: torch.Tensor) -> torch.Tensor:
        """Project SBERT embeddings through MLP head"""
        projected = self.projection(sbert_embeddings)
        # L2 normalize for cosine similarity
        return F.normalize(projected, p=2, dim=1)
    
    def encode(self, texts: List[str], device: torch.device = None) -> np.ndarray:
        """
        Full encoding pipeline: text -> SBERT -> projection -> normalized embedding
        
        This is the method to use for generating embeddings for DB storage
        """
        self.eval()
        device = device or next(self.projection.parameters()).device
        
        with torch.no_grad():
            sbert_emb = self.encode_texts(texts).to(device)
            projected = self.forward(sbert_emb)
        
        return projected.cpu().numpy()


class SiameseNetwork(nn.Module):
    """
    Siamese Network for contrastive learning
    
    Uses shared ResumeEncoder for both inputs
    Trained with contrastive loss on resume pairs
    """
    
    def __init__(
        self,
        encoder: ResumeEncoder,
        margin: float = 0.5
    ):
        super().__init__()
        self.encoder = encoder
        self.margin = margin
    
    def forward(
        self, 
        emb1: torch.Tensor, 
        emb2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for pair of SBERT embeddings
        
        Returns:
            (projected_emb1, projected_emb2)
        """
        proj1 = self.encoder(emb1)
        proj2 = self.encoder(emb2)
        return proj1, proj2
    
    def contrastive_loss(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Contrastive loss
        
        Args:
            emb1, emb2: Projected embeddings
            labels: 1 = same category (similar), 0 = different category (dissimilar)
        """
        # Cosine distance (1 - cosine_similarity)
        cosine_sim = F.cosine_similarity(emb1, emb2)
        distance = 1 - cosine_sim
        
        # Contrastive loss
        # Same category: minimize distance
        # Different category: maximize distance (up to margin)
        loss_similar = labels * distance.pow(2)
        loss_dissimilar = (1 - labels) * F.relu(self.margin - distance).pow(2)
        
        return (loss_similar + loss_dissimilar).mean()


class JobMatcher:
    """
    Job Matcher - wrapper for training and inference
    
    Training: Contrastive learning on resume pairs (same/different category)
    Inference: Generate embeddings for resumes/jobs, compute cosine similarity
    """
    
    def __init__(
        self,
        sbert_model: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 128,
        hidden_dims: List[int] = [512, 256],
        dropout: float = 0.3,
        margin: float = 0.5,
        device: Optional[torch.device] = None
    ):
        self.sbert_model = sbert_model
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.margin = margin
        self.device = device or get_device()
        
        self.encoder: Optional[ResumeEncoder] = None
        self.siamese: Optional[SiameseNetwork] = None
        self.is_fitted = False
        
        logger.info(f"Using device: {self.device}")
    
    def _build_model(self):
        """Build encoder and siamese network"""
        self.encoder = ResumeEncoder(
            sbert_model=self.sbert_model,
            embedding_dim=self.embedding_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            freeze_sbert=True
        ).to(self.device)
        
        self.siamese = SiameseNetwork(
            encoder=self.encoder,
            margin=self.margin
        ).to(self.device)
    
    def fit(
        self,
        texts: List[str],
        categories: List[str],
        epochs: int = 30,
        batch_size: int = 64,
        lr: float = 0.001,
        pairs_per_epoch: int = 10000,
        patience: int = 7
    ) -> Dict[str, float]:
        """
        Train using contrastive learning on resume pairs
        
        Args:
            texts: Resume texts
            categories: Category labels
            epochs: Training epochs
            batch_size: Batch size
            lr: Learning rate
            pairs_per_epoch: Number of pairs to sample per epoch
            patience: Early stopping patience
        """
        logger.info(f"Training Job Matcher with {len(texts)} resumes")
        
        # Build model
        self._build_model()
        
        # Pre-encode all texts with SBERT (one time, fast)
        logger.info("Pre-encoding all resumes with SBERT...")
        with torch.no_grad():
            all_sbert_embeddings = self.encoder.encode_texts(texts)
        
        # Create category to indices mapping
        category_to_indices = {}
        for idx, cat in enumerate(categories):
            if cat not in category_to_indices:
                category_to_indices[cat] = []
            category_to_indices[cat].append(idx)
        
        all_categories = list(category_to_indices.keys())
        logger.info(f"Categories: {len(all_categories)}")
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.encoder.projection.parameters(),
            lr=lr,
            weight_decay=0.01
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(epochs):
            self.siamese.train()
            
            # Sample pairs
            pairs, labels = self._sample_pairs(
                category_to_indices, all_categories, pairs_per_epoch
            )
            
            total_loss = 0
            n_batches = 0
            
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i+batch_size]
                batch_labels = labels[i:i+batch_size]
                
                idx1 = [p[0] for p in batch_pairs]
                idx2 = [p[1] for p in batch_pairs]
                
                emb1 = all_sbert_embeddings[idx1].to(self.device)
                emb2 = all_sbert_embeddings[idx2].to(self.device)
                batch_labels_t = torch.tensor(batch_labels, dtype=torch.float32).to(self.device)
                
                optimizer.zero_grad()
                
                proj1, proj2 = self.siamese(emb1, emb2)
                loss = self.siamese.contrastive_loss(proj1, proj2, batch_labels_t)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.encoder.projection.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            scheduler.step()
            avg_loss = total_loss / n_batches
            
            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = self.encoder.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        if best_state:
            self.encoder.load_state_dict(best_state)
        
        self.is_fitted = True
        logger.info(f"Training complete. Best Loss: {best_loss:.4f}")
        
        return {"best_loss": best_loss, "epochs_trained": epoch + 1}
    
    def _sample_pairs(
        self,
        category_to_indices: Dict[str, List[int]],
        all_categories: List[str],
        n_pairs: int
    ) -> Tuple[List[Tuple[int, int]], List[int]]:
        """Sample positive and negative pairs"""
        pairs = []
        labels = []
        
        n_positive = n_pairs // 2
        n_negative = n_pairs - n_positive
        
        # Positive pairs (same category)
        for _ in range(n_positive):
            cat = np.random.choice(all_categories)
            indices = category_to_indices[cat]
            if len(indices) >= 2:
                idx1, idx2 = np.random.choice(indices, size=2, replace=False)
                pairs.append((idx1, idx2))
                labels.append(1)
        
        # Negative pairs (different categories)
        for _ in range(n_negative):
            cat1, cat2 = np.random.choice(all_categories, size=2, replace=False)
            idx1 = np.random.choice(category_to_indices[cat1])
            idx2 = np.random.choice(category_to_indices[cat2])
            pairs.append((idx1, idx2))
            labels.append(0)
        
        # Shuffle
        combined = list(zip(pairs, labels))
        np.random.shuffle(combined)
        pairs, labels = zip(*combined)
        
        return list(pairs), list(labels)
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to embeddings
        
        Use this to:
        1. Encode resumes for matching
        2. Encode job descriptions for DB storage
        
        Returns:
            (N, embedding_dim) numpy array
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        return self.encoder.encode(texts, device=self.device)
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode single text"""
        return self.encode([text])[0]
    
    def compute_similarity(
        self,
        resume_embedding: np.ndarray,
        job_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between resume and jobs
        
        Args:
            resume_embedding: (embedding_dim,) single resume
            job_embeddings: (N, embedding_dim) job embeddings from DB
        
        Returns:
            (N,) similarity scores (0-1)
        """
        # Normalize (should already be normalized, but just in case)
        resume_norm = resume_embedding / (np.linalg.norm(resume_embedding) + 1e-8)
        job_norms = job_embeddings / (np.linalg.norm(job_embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Cosine similarity
        similarities = np.dot(job_norms, resume_norm)
        
        # Clip to [0, 1]
        return np.clip(similarities, 0, 1)
    
    def match(
        self,
        resume_text: str,
        job_embeddings: np.ndarray,
        job_ids: List[str],
        top_k: int = 10
    ) -> List[Dict]:
        """
        Match resume against job embeddings
        
        Args:
            resume_text: Resume text
            job_embeddings: Pre-computed job embeddings from DB
            job_ids: Job IDs corresponding to embeddings
            top_k: Number of top matches to return
        
        Returns:
            List of {job_id, score} sorted by score descending
        """
        resume_emb = self.encode_single(resume_text)
        scores = self.compute_similarity(resume_emb, job_embeddings)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = [
            {"job_id": job_ids[i], "score": float(scores[i])}
            for i in top_indices
        ]
        
        return results
    
    def save(self, path: Union[str, Path]) -> None:
        """Save model"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save encoder weights
        torch.save(self.encoder.state_dict(), path / "encoder.pt")
        
        # Save config
        config = {
            "sbert_model": self.sbert_model,
            "embedding_dim": self.embedding_dim,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "margin": self.margin
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path], device: Optional[torch.device] = None) -> "JobMatcher":
        """Load model"""
        path = Path(path)
        
        with open(path / "config.json", "r") as f:
            config = json.load(f)
        
        matcher = cls(**config, device=device)
        matcher._build_model()
        
        matcher.encoder.load_state_dict(
            torch.load(path / "encoder.pt", map_location=matcher.device)
        )
        matcher.encoder.eval()
        matcher.is_fitted = True
        
        logger.info(f"Model loaded from {path}")
        return matcher