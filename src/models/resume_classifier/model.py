"""
Resume Classifier Model
PyTorch Feed-Forward Neural Network for job category classification
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Get best available device (MPS for Mac, CUDA for NVIDIA, else CPU)"""
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


class ResumeCategoryNetwork(nn.Module):
    """
    Improved Feed-Forward Neural Network for resume classification
    
    Architecture:
        Input -> Linear(768) -> BN -> GELU -> Dropout
              -> Linear(384) -> BN -> GELU -> Dropout
              -> Linear(192) -> BN -> GELU -> Dropout
              -> Linear(num_classes)
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: List[int] = [768, 384, 192],
        dropout: float = 0.4
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),  # Better than ReLU
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)


class ResumeClassifier:
    """
    Resume Category Classifier
    
    Uses TF-IDF + optional skill encoding + PyTorch NN
    """
    
    def __init__(
        self,
        tfidf_max_features: int = 7000,
        hidden_dims: List[int] = [768, 384, 192],
        dropout: float = 0.4,
        device: Optional[torch.device] = None
    ):
        self.tfidf_max_features = tfidf_max_features
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.device = device or get_device()
        
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.char_vectorizer: Optional[TfidfVectorizer] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.skill_encoder: Optional[MultiLabelBinarizer] = None
        self.model: Optional[ResumeCategoryNetwork] = None
        
        self.num_classes = 0
        self.input_dim = 0
        self.is_fitted = False
        
        logger.info(f"Using device: {self.device}")
    
    def _prepare_features(
        self, 
        texts: List[str], 
        skills: Optional[List[List[str]]] = None,
        fit: bool = False
    ) -> torch.Tensor:
        """Convert texts and skills to feature tensor"""
        
        # Word TF-IDF features
        if fit:
            self.vectorizer = TfidfVectorizer(
                max_features=self.tfidf_max_features,
                ngram_range=(1, 3),  # Include trigrams
                stop_words="english",
                min_df=2,
                max_df=0.9,
                sublinear_tf=True
            )
            # Character n-gram vectorizer (captures technical terms better)
            self.char_vectorizer = TfidfVectorizer(
                max_features=2000,
                analyzer='char_wb',
                ngram_range=(3, 5),
                min_df=2,
                max_df=0.9
            )
            word_features = self.vectorizer.fit_transform(texts).toarray()
            char_features = self.char_vectorizer.fit_transform(texts).toarray()
        else:
            word_features = self.vectorizer.transform(texts).toarray()
            char_features = self.char_vectorizer.transform(texts).toarray()
        
        features = np.hstack([word_features, char_features])
        
        # Skill features (optional)
        if skills is not None:
            if fit:
                self.skill_encoder = MultiLabelBinarizer(sparse_output=False)
                skill_features = self.skill_encoder.fit_transform(skills)
            else:
                skill_features = self.skill_encoder.transform(skills)
            
            features = np.hstack([features, skill_features])
        
        return torch.tensor(features, dtype=torch.float32)
    
    def fit(
        self,
        texts: List[str],
        labels: List[str],
        skills: Optional[List[List[str]]] = None,
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[str]] = None,
        val_skills: Optional[List[List[str]]] = None,
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 0.001,
        patience: int = 10
    ) -> Dict[str, float]:
        """
        Train the classifier
        
        Args:
            texts: Training texts
            labels: Training labels
            skills: Optional skill lists per resume
            val_*: Validation data
            epochs: Number of epochs
            batch_size: Batch size
            lr: Learning rate
            patience: Early stopping patience
        """
        logger.info(f"Training Resume Classifier with {len(texts)} samples")
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(labels)
        self.num_classes = len(self.label_encoder.classes_)
        
        logger.info(f"Classes ({self.num_classes}): {list(self.label_encoder.classes_)}")
        
        # Prepare features
        X = self._prepare_features(texts, skills, fit=True)
        self.input_dim = X.shape[1]
        
        logger.info(f"Input dimension: {self.input_dim}")
        
        # Validation data
        if val_texts:
            val_y = self.label_encoder.transform(val_labels)
            val_X = self._prepare_features(val_texts, val_skills, fit=False)
        else:
            # Split from training
            from sklearn.model_selection import train_test_split
            X, val_X, y, val_y = train_test_split(
                X.numpy(), y, test_size=0.1, stratify=y, random_state=42
            )
            X = torch.tensor(X, dtype=torch.float32)
            val_X = torch.tensor(val_X, dtype=torch.float32)
        
        # Create model
        self.model = ResumeCategoryNetwork(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout
        ).to(self.device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, epochs=epochs, 
            steps_per_epoch=len(X) // batch_size + 1
        )
        
        # Convert to tensors
        y = torch.tensor(y, dtype=torch.long)
        val_y = torch.tensor(val_y, dtype=torch.long)
        
        # Training loop
        best_val_acc = 0
        patience_counter = 0
        best_state = None
        
        for epoch in range(epochs):
            self.model.train()
            
            # Shuffle
            perm = torch.randperm(len(X))
            X_shuffled = X[perm]
            y_shuffled = y[perm]
            
            total_loss = 0
            n_batches = 0
            
            for i in range(0, len(X), batch_size):
                batch_X = X_shuffled[i:i+batch_size].to(self.device)
                batch_y = y_shuffled[i:i+batch_size].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_X_device = val_X.to(self.device)
                val_outputs = self.model(val_X_device)
                val_preds = val_outputs.argmax(dim=1).cpu()
                val_acc = (val_preds == val_y).float().mean().item()
            
            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/n_batches:.4f} - Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        if best_state:
            self.model.load_state_dict(best_state)
        
        self.is_fitted = True
        
        logger.info(f"Training complete. Best Val Accuracy: {best_val_acc:.4f}")
        
        return {"val_accuracy": best_val_acc, "epochs_trained": epoch + 1}
    
    def predict(self, text: str, skills: Optional[List[str]] = None, top_k: int = 3) -> ClassificationResult:
        """Predict category for a single resume"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        skill_input = [skills] if skills else None
        X = self._prepare_features([text], skill_input, fit=False)
        
        self.model.eval()
        with torch.no_grad():
            X_device = X.to(self.device)
            probs = self.model.predict_proba(X_device)[0].cpu().numpy()
        
        # Get predictions
        top_indices = np.argsort(probs)[::-1][:top_k]
        top_categories = [
            (self.label_encoder.classes_[i], float(probs[i])) 
            for i in top_indices
        ]
        
        best_idx = top_indices[0]
        
        return ClassificationResult(
            category=self.label_encoder.classes_[best_idx],
            confidence=float(probs[best_idx]),
            top_k=top_categories,
            probabilities={
                self.label_encoder.classes_[i]: float(probs[i]) 
                for i in range(len(probs))
            }
        )
    
    def predict_batch(
        self, 
        texts: List[str], 
        skills: Optional[List[List[str]]] = None,
        top_k: int = 3
    ) -> List[ClassificationResult]:
        """Predict categories for multiple resumes"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        X = self._prepare_features(texts, skills, fit=False)
        
        self.model.eval()
        with torch.no_grad():
            X_device = X.to(self.device)
            probs = self.model.predict_proba(X_device).cpu().numpy()
        
        results = []
        for prob in probs:
            top_indices = np.argsort(prob)[::-1][:top_k]
            top_categories = [
                (self.label_encoder.classes_[i], float(prob[i])) 
                for i in top_indices
            ]
            
            best_idx = top_indices[0]
            
            results.append(ClassificationResult(
                category=self.label_encoder.classes_[best_idx],
                confidence=float(prob[best_idx]),
                top_k=top_categories,
                probabilities={
                    self.label_encoder.classes_[i]: float(prob[i]) 
                    for i in range(len(prob))
                }
            ))
        
        return results
    
    def save(self, path: Union[str, Path]) -> None:
        """Save model to directory"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save PyTorch model
        torch.save(self.model.state_dict(), path / "model.pt")
        
        # Save vectorizers
        with open(path / "tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)
        
        with open(path / "char_vectorizer.pkl", "wb") as f:
            pickle.dump(self.char_vectorizer, f)
        
        # Save label encoder
        with open(path / "label_encoder.pkl", "wb") as f:
            pickle.dump(self.label_encoder, f)
        
        # Save skill encoder if exists
        if self.skill_encoder:
            with open(path / "skill_encoder.pkl", "wb") as f:
                pickle.dump(self.skill_encoder, f)
        
        # Save config
        config = {
            "tfidf_max_features": self.tfidf_max_features,
            "hidden_dims": self.hidden_dims,
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
            tfidf_max_features=config["tfidf_max_features"],
            hidden_dims=config["hidden_dims"],
            dropout=config["dropout"],
            device=device
        )
        
        classifier.input_dim = config["input_dim"]
        classifier.num_classes = config["num_classes"]
        
        # Load vectorizers
        with open(path / "tfidf_vectorizer.pkl", "rb") as f:
            classifier.vectorizer = pickle.load(f)
        
        with open(path / "char_vectorizer.pkl", "rb") as f:
            classifier.char_vectorizer = pickle.load(f)
        
        # Load label encoder
        with open(path / "label_encoder.pkl", "rb") as f:
            classifier.label_encoder = pickle.load(f)
        
        # Load skill encoder if exists
        skill_encoder_path = path / "skill_encoder.pkl"
        if skill_encoder_path.exists():
            with open(skill_encoder_path, "rb") as f:
                classifier.skill_encoder = pickle.load(f)
        
        # Load PyTorch model
        classifier.model = ResumeCategoryNetwork(
            input_dim=classifier.input_dim,
            num_classes=classifier.num_classes,
            hidden_dims=classifier.hidden_dims,
            dropout=classifier.dropout
        ).to(classifier.device)
        
        classifier.model.load_state_dict(
            torch.load(path / "model.pt", map_location=classifier.device)
        )
        classifier.model.eval()
        
        classifier.is_fitted = True
        logger.info(f"Model loaded from {path}")
        
        return classifier