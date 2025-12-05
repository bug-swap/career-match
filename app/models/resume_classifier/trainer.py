"""
Resume Classifier Trainer
Loads data and trains the transformer-based model
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .model import ResumeClassifier, get_device

logger = logging.getLogger(__name__)


class ResumeClassifierTrainer:
    """
    Trainer for Resume Category Classifier
    
    Features:
        - Sentence-BERT embeddings
        - Transformer-based classifier
        - Mixup augmentation
        - Label smoothing
    """
    
    def __init__(
        self,
        data_dir: Path = Path("data"),
        artifacts_dir: Path = Path("app/artifacts/resume_classifier"),
        sbert_model: str = "all-MiniLM-L6-v2",
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 3,
        ff_dim: int = 2048,
        dropout: float = 0.3
    ):
        self.data_dir = Path(data_dir)
        self.artifacts_dir = Path(artifacts_dir)
        self.sbert_model = sbert_model
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.dropout = dropout
        
        self.device = get_device()
        self.model: Optional[ResumeClassifier] = None
        
        logger.info(f"Using device: {self.device}")
    
    def load_data(
        self,
        csv_path: Optional[Path] = None,
        text_column: str = "Resume_text",
        label_column: str = "Category",
    ) -> Tuple[List[str], List[str]]:
        """Load training data from CSV"""
        if csv_path is None:
            csv_path = self.data_dir / "processed" / "Resume.csv"
        
        logger.info(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)
        
        df = df.dropna(subset=[text_column, label_column])
        df = df[df[text_column].str.len() > 50]
        
        texts = df[text_column].tolist()
        labels = df[label_column].tolist()
        
        logger.info(f"Loaded {len(texts)} samples")
        logger.info(f"Categories: {df[label_column].nunique()} unique")
        logger.info(f"Distribution:\n{df[label_column].value_counts().head(10)}")
        
        return texts, labels
    
    def train(
        self,
        csv_path: Optional[Path] = None,
        text_column: str = "Resume_text",
        label_column: str = "Category",
        test_size: float = 0.2,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-4,
        patience: int = 10,
        mixup_alpha: float = 0.2,
        label_smoothing: float = 0.1
    ) -> Dict[str, float]:
        """Train the resume classifier"""
        logger.info("=" * 60)
        logger.info("Training Resume Classifier (Transformer)")
        logger.info("=" * 60)
        
        texts, labels = self.load_data(csv_path, text_column, label_column)
        
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=test_size, stratify=labels, random_state=42
        )
        
        logger.info(f"Train: {len(train_texts)}, Val: {len(val_texts)}")
        
        self.model = ResumeClassifier(
            sbert_model=self.sbert_model,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            ff_dim=self.ff_dim,
            dropout=self.dropout,
            device=self.device
        )
        
        metrics = self.model.fit(
            texts=train_texts,
            labels=train_labels,
            val_texts=val_texts,
            val_labels=val_labels,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            mixup_alpha=mixup_alpha,
            label_smoothing=label_smoothing
        )
        
        self.model.save(self.artifacts_dir)
        
        logger.info("=" * 60)
        logger.info(f"Training complete! Model saved to {self.artifacts_dir}")
        logger.info(f"Metrics: {metrics}")
        logger.info("=" * 60)
        
        return metrics
    
    def evaluate(
        self,
        csv_path: Optional[Path] = None,
        text_column: str = "Resume_text",
        label_column: str = "Category"
    ) -> Dict[str, float]:
        """Evaluate model"""
        from sklearn.metrics import accuracy_score, f1_score, classification_report
        
        if self.model is None:
            self.model = ResumeClassifier.load(self.artifacts_dir)
        
        texts, labels = self.load_data(csv_path, text_column, label_column)
        
        results = self.model.predict_batch(texts)
        predictions = [r.category for r in results]
        
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="macro")
        
        logger.info(f"Accuracy: {acc:.4f}")
        logger.info(f"F1 (macro): {f1:.4f}")
        logger.info(f"\n{classification_report(labels, predictions)}")
        
        return {"accuracy": acc, "f1_macro": f1}
    
    def test(self, texts: Optional[List[str]] = None):
        """Test on sample texts"""
        if self.model is None:
            self.model = ResumeClassifier.load(self.artifacts_dir)
        
        if texts is None:
            texts = [
                "Experienced software engineer with expertise in Python, Java, and cloud computing. Skilled in developing scalable web applications and working with cross-functional teams.",
                "Marketing professional with a strong background in digital marketing, SEO, and content creation. Proven track record of increasing brand awareness and driving online engagement.",
                "Data scientist proficient in machine learning, statistical analysis, and data visualization. Experienced in using Python, R, and SQL to extract insights from large datasets."
            ]
        
        logger.info("\nTest Predictions:")
        for text in texts:
            result = self.model.predict(text)
            logger.info(f"\nText: '{text[:80]}...'")
            logger.info(f"  Category: {result.category} ({result.confidence:.2%})")
            logger.info(f"  Top 3: {result.top_k}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    trainer = ResumeClassifierTrainer()
    trainer.train(
        test_size=0.15,
        epochs=20,
        batch_size=16,
        lr=3e-5,
        patience=4,
        mixup_alpha=0.0,
        label_smoothing=0.05
    )
    trainer.test()