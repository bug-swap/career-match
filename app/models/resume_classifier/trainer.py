"""
Resume Classifier Trainer
Loads data and trains the PyTorch model
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
    """
    
    def __init__(
        self,
        data_dir: Path = Path("data"),
        artifacts_dir: Path = Path("app/artifacts/resume_classifier"),
        tfidf_max_features: int = 7000,
        hidden_dims: List[int] = [768, 384, 192],
        dropout: float = 0.4
    ):
        self.data_dir = Path(data_dir)
        self.artifacts_dir = Path(artifacts_dir)
        self.tfidf_max_features = tfidf_max_features
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        
        self.device = get_device()
        self.model: Optional[ResumeClassifier] = None
        
        logger.info(f"Using device: {self.device}")
    
    def load_data(
        self,
        csv_path: Optional[Path] = None,
        text_column: str = "Resume_text",
        label_column: str = "Category",
        skills_column: Optional[str] = None
    ) -> Tuple[List[str], List[str], Optional[List[List[str]]]]:
        """
        Load training data from CSV
        
        Args:
            csv_path: Path to resume CSV
            text_column: Column containing resume text
            label_column: Column containing category labels
            skills_column: Optional column containing skills
        
        Returns:
            (texts, labels, skills)
        """
        if csv_path is None:
            csv_path = self.data_dir / "processed" / "Resume.csv"
        
        logger.info(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Clean data
        df = df.dropna(subset=[text_column, label_column])
        df = df[df[text_column].str.len() > 50]  # Filter very short resumes
        
        texts = df[text_column].tolist()
        labels = df[label_column].tolist()
        
        # Parse skills if column exists
        skills = None
        if skills_column and skills_column in df.columns:
            def parse_skills(s):
                if pd.isna(s) or not s:
                    return []
                return [skill.strip() for skill in str(s).split(",")]
            skills = df[skills_column].apply(parse_skills).tolist()
        
        logger.info(f"Loaded {len(texts)} samples")
        logger.info(f"Categories: {df[label_column].nunique()} unique")
        logger.info(f"Distribution:\n{df[label_column].value_counts().head(10)}")
        
        return texts, labels, skills
    
    def train(
        self,
        csv_path: Optional[Path] = None,
        text_column: str = "Resume_text",
        label_column: str = "Category",
        skills_column: Optional[str] = None,
        test_size: float = 0.2,
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 0.001,
        patience: int = 10
    ) -> Dict[str, float]:
        """
        Train the resume classifier
        
        Args:
            csv_path: Path to training CSV
            text_column: Column with resume text
            label_column: Column with category
            skills_column: Optional column with skills
            test_size: Test split ratio
            epochs: Training epochs
            batch_size: Batch size
            lr: Learning rate
            patience: Early stopping patience
        
        Returns:
            Training metrics
        """
        logger.info("=" * 60)
        logger.info("Training Resume Classifier")
        logger.info("=" * 60)
        
        # Load data
        texts, labels, skills = self.load_data(
            csv_path, text_column, label_column, skills_column
        )
        
        # Split data
        if skills:
            train_texts, val_texts, train_labels, val_labels, train_skills, val_skills = train_test_split(
                texts, labels, skills, test_size=test_size, stratify=labels, random_state=42
            )
        else:
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                texts, labels, test_size=test_size, stratify=labels, random_state=42
            )
            train_skills, val_skills = None, None
        
        logger.info(f"Train: {len(train_texts)}, Val: {len(val_texts)}")
        
        # Create and train model
        self.model = ResumeClassifier(
            tfidf_max_features=self.tfidf_max_features,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            device=self.device
        )
        
        metrics = self.model.fit(
            texts=train_texts,
            labels=train_labels,
            skills=train_skills,
            val_texts=val_texts,
            val_labels=val_labels,
            val_skills=val_skills,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience
        )
        
        # Save model
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
        """Evaluate model on test data"""
        from sklearn.metrics import accuracy_score, f1_score, classification_report
        
        if self.model is None:
            self.model = ResumeClassifier.load(self.artifacts_dir)
        
        texts, labels, _ = self.load_data(csv_path, text_column, label_column)
        
        # Predict
        results = self.model.predict_batch(texts)
        predictions = [r.category for r in results]
        
        # Metrics
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="macro")
        
        logger.info(f"Accuracy: {acc:.4f}")
        logger.info(f"F1 (macro): {f1:.4f}")
        logger.info(f"\n{classification_report(labels, predictions)}")
        
        return {"accuracy": acc, "f1_macro": f1}
    
    def test(self, texts: Optional[List[str]] = None):
        """Test predictions on sample texts"""
        if self.model is None:
            self.model = ResumeClassifier.load(self.artifacts_dir)
        
        if texts is None:
            texts = [
                "Software Engineer with 5 years experience in Python, Java, and cloud technologies. Built scalable microservices at Google.",
                "Data Scientist specializing in machine learning and deep learning. PhD in Statistics. Experience with TensorFlow and PyTorch.",
                "Marketing Manager with expertise in digital marketing, SEO, and brand strategy. Led campaigns for Fortune 500 companies.",
                "Registered Nurse with 10 years of clinical experience in emergency medicine and patient care."
            ]
        
        logger.info("\nTest Predictions:")
        for text in texts:
            result = self.model.predict(text)
            logger.info(f"\nText: '{text[:80]}...'")
            logger.info(f"  Category: {result.category} ({result.confidence:.2%})")
            logger.info(f"  Top 3: {result.top_k}")


# Entry point
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    trainer = ResumeClassifierTrainer()
    trainer.train(epochs=100, batch_size=24, lr=0.002, patience=15)
    trainer.test()