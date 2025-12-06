"""
Job Matcher Trainer
Trains on Resume.csv using contrastive learning
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .model import JobEmbeddor, get_device

logger = logging.getLogger(__name__)


class JobEmbeddorTrainer:
    """
    Trainer for Job Matcher
    
    Trains encoder using contrastive learning on resume pairs
    Same category = similar embeddings, different category = dissimilar
    """
    
    def __init__(
        self,
        data_dir: Path = Path("data"),
        artifacts_dir: Path = Path("app/artifacts/job_matcher"),
        sbert_model: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 128,
        hidden_dims: List[int] = [512, 256],
        dropout: float = 0.3,
        margin: float = 0.5
    ):
        self.data_dir = Path(data_dir)
        self.artifacts_dir = Path(artifacts_dir)
        self.sbert_model = sbert_model
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.margin = margin
        
        self.device = get_device()
        self.model: Optional[JobEmbeddor] = None
        
        logger.info(f"Using device: {self.device}")
    
    def load_data(
        self,
        csv_path: Optional[Path] = None,
        text_column: str = "Resume_text",
        category_column: str = "Category"
    ) -> tuple:
        """Load resume data"""
        if csv_path is None:
            csv_path = self.data_dir / "processed" / "Resume.csv"
        
        logger.info(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)
        
        df = df.dropna(subset=[text_column, category_column])
        df = df[df[text_column].str.len() > 50]
        
        texts = df[text_column].tolist()
        categories = df[category_column].tolist()
        
        logger.info(f"Loaded {len(texts)} resumes")
        logger.info(f"Categories: {df[category_column].nunique()}")
        
        return texts, categories
    
    def train(
        self,
        csv_path: Optional[Path] = None,
        text_column: str = "Resume_text",
        category_column: str = "Category",
        epochs: int = 30,
        batch_size: int = 64,
        lr: float = 0.001,
        pairs_per_epoch: int = 10000,
        patience: int = 7
    ) -> Dict[str, float]:
        """Train the job matcher"""
        logger.info("=" * 60)
        logger.info("Training Job Matcher")
        logger.info("=" * 60)
        
        texts, categories = self.load_data(csv_path, text_column, category_column)
        
        self.model = JobEmbeddor(
            sbert_model=self.sbert_model,
            embedding_dim=self.embedding_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            margin=self.margin,
            device=self.device
        )
        
        metrics = self.model.fit(
            texts=texts,
            categories=categories,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            pairs_per_epoch=pairs_per_epoch,
            patience=patience
        )
        
        self.model.save(self.artifacts_dir)
        
        logger.info("=" * 60)
        logger.info(f"Training complete! Model saved to {self.artifacts_dir}")
        logger.info("=" * 60)
        
        return metrics
    
    def evaluate(
        self,
        csv_path: Optional[Path] = None,
        text_column: str = "Resume_text",
        category_column: str = "Category",
        n_queries: int = 100
    ) -> Dict[str, float]:
        """
        Evaluate matching quality
        
        For each query resume, check if top-k retrieved resumes
        are from the same category
        """
        if self.model is None:
            self.model = JobEmbeddor.load(self.artifacts_dir)
        
        texts, categories = self.load_data(csv_path, text_column, category_column)
        
        logger.info("Encoding all resumes...")
        all_embeddings = self.model.encode(texts)
        
        # Sample query indices
        query_indices = np.random.choice(len(texts), size=min(n_queries, len(texts)), replace=False)
        
        hits_at_1 = 0
        hits_at_5 = 0
        hits_at_10 = 0
        
        for query_idx in query_indices:
            query_emb = all_embeddings[query_idx]
            query_cat = categories[query_idx]
            
            # Compute similarities (exclude self)
            sims = self.model.compute_similarity(query_emb, all_embeddings)
            sims[query_idx] = -1  # Exclude self
            
            # Get top-k
            top_indices = np.argsort(sims)[::-1]
            
            # Check hits
            if categories[top_indices[0]] == query_cat:
                hits_at_1 += 1
            
            top_5_cats = [categories[i] for i in top_indices[:5]]
            if query_cat in top_5_cats:
                hits_at_5 += 1
            
            top_10_cats = [categories[i] for i in top_indices[:10]]
            if query_cat in top_10_cats:
                hits_at_10 += 1
        
        metrics = {
            "hits@1": hits_at_1 / len(query_indices),
            "hits@5": hits_at_5 / len(query_indices),
            "hits@10": hits_at_10 / len(query_indices)
        }
        
        logger.info(f"Evaluation Results:")
        logger.info(f"  Hits@1:  {metrics['hits@1']:.2%}")
        logger.info(f"  Hits@5:  {metrics['hits@5']:.2%}")
        logger.info(f"  Hits@10: {metrics['hits@10']:.2%}")
        
        return metrics
    
    def test(self):
        """Test the model with sample queries"""
        if self.model is None:
            self.model = JobEmbeddor.load(self.artifacts_dir)
        
        # Sample resumes
        test_resumes = [
            "Software Engineer with 5 years experience in Python and machine learning. Built scalable APIs at Google.",
            "Marketing Manager specializing in digital marketing, SEO, and brand strategy for Fortune 500 companies.",
            "Registered Nurse with 10 years clinical experience in emergency medicine and patient care.",
            "Financial Analyst with expertise in Excel modeling, budgeting, and investment analysis."
        ]
        
        # Sample job descriptions (what would be in DB)
        test_jobs = [
            "Senior Python Developer needed for ML team. Experience with TensorFlow required.",
            "Digital Marketing Specialist to lead SEO and content marketing initiatives.",
            "Emergency Room Nurse for night shift. BLS certification required.",
            "Junior Financial Analyst for investment banking division."
        ]
        job_ids = ["job_1", "job_2", "job_3", "job_4"]
        
        # Encode jobs (simulating what DB would store)
        logger.info("Encoding sample jobs...")
        job_embeddings = self.model.encode(test_jobs)
        
        logger.info("\nTest Matches:")
        for i, resume in enumerate(test_resumes):
            matches = self.model.match(resume, job_embeddings, job_ids, top_k=2)
            logger.info(f"\nResume {i+1}: '{resume[:60]}...'")
            for match in matches:
                job_idx = job_ids.index(match['job_id'])
                logger.info(f"  -> {match['job_id']}: '{test_jobs[job_idx][:50]}...' (score: {match['score']:.2%})")


# Entry point
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    trainer = JobEmbeddorTrainer()
    trainer.train(epochs=80, batch_size=64, pairs_per_epoch=10000)
    trainer.evaluate(n_queries=100)
    trainer.test()