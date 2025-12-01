"""
Wrapper for JobMatcher model
Handles loading and provides type-safe interface
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional
import sys
import torch
import numpy as np

logger = logging.getLogger(__name__)


class JobMatcherModel:
    """Wrapper class for JobMatcher model"""
    
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model = None
        self._load()
    
    def _load(self):
        """Load the JobMatcher model"""
        try:
            # Add models directory to path
            models_dir = self.model_path.parent.parent.parent / 'models'
            if str(models_dir) not in sys.path:
                sys.path.insert(0, str(models_dir))
            
            logger.info(f"[JobMatcher] Loading from {self.model_path}")
            logger.info(f"[JobMatcher] Path exists: {self.model_path.exists()}")
            
            if not self.model_path.exists():
                logger.error(f"[JobMatcher] Model path does not exist: {self.model_path}")
                return
            
            # List files
            files = list(self.model_path.glob('*'))
            logger.info(f"[JobMatcher] Files found: {[f.name for f in files]}")
            
            # Import and load
            from job_matcher.model import JobMatcher
            self.model = JobMatcher.load(self.model_path, device=torch.device('cpu'))
            
            # Set to eval mode
            if self.model and hasattr(self.model, 'encoder') and self.model.encoder:
                self.model.encoder.eval()
            
            logger.info(f"[JobMatcher] ✅ Loaded successfully")
            logger.info(f"[JobMatcher] is_fitted: {self.model.is_fitted}")
            logger.info(f"[JobMatcher] Embedding dim: {self.model.embedding_dim}")
            
        except Exception as e:
            logger.error(f"[JobMatcher] ❌ Failed to load: {e}", exc_info=True)
            self.model = None
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to embeddings
        
        Args:
            texts: List of texts to encode
            
        Returns:
            Numpy array of embeddings
        """
        if not self.model:
            logger.error("[JobMatcher] Model not loaded, cannot encode")
            return np.array([])
        
        try:
            return self.model.encode(texts)
        except Exception as e:
            logger.error(f"[JobMatcher] Error during encoding: {e}", exc_info=True)
            return np.array([])
    
    def encode_single(self, text: str) -> np.ndarray:
        """
        Encode single text to embedding
        
        Args:
            text: Text to encode
            
        Returns:
            Numpy array embedding
        """
        if not self.model:
            logger.error("[JobMatcher] Model not loaded, cannot encode")
            return np.array([])
        
        try:
            return self.model.encode_single(text)
        except Exception as e:
            logger.error(f"[JobMatcher] Error during encoding: {e}", exc_info=True)
            return np.array([])
    
    def compute_similarity(self, resume_embedding: np.ndarray, job_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute similarity between resume and jobs
        
        Args:
            resume_embedding: Single resume embedding
            job_embeddings: Array of job embeddings
            
        Returns:
            Array of similarity scores
        """
        if not self.model:
            logger.error("[JobMatcher] Model not loaded, cannot compute similarity")
            return np.array([])
        
        try:
            return self.model.compute_similarity(resume_embedding, job_embeddings)
        except Exception as e:
            logger.error(f"[JobMatcher] Error computing similarity: {e}", exc_info=True)
            return np.array([])
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None and hasattr(self.model, 'is_fitted') and self.model.is_fitted
