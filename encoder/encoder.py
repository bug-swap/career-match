
import os
import logging
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
# ============================================================
# Encoder
# ============================================================

class JobEncoder:
    def __init__(self, model_path: str, config_path: str):
        # Use MPS if available, else CPU
        self.device = torch.device("cpu")
        self.sbert = SentenceTransformer("all-MiniLM-L6-v2", device=str(self.device))

        self.projection = nn.Sequential(
            nn.Linear(384, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        ).to(self.device)
        self.logger = logging.getLogger(__name__)

        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            proj_state = {k.replace("projection.", ""): v for k, v in state_dict.items() if "projection" in k}
            self.projection.load_state_dict(proj_state if proj_state else state_dict)
            self.logger.info("Loaded model weights")

        self.projection.eval()
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        with torch.no_grad():
            sbert_emb = self.sbert.encode(texts, convert_to_tensor=True, device=str(self.device))
            sbert_emb = sbert_emb.to(self.device)
            projected = self.projection(sbert_emb)
            normalized = F.normalize(projected, p=2, dim=1)

        return normalized.cpu().numpy().tolist()

