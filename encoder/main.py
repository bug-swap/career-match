"""
Cloud Run Function - Generate Job Embeddings
Models mounted at /mnt/models
"""

import os
import logging
import functions_framework
from flask import jsonify
from typing import List, Dict, Any, Optional
from supabase import create_client, Client
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from dotenv import load_dotenv

from encoder import JobEncoder

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model path
# MODEL_PATH = "/mnt/models/encoder.pt"
# CONFIG_PATH = "/mnt/models/config.json"
MODEL_PATH = "../data/artifacts/job_matcher/encoder.pt"
CONFIG_PATH = "../data/artifacts/job_matcher/config.json"


# ============================================================
# Globals (reused across invocations)
# ============================================================

encoder: Optional[JobEncoder] = None
db_client: Optional[Client] = None


def get_encoder() -> JobEncoder:
    global encoder
    if encoder is None:
        encoder = JobEncoder(
            config_path=CONFIG_PATH,
            model_path=MODEL_PATH
        )
    return encoder


def get_db() -> Client:
    global db_client
    if db_client is None:
        db_client = create_client(
            os.environ.get("SUPABASE_URL"),
            os.environ.get("SUPABASE_API_KEY")
        )
    return db_client


def job_to_text(job: Dict[str, Any]) -> str:
    """Convert job row to text for embedding"""
    parts = []
    if job.get("title"):
        parts.append(job["title"])
    if job.get("company"):
        parts.append(f"at {job['company']}")
    if job.get("location"):
        parts.append(f"in {job['location']}")
    if job.get("is_remote"):
        parts.append("(Remote)")
    return " ".join(parts)





# ============================================================
# Cloud Function
# ============================================================

@functions_framework.http
def main(request):
    logger.info("Received request for job embeddings")
    load_dotenv()
    if request.method == "OPTIONS":
        return ("", 204, {"Access-Control-Allow-Origin": "*", "Access-Control-Allow-Methods": "POST"})
    
    headers = {"Access-Control-Allow-Origin": "*"}
    
    try:
        data = request.get_json(silent=True) or {}
        logger.info(f"Request data: {data}")
        db = get_db()
        enc = get_encoder()
        
        jobs = data.get("jobs")
        if not jobs:
            return jsonify({"success": True, "message": "No jobs to process", "processed": 0}), 200, headers
        
        logger.info(f"Processing {len(jobs)} jobs")
        
        # Generate embeddings
        texts = [job_to_text(job) for job in jobs]
        embeddings = enc.encode(texts)
        print(f"Generated {len(embeddings)} embeddings")
        # Update DB
        updated = 0
        for job, emb in zip(jobs, embeddings):
            try:
                res = db.from_("jobs").update({"embedding": emb}).eq("id", job["id"]).execute()
                if res[1] != 200:
                    logger.error(f"Failed to update job {job['id']}: {res[1]} {res[2]}")
                    raise Exception(f"DB update failed with status {res[1]}")
                else:   
                    logger.info(f"Updated job {job['id']} successfully")        
                    updated += 1
            except Exception as e:
                logger.error(f"Failed to update {job['id']}: {e}")
        
        return jsonify({"success": True, "processed": len(jobs), "updated": updated}), 200, headers
    
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({"error": str(e)}), 500, headers