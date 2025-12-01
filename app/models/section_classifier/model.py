import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

from .rules import SectionRules

logger = logging.getLogger(__name__)


@dataclass
class SectionPrediction:
    """Prediction result for a text block"""
    section: str
    confidence: float
    all_scores: Dict[str, float]


class SectionClassifier:
    """
    Hybrid Section Classifier
    - TF-IDF + ML classifier (configurable weight)
    - Rule-based keyword matching
    """

    BASE_SECTION_CLASSES = [
        "person_skills"  # structured data produces this label even if rules do not
    ]
    
    def __init__(
        self,
        algorithm: str = "logistic_regression",
        tfidf_max_features: int = 3000,
        tfidf_ngram_range: Tuple[int, int] = (1, 2),
        rule_weight: float = 0.3,
        random_state: int = 42,
        section_classes: Optional[List[str]] = None
    ):
        self.algorithm = algorithm
        self.tfidf_max_features = tfidf_max_features
        self.tfidf_ngram_range = tfidf_ngram_range
        self.rule_weight = rule_weight
        self.ml_weight = 1 - rule_weight
        self.random_state = random_state
        
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.classifier = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.rules = SectionRules(weight=rule_weight)
        self.section_classes = section_classes or self._default_section_classes()
        self._section_class_set = set(self.section_classes)
        self.is_fitted = False

    @classmethod
    def _default_section_classes(cls) -> List[str]:
        classes = set(SectionRules.SECTION_KEYWORDS.keys()) | set(cls.BASE_SECTION_CLASSES)
        return sorted(classes)
    
    def _create_classifier(self):
        if self.algorithm == "logistic_regression":
            return LogisticRegression(
                max_iter=1000,
                multi_class="multinomial",
                solver="lbfgs",
                class_weight="balanced",
                random_state=self.random_state
            )
        elif self.algorithm == "svm":
            return LinearSVC(
                max_iter=1000,
                class_weight="balanced",
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def fit(
        self, 
        texts: List[str], 
        labels: List[str],
        val_split: float = 0.1
    ) -> Dict[str, float]:
        """Train the classifier"""
        logger.info(f"Training Section Classifier with {len(texts)} samples")
        
        self.vectorizer = TfidfVectorizer(
            max_features=self.tfidf_max_features,
            ngram_range=self.tfidf_ngram_range,
            stop_words="english",
            min_df=2,
            max_df=0.98
        )
        
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.section_classes)
        self.classifier = self._create_classifier()
        
        valid_texts: List[str] = []
        valid_labels: List[str] = []
        invalid_labels: set[str] = set()
        for text, label in zip(texts, labels):
            if label in self._section_class_set:
                valid_texts.append(text)
                valid_labels.append(label)
            else:
                invalid_labels.add(label)

        if not valid_texts:
            raise ValueError("No samples with supported section labels were provided")

        if invalid_labels:
            logger.warning(
                "Dropping %d samples with unsupported labels: %s",
                len(texts) - len(valid_texts),
                ", ".join(sorted(invalid_labels))
            )

        y = self.label_encoder.transform(valid_labels)
        
        X_train, X_val, y_train, y_val = train_test_split(
            valid_texts, y, test_size=val_split, stratify=y, random_state=self.random_state
        )
        
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_val_tfidf = self.vectorizer.transform(X_val)
        
        self.classifier.fit(X_train_tfidf, y_train)
        
        y_pred = self.classifier.predict(X_val_tfidf)
        f1 = f1_score(y_val, y_pred, average="macro")
        
        logger.info(f"Validation F1 (macro): {f1:.4f}")
        logger.info("\n" + classification_report(y_val, y_pred, target_names=self.label_encoder.classes_))
        
        self.is_fitted = True
        return {"val_f1_macro": f1, "n_train": len(X_train), "n_val": len(X_val)}
    
    def _get_ml_probabilities(self, text: str) -> Dict[str, float]:
        X = self.vectorizer.transform([text])
        if hasattr(self.classifier, "predict_proba"):
            logger.debug("Using predict_proba for probabilities")
            probs = self.classifier.predict_proba(X)[0]
        else:
            logger.debug("Using decision_function for probabilities")
            decision = self.classifier.decision_function(X)[0]
            exp_scores = np.exp(decision - np.max(decision))
            probs = exp_scores / exp_scores.sum()
        return {label: float(prob) for label, prob in zip(self.label_encoder.classes_, probs)}
    
    def predict(self, text: str) -> SectionPrediction:
        """Predict section for a text block"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        ml_scores = self._get_ml_probabilities(text)
        rule_scores = self.rules.score_text(text)
        
        combined = {}
        for section in self.section_classes:
            combined[section] = (
                self.ml_weight * ml_scores.get(section, 0.0) + 
                self.rule_weight * rule_scores.get(section, 0.0)
            )
        
        best = max(combined, key=combined.get)
        return SectionPrediction(section=best, confidence=combined[best], all_scores=combined)
    
    def predict_batch(self, texts: List[str]) -> List[SectionPrediction]:
        return [self.predict(text) for text in texts]
    
    def segment_resume(self, resume_text: str, min_confidence: float = 0.3) -> Dict[str, str]:
        """Segment a full resume into sections"""
        rule_segments = self.rules.segment_resume(resume_text)
        final = {}
        
        for section, content in rule_segments.items():
            if not content.strip():
                continue
            pred = self.predict(content)
            final_section = pred.section if pred.confidence >= min_confidence else section
            final[final_section] = final.get(final_section, '') + '\n' + content
        
        return {k: v.strip() for k, v in final.items()}
    
    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        with open(path / "tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)
        with open(path / "classifier.pkl", "wb") as f:
            pickle.dump(self.classifier, f)
        with open(path / "label_encoder.pkl", "wb") as f:
            pickle.dump(self.label_encoder, f)
        
        config = {
            "algorithm": self.algorithm,
            "tfidf_max_features": self.tfidf_max_features,
            "tfidf_ngram_range": self.tfidf_ngram_range,
            "rule_weight": self.rule_weight,
            "random_state": self.random_state
        }
        with open(path / "config.pkl", "wb") as f:
            pickle.dump(config, f)
        
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "SectionClassifier":
        path = Path(path)
        
        with open(path / "config.pkl", "rb") as f:
            config = pickle.load(f)
        
        model = cls(**config)
        
        with open(path / "tfidf_vectorizer.pkl", "rb") as f:
            model.vectorizer = pickle.load(f)
        with open(path / "classifier.pkl", "rb") as f:
            model.classifier = pickle.load(f)
        with open(path / "label_encoder.pkl", "rb") as f:
            model.label_encoder = pickle.load(f)
        
        model.is_fitted = True
        logger.info(f"Model loaded from {path}")
        return model