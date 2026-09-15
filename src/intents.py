import enum
from typing import List
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ----------------------------------------------------------------------
# Intent enumeration (unchanged from original)
# ----------------------------------------------------------------------
class Intent(str, enum.Enum):
    DELIVERY_DELAY = "DELIVERY_DELAY"
    DELIVERED_NOT_RECEIVED = "DELIVERED_NOT_RECEIVED"
    SHIPPING_TRACKING = "SHIPPING_TRACKING"
    REFUND = "REFUND"
    PAYMENT_CHARGE = "PAYMENT_CHARGE"
    PRIME = "PRIME"
    ACCOUNT_ACCESS = "ACCOUNT_ACCESS"
    PRODUCT_DEVICE = "PRODUCT_DEVICE"
    PRODUCT_PROBLEM = "PRODUCT_PROBLEM"
    GENERAL_SUPPORT = "GENERAL_SUPPORT"
    OTHER = "OTHER"

# ----------------------------------------------------------------------
# Simple keyword‑based rule classifier (kept for quick inspection)
# ----------------------------------------------------------------------
_INTENT_KEYWORDS = {
    Intent.DELIVERY_DELAY: ["delivery date", "haven't arrived", "delayed", "late", "expected delivery"],
    Intent.DELIVERED_NOT_RECEIVED: ["delivered but i didn't receive", "marked delivered", "not received"],
    Intent.SHIPPING_TRACKING: ["tracking", "order status", "where is my order", "track my package"],
    Intent.REFUND: ["refund", "money back", "return", "reimburse"],
    Intent.PAYMENT_CHARGE: ["charge", "payment", "billing", "charged"],
    Intent.PRIME: ["prime", "membership", "prime trial", "prime benefit"],
    Intent.ACCOUNT_ACCESS: ["login", "password", "account access", "sign in", "register"],
    Intent.PRODUCT_DEVICE: ["kindle", "echo", "fire tv", "alexa", "device", "app"],
    Intent.PRODUCT_PROBLEM: ["defective", "damaged", "broken", "wrong item", "missing part"],
    Intent.GENERAL_SUPPORT: ["help", "question", "support", "issue", "problem"],
    Intent.OTHER: [],
}

def classify_intent(text: str) -> Intent:
    lowered = text.lower()
    for intent, keywords in _INTENT_KEYWORDS.items():
        for kw in keywords:
            if kw in lowered:
                return intent
    if any(word in lowered for word in ["thanks", "great", "awesome", "love", "congrats"]):
        return Intent.OTHER
    if any(word in lowered for word in ["how", "what", "when", "where", "why"]):
        return Intent.GENERAL_SUPPORT
    return Intent.OTHER

# ----------------------------------------------------------------------
# Trainable IntentClassifier – uses TF‑IDF + LogisticRegression
# ----------------------------------------------------------------------
class IntentClassifier:
    """A lightweight classifier that can be trained on the golden set.

    The model is persisted with joblib to ``data/processed/intent_model.joblib``.
    """

    def __init__(self):
        # TF‑IDF + Linear model
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english")),
            ("clf", LogisticRegression(max_iter=1000)),
        ])
        self._ensemble = None  # set when loading an ensemble joblib

    def train(self, texts: List[str], labels: List[str]):
        """Fit the model on ``texts`` and ``labels`` (string intent names)."""
        self.pipeline.fit(texts, labels)

    def predict(self, text: str):
        """Return a dict ``{"intent": <str>, "confidence": <float>}``.

        Handles both single-pipeline and ensemble (word+char) joblib formats.
        """
        if self._ensemble:
            word_w = self._ensemble["word_weight"]
            char_w = self._ensemble["char_weight"]
            classes = self._ensemble["classes"]
            probs = (
                word_w * self._ensemble["word"].predict_proba([text])[0] +
                char_w * self._ensemble["char"].predict_proba([text])[0]
            )
            idx = probs.argmax()
            return {"intent": classes[idx], "confidence": float(probs[idx])}
        probs = self.pipeline.predict_proba([text])[0]
        idx = probs.argmax()
        return {"intent": self.pipeline.classes_[idx], "confidence": float(probs[idx])}

    def save(self, path: str):
        joblib.dump(self.pipeline, path)

    def load(self, path: str):
        data = joblib.load(path)
        if isinstance(data, dict) and data.get("type") == "ensemble":
            self._ensemble = data
        else:
            self.pipeline = data
            self._ensemble = None