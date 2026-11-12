import os
from typing import Tuple
import joblib

MODEL_PATH = "spam_model.joblib"
_model = None


def get_model():
    """Lazy-loads the serialized pipeline once per Python session."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Serialized model '{MODEL_PATH}' not found. Run 'python train.py' first."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_message(text: str) -> Tuple[str, float]:
    """
    Classifies a single SMS message as 'spam' or 'not spam' with confidence score.
    Handles empty or whitespace-only strings gracefully without crashing.
    """
    if not isinstance(text, str) or not text.strip():
        return "invalid input (empty message)", 0.0

    model = get_model()
    proba = model.predict_proba([text.strip()])[0]
    label = "spam" if proba[1] >= 0.5 else "not spam"
    confidence = float(max(proba))
    return label, confidence


if __name__ == "__main__":
    sample_texts = [
        "WINNER!! You have won a free prize, call now",
        "Hey, are we still meeting for lunch?",
        "Free entry in a weekly competition, text WIN to 80086",
        "I'm free tonight",
        "Your parcel is waiting, click here",
        "   ",
        "",
    ]

    print("=" * 68)
    print("  SPAM DETECTOR INFERENCE DEMO (LOADED FROM spam_model.joblib)")
    print("=" * 68)
    for msg in sample_texts:
        lbl, conf = predict_message(msg)
        display_msg = repr(msg) if len(msg.strip()) == 0 else (msg[:38] + "..." if len(msg) > 41 else msg)
        print(f"{display_msg:<42} -> {lbl:<26} ({conf * 100:.2f}%)")
    print("=" * 68)