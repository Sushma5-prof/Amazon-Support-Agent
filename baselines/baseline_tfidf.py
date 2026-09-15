import sys
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

GOLDEN_PATH = PROJECT_ROOT / "data" / "golden_set.csv"

def main():
    df = pd.read_csv(GOLDEN_PATH)
    X, y = df["customer_text"].astype(str), df["intent"].astype(str)
    from sklearn.pipeline import Pipeline
    clf = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("lr", LogisticRegression(max_iter=1000)),
    ])
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    print(f"TF-IDF baseline 5-fold CV accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

if __name__ == "__main__":
    main()
