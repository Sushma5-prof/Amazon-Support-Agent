import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
sys.path.insert(0, r'D:/Progress/Hiver/hiver-ai-support-agent')

GOLDEN_PATH = r'D:/Progress/Hiver/hiver-ai-support-agent/data/golden_set.csv'

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
