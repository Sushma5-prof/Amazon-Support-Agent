import os
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.intents import IntentClassifier
from src.preprocessing import train_dev_split

DATA_PATH = PROJECT_ROOT / "data" / "golden_set.csv"
MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "intent_model.joblib"

def main():
    df = pd.read_csv(DATA_PATH)
    # split while preserving conversation roots to avoid leakage
    train_df, dev_df = train_dev_split(df)
    clf = IntentClassifier()
    clf.train(train_df['customer_text'].tolist(), train_df['intent'].tolist())
    # dev accuracy for sanity
    dev_preds = [clf.predict(txt)['intent'] for txt in dev_df['customer_text']]
    acc = sum(p == t for p, t in zip(dev_preds, dev_df['intent'])) / len(dev_df)
    print(f'Development accuracy: {acc:.4f}')
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    clf.save(MODEL_PATH)
    print(f'Intent model saved to {MODEL_PATH}')

if __name__ == "__main__":
    main()
