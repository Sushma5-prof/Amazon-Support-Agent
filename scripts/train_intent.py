import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.intents import IntentClassifier
from src.preprocessing import train_dev_split

DATA_PATH = r'D:/Progress/Hiver/hiver-ai-support-agent/data/golden_set.csv'
MODEL_PATH = r'D:/Progress/Hiver/hiver-ai-support-agent/data/processed/intent_model.joblib'

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
