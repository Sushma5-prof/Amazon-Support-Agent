import sys
import pandas as pd
from collections import Counter
sys.path.insert(0, r'D:/Progress/Hiver/hiver-ai-support-agent')

GOLDEN_PATH = r'D:/Progress/Hiver/hiver-ai-support-agent/data/golden_set.csv'

def main():
    df = pd.read_csv(GOLDEN_PATH)
    counts = Counter(df["intent"])
    majority = counts.most_common(1)[0][0]
    preds = [majority] * len(df)
    acc = sum(p == t for p, t in zip(preds, df["intent"])) / len(df)
    print(f"Majority class: {majority}")
    print(f"Majority baseline accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
