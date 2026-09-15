import sys
from pathlib import Path
import pandas as pd
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

GOLDEN_PATH = PROJECT_ROOT / "data" / "golden_set.csv"

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
