import os
import sys
import json
import pandas as pd
sys.path.insert(0, r'D:/Progress/Hiver/hiver-ai-support-agent')

from src.pipeline import SupportAgent

GOLDEN_PATH = r'D:/Progress/Hiver/hiver-ai-support-agent/data/golden_set.csv'
RESULTS_PATH = r'D:/Progress/Hiver/hiver-ai-support-agent/results/metrics.json'

def evaluate():
    df = pd.read_csv(GOLDEN_PATH)
    agent = SupportAgent()

    correct_intent = 0
    escalation_preds = []
    escalation_labels = []
    grounding_pass = 0
    total = len(df)
    failures = []

    for _, row in df.iterrows():
        state = {"message": str(row["customer_text"])}
        result = agent.graph.invoke(state)

        predicted_intent = result.get("intent", "")
        true_intent = str(row.get("intent", ""))
        if predicted_intent == true_intent:
            correct_intent += 1
        else:
            failures.append({
                "text": row["customer_text"],
                "true_intent": true_intent,
                "predicted_intent": predicted_intent,
            })

        if result.get("grounding_passed", True):
            grounding_pass += 1

        if "should_escalate" in df.columns:
            escalation_preds.append(result.get("decision", "AUTO"))
            escalation_labels.append(str(row.get("should_escalate", "AUTO")))

    intent_acc = correct_intent / total if total else 0
    grounding_rate = grounding_pass / total if total else 0

    metrics = {
        "total": total,
        "intent_accuracy": round(intent_acc, 4),
        "grounding_pass_rate": round(grounding_rate, 4),
    }

    if escalation_preds:
        from sklearn.metrics import classification_report
        report = classification_report(escalation_labels, escalation_preds, output_dict=True)
        metrics["escalation"] = report

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Intent accuracy: {intent_acc:.4f}")
    print(f"Grounding pass rate: {grounding_rate:.4f}")
    print(f"Results written to {RESULTS_PATH}")

    if failures:
        print(f"\nSample failures ({min(5, len(failures))}):")
        for fail in failures[:5]:
            print(f"  TRUE={fail['true_intent']} PRED={fail['predicted_intent']} TEXT={fail['text'][:80]}")

if __name__ == "__main__":
    evaluate()
