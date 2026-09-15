"""
Measures agreement between LLM judge scores and human scores.
Uses Cohen's kappa on the 'overall' dimension.

Usage:
    python evaluation/human_agreement.py

Edit the human_scores list below with your own ratings for the same
samples the judge scored.
"""
from sklearn.metrics import cohen_kappa_score


# LLM judge scores (overall 1-5) for the sampled responses
# run evaluation/judge.py to generate these
judge_scores = []

# Human scores for the same responses (fill these in manually)
human_scores = []


def calculate_agreement(judge, human):
    if len(judge) != len(human):
        raise ValueError("lists must be the same length")
    if not judge:
        print("no scores yet - run judge.py first and fill in human_scores")
        return None
    kappa = cohen_kappa_score(judge, human)
    print(f"Cohen's kappa: {kappa:.3f}")
    if kappa >= 0.8:
        print("strong agreement")
    elif kappa >= 0.6:
        print("moderate agreement")
    else:
        print("weak agreement - review judge rubric")
    return kappa


if __name__ == "__main__":
    calculate_agreement(judge_scores, human_scores)