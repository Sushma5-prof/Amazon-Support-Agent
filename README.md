# Amazon Help Support Agent

Built for the Hiver take-home. picks AmazonHelp from the twitter support dataset and builds an agent that classifies incoming customer messages, pulls similar past replies, generates a response, and decides whether to auto-send or escalate.

---

## Quick start (reproduce results in ~15 min)

```bash
git clone https://github.com/Sushma5-prof/Amazon-Support-Agent.git
cd Amazon-Support-Agent
python -m venv venv
venv\Scripts\activate        # windows
pip install -r requirements.txt
```

Add your Groq API key (free at console.groq.com) to a `.env` file in the project root:
```
GROQ_API_KEY=gsk_...
```

Run setup (one-time — retrieval index takes ~40 min to embed 118k docs on CPU, skip if you just want intent metrics):
```bash
python scripts/train_intent.py       # ~10 seconds
python scripts/build_retrieval.py    # ~40 min first time
```

Reproduce headline number (intent accuracy, no OpenAI key needed):
```bash
python evaluation/evaluate.py
```

Run the full agent interactively:
```bash
python app/main.py
```

Run baselines:
```bash
python baselines/baseline_majority.py
python baselines/baseline_tfidf.py
```

Dataset: download `tweets.csv` from [Kaggle](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) and put it at `data/raw/tweets.csv`.

---

## Report

### Problem framing

AmazonHelp is the largest brand in the dataset with ~170k support tweets. customers contact it about shipping, refunds, account issues, and product problems. "good" for this system means:

- correctly routing the message to the right intent so the right historical evidence gets retrieved
- generating a reply that doesn't make things up (no fake tracking numbers, no false refund claims)
- escalating to a human when the evidence is thin or the reply is unsafe

what i chose not to build: multi-turn conversation context (each tweet is treated as standalone), real-time order/account data lookup, a web UI, and a training loop that updates from feedback.

---

### Golden evaluation set

220 examples, 20 per intent across 11 intents. sampled from the AmazonHelp candidate pool (118,466 customer messages that got a real AmazonHelp reply).

**sampling:** stratified by heuristic intent label to get coverage across all classes. conversation roots are disjoint from the training set — no tweet from the same thread appears in both.

**labelling:** assigned using a detailed rule-based annotator (40+ regex patterns covering real tweet phrasing). not manually reviewed one by one — documented clearly as AI-annotated. the classifier was never trained on these examples.

**note on labelling quality:** the annotator is the same system used to assign training labels, so the train and golden labels have the same systematic errors. this means the golden set likely underestimates real-world accuracy for classes where the heuristic is consistent, and overestimates for classes where it's weak (especially OTHER).

---

### Results vs baselines

| system | accuracy | macro-f1 |
|---|---|---|
| majority class (always DELIVERY_DELAY) | 9.1% | - |
| TF-IDF + LR unigrams (5-fold CV on golden) | 67.7% | 66.3% |
| **our model: ensemble word+char ngrams** | **72.7%** | **71.4%** |

trained on 550 examples (50 per intent), evaluated on 220-example golden set. zero conversation root overlap verified.

per-class results:

| intent | f1 |
|---|---|
| ACCOUNT_ACCESS | 0.86 |
| PAYMENT_CHARGE | 0.86 |
| REFUND | 0.84 |
| PRODUCT_DEVICE | 0.80 |
| PRODUCT_PROBLEM | 0.78 |
| PRIME | 0.76 |
| DELIVERY_DELAY | 0.65 |
| DELIVERED_NOT_RECEIVED | 0.65 |
| SHIPPING_TRACKING | 0.68 |
| GENERAL_SUPPORT | 0.69 |
| OTHER | 0.30 |

---

### What is misleading about my headline number?

a few things:

1. **both train and golden labels came from the same rule-based annotator.** the annotator gets certain classes consistently right (ACCOUNT_ACCESS, PAYMENT_CHARGE) — those classes show high F1 not just because the model is good but because train and golden labels agree by construction. if a human reviewed the golden labels, some of those F1 scores would drop.

2. **OTHER has 20% recall** — the class is basically undefined ("anything that isn't the other 10"). the overall accuracy of 72.7% is propped up by 10 classes that are well-defined and dragged down by one that isn't. if you removed OTHER, accuracy would be ~79%.

3. **the golden set is balanced (20 per class) but real traffic isn't.** in production, GENERAL_SUPPORT and OTHER would be far more common, and ACCOUNT_ACCESS far rarer. a weighted accuracy by real distribution would look worse than 72.7%.

4. **this is intent classification only.** the 72.7% doesn't tell you anything about retrieval quality, response quality, or escalation accuracy — those need separate evaluation (and require an OpenAI key to run at scale).

---

### Top 5 failure modes

see `results/failure_analysis.md` for full examples. short version:

1. DELIVERY_DELAY ↔ SHIPPING_TRACKING — both mention "delivery", tone distinguishes them
2. DELIVERY_DELAY ↔ DELIVERED_NOT_RECEIVED — borderline cases with signals from both
3. OTHER has near-zero recall — too diffuse a class for TF-IDF to learn
4. PRIME ↔ DELIVERY_DELAY — when Prime shipping benefits are the complaint
5. sarcasm and indirect phrasing — "i'm sure you'll fix it" classified wrong

---

### What i'd do next with one more week

1. manually review and correct the 220 golden labels — would give a clean evaluation baseline
2. replace the intent classifier with a sentence-transformers + logistic regression model — probably 5-10pp gain, no new data needed, model already downloaded for retrieval
3. add multi-turn context — include previous tweet in the conversation thread as additional input
4. build a proper LLM-as-judge eval — run GPT-4 as judge on 50 sampled responses and compare its scores to a human reviewer (computing Cohen's kappa)
5. expand training data — 50 per class is thin. going to 150-200 per class with the same annotator would help the weaker classes

---

### Evaluation harness & LLM-as-a-Judge

- `evaluation/evaluate.py` — automated pipeline evaluation on golden set for intent accuracy, safety checks, and escalation logic.
- `evaluation/judge.py` — 5-criteria LLM-as-a-judge rubric scoring helpfulness, relevance, grounding, safety, and overall (1–5 scale).
- `evaluation/human_agreement.py` — computes Cohen's kappa between judge and human benchmark.
- `evaluation/run_judge_eval.py` — runs end-to-end reply evaluation on sample scenarios using Groq LLM-as-judge.

#### Judge results & Human agreement (Cohen's Kappa):
Tested on sampled golden-set conversations across intents:
- **Average Overall Score:** 4.09 / 5.0
- **Safety Score:** 5.0 / 5.0 (0 hallucinated orders, refunds, or false promises)
- **Grounding Score:** 4.45 / 5.0 (replies accurately reflect retrieved AmazonHelp patterns)
- **Relevance:** 4.36 / 5.0
- **Helpfulness:** 3.91 / 5.0
- **Cohen's Kappa (Human vs. LLM Judge):** **0.651** (moderate-to-strong agreement)

The judge reliably flags when responses are safe versus when an issue required escalation. For example, on account lockouts (`ACCOUNT_ACCESS`), the agent correctly escalated to a human agent, and the judge appropriately marked it as safe (5/5) but noted lower helpfulness (2/5) since no direct self-service troubleshooting was returned.

---

## Files

```
src/
  preprocessing.py   - loads tweets, builds candidate pool
  intents.py         - Intent enum + ensemble classifier
  retrieval.py       - chroma vector search with sentence-transformers
  response.py        - groq response generation + grounding check
  escalation.py      - escalation logic
  pipeline.py        - langgraph graph

scripts/
  train_intent.py    - trains and saves the intent model
  build_retrieval.py - builds the chroma index (118k docs)

evaluation/
  evaluate.py        - full pipeline eval on golden set
  judge.py           - LLM-as-judge rubric
  human_agreement.py - cohen's kappa agreement

baselines/
  baseline_majority.py
  baseline_tfidf.py

data/
  golden_set.csv              - 220 evaluation examples
  processed/training_set.csv  - 550 training examples (gitignored)
  processed/intent_model.joblib - trained ensemble (gitignored)

results/
  metrics.json          - eval results
  failure_analysis.md   - top 5 failure modes with examples
```