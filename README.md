# Amazon Help Support Agent

Built for the Hiver take-home. picks AmazonHelp from the twitter support dataset and builds an agent that classifies incoming customer messages, pulls similar past replies, generates a response, and decides whether to auto-send or escalate.

## Quick start

```bash
git clone <repo>
cd hiver-ai-support-agent
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Add your OpenAI key to `.env`:
```
OPENAI_API_KEY=sk-...
```

Then run setup (only need to do this once — takes ~40 min for the retrieval index because it embeds 118k docs):
```bash
python scripts/train_intent.py
python scripts/build_retrieval.py
```

Test it:
```bash
python app/main.py
```

Run evaluation against the golden set:
```bash
python evaluation/evaluate.py
```

## What it does

1. classifies the message into one of 11 intents (delivery delay, refund, tracking, etc.)
2. retrieves the 5 most similar historical AmazonHelp conversations from a vector index
3. generates a reply grounded in those examples using gpt-3.5-turbo
4. does a basic safety check on the reply (catches things like "i have refunded your order")
5. escalates if evidence is too thin or the reply fails the safety check

## Intent taxonomy

DELIVERY_DELAY, DELIVERED_NOT_RECEIVED, SHIPPING_TRACKING, REFUND, PAYMENT_CHARGE, PRIME, ACCOUNT_ACCESS, PRODUCT_DEVICE, PRODUCT_PROBLEM, GENERAL_SUPPORT, OTHER

## Results

trained on 550 examples (50 per intent), evaluated on a separate 220-example golden set (20 per intent, zero conversation root overlap with training)

| model | accuracy | macro-f1 |
|---|---|---|
| majority class | 9.1% | - |
| tfidf + LR unigrams | 67.7% | 66.3% |
| **ensemble word+char ngrams** | **72.7%** | **71.4%** |

## Files

```
data/
  raw/tweets.csv              - original kaggle dataset (not in repo, download separately)
  processed/training_set.csv  - 550 training examples
  processed/intent_model.joblib - trained ensemble
  processed/chroma_db/        - vector index (not in repo, rebuild with build_retrieval.py)
  golden_set.csv              - 220 held-out evaluation examples

src/
  preprocessing.py   - loads and cleans the tweets
  intents.py         - intent enum + classifier
  retrieval.py       - chroma vector search
  response.py        - openai response generation + grounding check
  escalation.py      - escalation logic
  pipeline.py        - langgraph graph wiring everything together

scripts/
  train_intent.py    - trains and saves the intent model
  build_retrieval.py - builds the chroma index

evaluation/
  evaluate.py        - runs the full pipeline on golden set

baselines/
  baseline_majority.py  - always predict most common class
  baseline_tfidf.py     - 5-fold cv on golden set
```

## Dataset

Kaggle: [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)

download tweets.csv and put it in data/raw/tweets.csv

## Notes on the golden set

sampled stratified (20 per intent), conversation roots are disjoint from training. labels were assigned using a detailed rule-based annotator — not manually reviewed one by one, so treat them as "good enough for a first eval" rather than perfect ground truth. the classifier was never trained on these examples.

the 72.7% accuracy is honest — trained on training_set.csv, evaluated on golden_set.csv, zero overlap verified.