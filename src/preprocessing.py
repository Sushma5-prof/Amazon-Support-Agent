import pandas as pd
import re
from typing import List, Tuple
from pathlib import Path

RAW_COLUMNS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
]

def load_raw_data(csv_path: str) -> pd.DataFrame:
    """Load the full Kaggle CSV without skipping bad lines.

    Parameters
    ----------
    csv_path: str
        Absolute path to ``tweets.csv``.
    Returns
    -------
    pd.DataFrame
        DataFrame with the selected columns.
    """
    return pd.read_csv(
        csv_path,
        usecols=RAW_COLUMNS,
        low_memory=False,
        dtype=str,
    )

def clean_text(text: str) -> str:
    """Basic tweet cleaning used for candidate generation.
    - Remove URLs, mentions, hashtag symbols, HTML entities.
    - Keep alphanumeric and spaces, lower‑case.
    """
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"https?://\S+", " ", text)  # URLs
    text = re.sub(r"@\w+", " ", text)          # mentions
    text = re.sub(r"#", " ", text)              # drop # but keep word
    text = re.sub(r"&amp;|&lt;|&gt;", " ", text)  # HTML entities
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text)  # non‑alphanumeric
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

def extract_amazon_help(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows where the support account is AmazonHelp.
    inbound is stored as string 'True'/'False' in the CSV.
    """
    mask = (df["inbound"].str.lower() == "false") & df["author_id"].str.contains("AmazonHelp", case=False, na=False)
    return df[mask]

def build_candidate_pool(df: pd.DataFrame) -> pd.DataFrame:
    """Construct the candidate pool from customer tweets that received an AmazonHelp reply.
    Returns a DataFrame with required columns.
    """
    # Support tweets (outbound from AmazonHelp)
    support = extract_amazon_help(df)
    support_ids = set(support["tweet_id"].dropna().tolist())

    # Customer tweets that directly reference a support tweet via response_tweet_id
    # inbound is stored as string 'True'/'False'
    cust_mask = df["inbound"].str.lower() == "true"
    cust = df[cust_mask].copy()
    cust = cust[cust["response_tweet_id"].isin(support_ids)]

    # Clean and filter short messages
    cust["clean_text"] = cust["text"].apply(clean_text)
    cust = cust[cust["clean_text"].str.len() >= 20]

    # Remove exact duplicate customer texts
    cust = cust.drop_duplicates(subset=["clean_text"], keep="first")

    # Build conversation root mapping (fallback to own tweet_id if missing)
    if "conversation_root" in df.columns:
        root_map = df.set_index("tweet_id")["conversation_root"].to_dict()
    else:
        root_map = {}
    def resolve_root(tid):
        visited = set()
        cur = tid
        while cur in root_map and root_map[cur] != cur and cur not in visited:
            visited.add(cur)
            cur = root_map[cur]
        return cur if cur else tid
    cust["conversation_root"] = cust["tweet_id"].apply(resolve_root)

    # Historical replies per conversation root (support tweets only)
    reply_map = {}
    for _, row in support.iterrows():
        root = resolve_root(row["in_response_to_tweet_id"])
        reply_map.setdefault(root, []).append(row["text"].strip())
    cust["historical_replies"] = cust["conversation_root"].map(lambda r: reply_map.get(r, []))
    cust["num_historical_replies"] = cust["historical_replies"].apply(len)
    cust["text_length"] = cust["clean_text"].str.len()

    candidate = cust.rename(columns={
        "tweet_id": "customer_tweet_id",
        "author_id": "customer_id",
        "created_at": "customer_created_at",
        "text": "customer_text",
    })
    return candidate[[
        "customer_tweet_id",
        "customer_id",
        "customer_created_at",
        "customer_text",
        "conversation_root",
        "historical_replies",
        "num_historical_replies",
        "text_length",
    ]]

def train_dev_split(candidate_df: pd.DataFrame, train_frac: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Random split ensuring conversation roots are not shared between splits."""
    candidate_df = candidate_df.sample(frac=1, random_state=42).reset_index(drop=True)
    roots = candidate_df["conversation_root"].unique()
    train_roots = set(roots[: int(len(roots) * train_frac)])
    train = candidate_df[candidate_df["conversation_root"].isin(train_roots)]
    dev = candidate_df[~candidate_df["conversation_root"].isin(train_roots)]
    return train, dev