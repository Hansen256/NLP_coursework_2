from __future__ import annotations

from dataclasses import dataclass
import re

import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HASHTAG_RE = re.compile(r"#\w+")
MENTION_RE = re.compile(r"@\w+")
REPEATED_CHARS_RE = re.compile(r"(.)\1{2,}")
EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)
TOKEN_RE = re.compile(r"[a-z']+")


@dataclass
class NoiseStats:
    urls_count: int
    hashtags_count: int
    mentions_count: int
    emojis_count: int
    repeated_chars_count: int
    avg_chars_per_tweet: float


def _ensure_nltk_resources() -> None:
    resource_paths = {
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
    }
    for check_path, resource_name in resource_paths.items():
        try:
            nltk.data.find(check_path)
        except LookupError:
            nltk.download(resource_name, quiet=True)


def compute_noise_stats(text_series: pd.Series) -> NoiseStats:
    texts = text_series.fillna("").astype(str)
    urls_count = int(texts.str.count(URL_RE).sum())
    hashtags_count = int(texts.str.count(HASHTAG_RE).sum())
    mentions_count = int(texts.str.count(MENTION_RE).sum())
    emojis_count = int(texts.str.count(EMOJI_RE).sum())
    repeated_chars_count = int(texts.str.count(REPEATED_CHARS_RE).sum())
    avg_chars = float(texts.str.len().mean()) if len(texts) else 0.0

    return NoiseStats(
        urls_count=urls_count,
        hashtags_count=hashtags_count,
        mentions_count=mentions_count,
        emojis_count=emojis_count,
        repeated_chars_count=repeated_chars_count,
        avg_chars_per_tweet=avg_chars,
    )


def preprocess_text_series(text_series: pd.Series) -> pd.DataFrame:
    _ensure_nltk_resources()

    texts = text_series.fillna("").astype(str).str.lower()
    texts = texts.str.replace(URL_RE, " ", regex=True)
    texts = texts.str.replace(HASHTAG_RE, " ", regex=True)
    texts = texts.str.replace(MENTION_RE, " ", regex=True)
    texts = texts.str.replace(EMOJI_RE, " ", regex=True)
    texts = texts.str.replace(r"[^a-z\s']", " ", regex=True)
    texts = texts.str.replace(r"\s+", " ", regex=True).str.strip()

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    clean_tokens: list[list[str]] = []
    clean_text: list[str] = []
    for text in texts.tolist():
        tokens = [tok for tok in TOKEN_RE.findall(text) if tok and tok not in stop_words]
        lemmas = [lemmatizer.lemmatize(tok) for tok in tokens if len(tok) > 1]
        clean_tokens.append(lemmas)
        clean_text.append(" ".join(lemmas))

    return pd.DataFrame({"clean_text": clean_text, "clean_tokens": clean_tokens})
