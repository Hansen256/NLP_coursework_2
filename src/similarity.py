from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def rank_similarity_to_reference(
    tweets: pd.Series,
    reference_text: str,
    max_features: int = 20000,
    min_df: int = 2,
) -> pd.DataFrame:
    tweet_texts = tweets.fillna("").astype(str).tolist()
    docs = tweet_texts + [reference_text]

    vectorizer = TfidfVectorizer(max_features=max_features, min_df=min_df)
    tfidf = vectorizer.fit_transform(docs)

    tweet_vectors = tfidf[:-1]
    reference_vector = tfidf[-1]
    scores = cosine_similarity(tweet_vectors, reference_vector).ravel()

    ranked = pd.DataFrame(
        {
            "tweet": tweet_texts,
            "similarity_score": scores,
        }
    ).sort_values("similarity_score", ascending=False)

    return ranked.reset_index(drop=True)
