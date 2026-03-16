from __future__ import annotations

import pandas as pd


def build_method_comparison_table() -> pd.DataFrame:
    rows = [
        {
            "method": "Bag-of-Words",
            "advantages": "Simple, fast, interpretable term frequencies.",
            "limitations": "Ignores context and word order; favors frequent terms.",
            "suitable_applications": "Baselines, sparse linear models, quick diagnostics.",
        },
        {
            "method": "TF-IDF",
            "advantages": "Highlights informative terms; down-weights common words.",
            "limitations": "Still ignores semantics and long-range context.",
            "suitable_applications": "Search, ranking, topic-centric relevance scoring.",
        },
        {
            "method": "Word2Vec",
            "advantages": "Captures semantic similarity and neighborhood structure.",
            "limitations": "Needs sufficient data and tuning; less directly interpretable.",
            "suitable_applications": "Semantic exploration, similarity expansion, query enrichment.",
        },
    ]
    return pd.DataFrame(rows)


def recommend_pipeline() -> str:
    return (
        "Recommended pipeline: TF-IDF for document-level relevance ranking, "
        "supplemented by Word2Vec for semantic term exploration. "
        "Use Bag-of-Words as a baseline for interpretability checks."
    )
