from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.data_loader import load_tweets_dataframe
from src.evaluation import build_method_comparison_table, recommend_pipeline
from src.preprocessing import compute_noise_stats, preprocess_text_series
from src.representations import (
    build_bow,
    build_tfidf,
    corpus_vocabulary_size,
    most_similar_terms,
    top_terms_from_bow,
    top_terms_from_tfidf,
    train_word2vec,
)
from src.similarity import rank_similarity_to_reference


RANDOM_STATE = 42
MIN_REQUIRED_ROWS = 1000
SAMPLE_SIZE = int(os.getenv("PIPELINE_SAMPLE_SIZE", "10000"))
REFERENCE_TOPIC = "covid pandemic vaccine lockdown coronavirus public health"
QUERY_TERMS = ["covid", "vaccine", "lockdown", "pandemic", "coronavirus"]

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    df, text_col, source_path = load_tweets_dataframe(
        raw_dir=RAW_DIR,
        minimum_rows=MIN_REQUIRED_ROWS,
        sample_size=SAMPLE_SIZE,
        random_state=RANDOM_STATE,
    )

    print(f"Source file: {source_path}")
    print(f"Detected text column: {text_col}")
    print(f"Rows analyzed: {len(df)}")

    noise = compute_noise_stats(df[text_col])
    cleaned = preprocess_text_series(df[text_col])
    df = pd.concat([df.reset_index(drop=True), cleaned], axis=1)

    cleaned_out = PROCESSED_DIR / "covid_tweets_cleaned.csv"
    df.to_csv(cleaned_out, index=False)

    corpus = [str(x) for x in df["clean_text"].tolist()]
    tokenized_corpus = [list(x) for x in df["clean_tokens"].tolist()]

    bow_vectorizer, bow_matrix = build_bow(corpus=corpus, max_features=20000, min_df=2)
    tfidf_vectorizer, tfidf_matrix = build_tfidf(corpus=corpus, max_features=20000, min_df=2)
    w2v_model = train_word2vec(
        tokenized_corpus=tokenized_corpus,
        vector_size=100,
        window=5,
        min_count=5,
        workers=max(1, (os.cpu_count() or 2) - 1),
        epochs=8,
        seed=RANDOM_STATE,
    )

    ranked = rank_similarity_to_reference(
        tweets=df["clean_text"],
        reference_text=REFERENCE_TOPIC,
        max_features=20000,
        min_df=2,
    )
    comparison_df = build_method_comparison_table()

    comparison_path = OUTPUTS_DIR / "representation_comparison.csv"
    ranking_path = OUTPUTS_DIR / "tweet_similarity_ranking.csv"
    w2v_path = OUTPUTS_DIR / "word2vec_covid.model"
    summary_path = OUTPUTS_DIR / "analysis_summary.txt"

    comparison_df.to_csv(comparison_path, index=False)
    ranked.to_csv(ranking_path, index=False)
    w2v_model.save(str(w2v_path))

    bow_top_terms = top_terms_from_bow(bow_vectorizer, bow_matrix, top_k=20)
    tfidf_top_terms = top_terms_from_tfidf(tfidf_vectorizer, tfidf_matrix, top_k=20)
    similar_dict = most_similar_terms(w2v_model, QUERY_TERMS, topn=10)
    recommendation = recommend_pipeline()

    summary_lines = [
        "COVID-19 Public Discussion Analysis Summary",
        "",
        f"Source file: {source_path}",
        f"Detected text column: {text_col}",
        f"Rows analyzed: {len(df)}",
        "",
        "Noise stats:",
        f"- URLs: {noise.urls_count}",
        f"- Hashtags: {noise.hashtags_count}",
        f"- Mentions: {noise.mentions_count}",
        f"- Emojis: {noise.emojis_count}",
        f"- Repeated characters: {noise.repeated_chars_count}",
        "",
        "Representation snapshots:",
        f"- BoW vocab size: {len(bow_vectorizer.get_feature_names_out())}",
        f"- TF-IDF vocab size: {len(tfidf_vectorizer.get_feature_names_out())}",
        f"- Word2Vec vocab size: {len(w2v_model.wv)}",
        f"- Cleaned token vocab size: {corpus_vocabulary_size(tokenized_corpus)}",
        "",
        "Top BoW terms:",
    ]
    summary_lines.extend([f"- {term}: {count:.2f}" for term, count in bow_top_terms])
    summary_lines.append("")
    summary_lines.append("Top TF-IDF terms:")
    summary_lines.extend([f"- {term}: {score:.6f}" for term, score in tfidf_top_terms])
    summary_lines.append("")
    summary_lines.append("Word2Vec query expansion:")
    for query_term, items in similar_dict.items():
        if not items:
            summary_lines.append(f"- {query_term}: [no in-vocabulary neighbors]")
            continue
        top_items = ", ".join([f"{word} ({score:.3f})" for word, score in items[:5]])
        summary_lines.append(f"- {query_term}: {top_items}")
    summary_lines.append("")
    summary_lines.append("Recommendation:")
    summary_lines.append(recommendation)

    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print("Saved:")
    print(cleaned_out)
    print(comparison_path)
    print(ranking_path)
    print(w2v_path)
    print(summary_path)


if __name__ == "__main__":
    main()
