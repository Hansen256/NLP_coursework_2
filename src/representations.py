from __future__ import annotations

import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def build_bow(corpus: list[str], max_features: int = 20000, min_df: int = 2):
    vectorizer = CountVectorizer(max_features=max_features, min_df=min_df)
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix


def build_tfidf(corpus: list[str], max_features: int = 20000, min_df: int = 2):
    vectorizer = TfidfVectorizer(max_features=max_features, min_df=min_df)
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix


def corpus_vocabulary_size(tokenized_corpus: list[list[str]]) -> int:
    unique_tokens = {token for doc in tokenized_corpus for token in doc}
    return len(unique_tokens)


def top_terms_from_bow(vectorizer, matrix, top_k: int = 20) -> list[tuple[str, float]]:
    features = vectorizer.get_feature_names_out()
    scores = np.asarray(matrix.sum(axis=0)).ravel()
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(features[i], float(scores[i])) for i in top_idx]


def top_terms_from_tfidf(vectorizer, matrix, top_k: int = 20) -> list[tuple[str, float]]:
    features = vectorizer.get_feature_names_out()
    scores = np.asarray(matrix.mean(axis=0)).ravel()
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(features[i], float(scores[i])) for i in top_idx]


def train_word2vec(
    tokenized_corpus: list[list[str]],
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 5,
    workers: int = 1,
    epochs: int = 8,
    seed: int = 42,
) -> Word2Vec:
    filtered_corpus = [doc for doc in tokenized_corpus if doc]
    if not filtered_corpus:
        raise ValueError("Tokenized corpus is empty after preprocessing.")

    model = Word2Vec(
        sentences=filtered_corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        seed=seed,
    )
    return model


def most_similar_terms(
    w2v_model: Word2Vec,
    query_terms: list[str],
    topn: int = 10,
) -> dict[str, list[tuple[str, float]]]:
    results: dict[str, list[tuple[str, float]]] = {}
    for term in query_terms:
        if term in w2v_model.wv:
            items = w2v_model.wv.most_similar(term, topn=topn)
            results[term] = [(word, float(score)) for word, score in items]
        else:
            results[term] = []
    return results
