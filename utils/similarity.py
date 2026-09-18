"""
similarity.py
-------------
Uses TF-IDF to convert incident error messages into vectors,
then uses cosine similarity to find similar historical incidents.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


_vectorizer = None
_embeddings = None


def build_embeddings(df, text_column="error_message"):
    """
    Convert error messages into TF-IDF vectors.
    """
    global _vectorizer, _embeddings

    texts = df[text_column].astype(str).tolist()

    _vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english"
    )

    _embeddings = _vectorizer.fit_transform(texts)

    return _embeddings


def find_most_similar(df, embeddings, query_index, top_n=1):
    """
    Find the most similar OTHER incident.
    """

    query_vector = embeddings[query_index]

    similarities = cosine_similarity(
        query_vector,
        embeddings
    )[0]

    # Don't compare the incident with itself
    similarities[query_index] = -1

    top_indices = np.argsort(similarities)[::-1][:top_n]

    results = []

    for idx in top_indices:
        results.append({
            "index": int(idx),
            "similarity_percent": round(
                float(similarities[idx]) * 100, 1
            ),
            "row": df.iloc[idx]
        })

    return results


def find_most_similar_for_text(query_text, df, embeddings):
    """
    Find the most similar historical incident
    for a new text description.
    """

    global _vectorizer

    query_vector = _vectorizer.transform([query_text])

    similarities = cosine_similarity(
        query_vector,
        embeddings
    )[0]

    best_idx = int(np.argmax(similarities))

    return {
        "index": best_idx,
        "similarity_percent": round(
            float(similarities[best_idx]) * 100, 1
        ),
        "row": df.iloc[best_idx]
    }