"""Text preprocessing and TF-IDF utilities for pet/adopter matching."""
import re
from collections import Counter
from typing import Iterable, List

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CUSTOM_STOP_WORDS = set(ENGLISH_STOP_WORDS) | {
    "pet", "animal", "adopt", "adoption", "home", "looking", "need", "needs",
    "would", "like", "really", "one", "also", "get", "make"
}

def clean_text(text: str) -> str:
    """Lowercase text, remove punctuation/digits, and normalize whitespace."""
    text = str(text or "").lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def tokenize(text: str) -> List[str]:
    """Tokenize cleaned text and remove English/custom stop words."""
    return [t for t in clean_text(text).split() if t not in CUSTOM_STOP_WORDS and len(t) > 1]

def prepare_documents(items) -> List[str]:
    """Combine each pet bio and lifestyle tags into one NLP document."""
    return [
        f"{row.personality_bio} {row.lifestyle_tags.replace(',', ' ')}"
        for row in items.itertuples(index=False)
    ]

class PetNLP:
    """Fit one TF-IDF vocabulary and reuse it for all new queries."""
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            preprocessor=clean_text,
            stop_words=list(CUSTOM_STOP_WORDS),
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True
        )
        self.item_matrix = None
        self.feature_names = None

    def fit(self, documents: Iterable[str]):
        self.item_matrix = self.vectorizer.fit_transform(list(documents))
        self.feature_names = self.vectorizer.get_feature_names_out()
        return self

    def transform(self, texts):
        """Transform new free-text queries with the fitted vocabulary."""
        if self.item_matrix is None:
            raise RuntimeError("Call fit() before transform().")
        return self.vectorizer.transform(texts)

    def similarities(self, query: str) -> np.ndarray:
        query_vector = self.transform([query])
        return cosine_similarity(query_vector, self.item_matrix).ravel()

    def matched_keywords(self, query: str, top_n: int = 5) -> List[str]:
        """Return query terms present in the fitted TF-IDF vocabulary."""
        query_terms = tokenize(query)
        vocabulary = set(self.feature_names if self.feature_names is not None else [])
        counts = Counter(t for t in query_terms if t in vocabulary)
        return [term for term, _ in counts.most_common(top_n)]

    def item_keywords(self, item_index: int, query: str = "", top_n: int = 5) -> List[str]:
        """Explain a pet match using terms shared with the query."""
        query_terms = set(tokenize(query))
        row = self.item_matrix[item_index].toarray().ravel()
        ranked = np.argsort(row)[::-1]
        terms = []
        for idx in ranked:
            term = self.feature_names[idx]
            if row[idx] <= 0:
                break
            if not query_terms or any(q in term.split() or term in q for q in query_terms):
                terms.append(term)
            if len(terms) >= top_n:
                break
        return terms or [self.feature_names[i] for i in ranked[:top_n] if row[i] > 0]
