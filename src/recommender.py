"""Popularity, NLP content-based, item-CF, and hybrid recommenders."""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .nlp_pipeline import PetNLP, prepare_documents

class PetRecommender:
    def __init__(self, items: pd.DataFrame, ratings: pd.DataFrame):
        self.items = items.reset_index(drop=True).copy()
        self.ratings = ratings.copy()
        self.nlp = PetNLP().fit(prepare_documents(self.items))
        self.item_similarity = None
        self.user_item = None
        self._fit_collaborative()

    def _fit_collaborative(self):
        self.user_item = self.ratings.pivot_table(
            index="user_id", columns="item_id", values="fit_rating", aggfunc="mean"
        )
        item_user = self.user_item.reindex(columns=self.items.item_id).fillna(0).T
        self.item_similarity = cosine_similarity(item_user)
        self.item_similarity = pd.DataFrame(
            self.item_similarity, index=self.items.item_id, columns=self.items.item_id
        )

    def _seen_items(self, user_id):
        return set(self.ratings.loc[self.ratings.user_id == user_id, "item_id"])

    def popularity(self, user_id=None, top_n=5, min_ratings=2):
        stats = self.ratings.groupby("item_id").fit_rating.agg(["mean", "count"])
        stats = stats[stats["count"] >= min_ratings]
        if user_id is not None:
            stats = stats.drop(index=list(self._seen_items(user_id)), errors="ignore")
        result = self.items.merge(stats, left_on="item_id", right_index=True)
        result = result.sort_values(["mean", "count"], ascending=False).head(top_n)
        return self._format(result, "mean", "Popularity score", "Average fit rating")

    def content_query(self, query, user_id=None, top_n=5):
        sims = self.nlp.similarities(query)
        frame = self.items.copy()
        frame["score"] = sims
        if user_id is not None:
            frame = frame[~frame.item_id.isin(self._seen_items(user_id))]
        frame = frame.sort_values("score", ascending=False).head(top_n).copy()
        frame["keywords"] = [
            ", ".join(self.nlp.item_keywords(self.items.index[self.items.item_id == row.item_id][0], query))
            for row in frame.itertuples()
        ]
        return self._format(frame, "score", "Similarity", "Matched keywords", include_keywords=True)

    def similar_profile(self, item_id, top_n=5):
        idx = self.items.index[self.items.item_id == item_id]
        if len(idx) == 0:
            raise ValueError(f"Unknown item_id: {item_id}")
        query = self.items.loc[idx[0], "personality_bio"] + " " + self.items.loc[idx[0], "lifestyle_tags"]
        return self.content_query(query, top_n=top_n+1).query("item_id != @item_id").head(top_n)

    def _cf_scores(self, user_id):
        """Item-based CF: weighted ratings from items the user has rated."""
        if user_id not in self.user_item.index:
            return pd.Series(0.0, index=self.items.item_id)
        user_ratings = self.user_item.loc[user_id].dropna()
        scores = {}
        for target in self.items.item_id:
            if target in user_ratings.index:
                scores[target] = 0.0
                continue
            sims = self.item_similarity.loc[target, user_ratings.index]
            denom = float(np.abs(sims).sum())
            scores[target] = float(np.dot(sims, user_ratings.values) / denom) if denom else 0.0
        return pd.Series(scores)

    def hybrid_query(self, query, user_id=None, top_n=5, alpha=0.65):
        """Hybrid = alpha * NLP similarity + (1-alpha) * normalized item-CF."""
        content = self.nlp.similarities(query)
        cf = self._cf_scores(user_id) if user_id is not None else pd.Series(0.0, index=self.items.item_id)
        cf_values = cf.reindex(self.items.item_id).fillna(0).to_numpy(dtype=float)
        if cf_values.max() > cf_values.min():
            cf_norm = (cf_values - cf_values.min()) / (cf_values.max() - cf_values.min())
        else:
            cf_norm = np.zeros_like(cf_values)
        combined = alpha * content + (1-alpha) * cf_norm
        frame = self.items.copy()
        frame["score"] = combined
        frame["nlp_score"] = content
        frame["cf_score"] = cf_norm
        if user_id is not None:
            frame = frame[~frame.item_id.isin(self._seen_items(user_id))]
        frame = frame.sort_values("score", ascending=False).head(top_n).copy()
        frame["keywords"] = [
            ", ".join(self.nlp.item_keywords(self.items.index[self.items.item_id == row.item_id][0], query))
            for row in frame.itertuples()
        ]
        return frame[["item_id","pet_name","species","score","nlp_score","cf_score","keywords"]]

    @staticmethod
    def _format(frame, score_col, score_label, extra_label, include_keywords=False):
        cols = ["item_id","pet_name","species",score_col]
        if include_keywords and "keywords" in frame:
            cols.append("keywords")
        out = frame[cols].copy()
        out = out.rename(columns={score_col: score_label})
        return out.reset_index(drop=True)
