"""Offline evaluation and chart generation."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .recommender import PetRecommender

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

def precision_at_k(recommended, relevant, k=5):
    recommended = list(recommended)[:k]
    if not recommended:
        return 0.0
    return len(set(recommended) & set(relevant)) / k

def evaluate(test_fraction=0.20, k=5, seed=42):
    items = pd.read_csv(DATA / "items.csv")
    ratings = pd.read_csv(DATA / "ratings.csv")
    rng = np.random.default_rng(seed)
    train_parts, test_parts = [], []
    for _, group in ratings.groupby("user_id"):
        indices = group.index.to_numpy().copy()
        rng.shuffle(indices)
        n_test = max(2, int(round(len(indices) * test_fraction)))
        test_idx = set(indices[:n_test])
        test_parts.append(group.loc[list(test_idx)])
        train_parts.append(group.loc[~group.index.isin(test_idx)])
    train = pd.concat(train_parts).reset_index(drop=True)
    test = pd.concat(test_parts).reset_index(drop=True)
    model = PetRecommender(items, train)

    rows=[]
    actual, predicted = [], []
    for user_id, user_test in test.groupby("user_id"):
        relevant = user_test.loc[user_test.fit_rating >= 4, "item_id"].tolist()
        # Use a generic home/lifestyle profile derived from held-out pet descriptions.
        query = " ".join(
            items.loc[items.item_id.isin(user_test.item_id), "personality_bio"].tolist()
        )
        pop = model.popularity(user_id=user_id, top_n=k)["item_id"].tolist()
        content = model.content_query(query, user_id=user_id, top_n=k)["item_id"].tolist()
        hybrid = model.hybrid_query(query, user_id=user_id, top_n=k)["item_id"].tolist()
        rows.extend([
            {"user_id":user_id,"method":"Popularity","precision_at_5":precision_at_k(pop,relevant,k)},
            {"user_id":user_id,"method":"NLP Content-Based","precision_at_5":precision_at_k(content,relevant,k)},
            {"user_id":user_id,"method":"Hybrid","precision_at_5":precision_at_k(hybrid,relevant,k)},
        ])
        # Rating prediction proxy for held-out user-item pairs using item similarity.
        for rec in user_test.itertuples():
            scores = model._cf_scores(user_id)
            pred = float(scores.get(rec.item_id, train.fit_rating.mean()))
            predicted.append(max(1.0,min(5.0,pred)))
            actual.append(float(rec.fit_rating))

    per_user = pd.DataFrame(rows)
    summary = per_user.groupby("method").precision_at_5.mean().reset_index()
    rmse = float(np.sqrt(mean_squared_error(actual,predicted)))
    mae = float(mean_absolute_error(actual,predicted))
    RESULTS.mkdir(exist_ok=True)
    summary.to_csv(RESULTS/"evaluation_metrics.csv",index=False)
    per_user.to_csv(RESULTS/"per_user_evaluation.csv",index=False)

    # Chart 1: rating distribution
    plt.figure(figsize=(7,4))
    ratings.fit_rating.value_counts().sort_index().plot(kind="bar")
    plt.title("Distribution of Pet Fit Ratings")
    plt.xlabel("Fit rating (1-5)")
    plt.ylabel("Number of ratings")
    plt.tight_layout()
    plt.savefig(RESULTS/"rating_distribution.png",dpi=160)
    plt.close()

    # Chart 2: Precision@5 comparison
    plt.figure(figsize=(7,4))
    plt.bar(summary.method,summary.precision_at_5)
    plt.title("Recommendation Methods: Precision@5")
    plt.xlabel("Method")
    plt.ylabel("Mean Precision@5")
    plt.ylim(0,1)
    plt.tight_layout()
    plt.savefig(RESULTS/"precision_at_5_comparison.png",dpi=160)
    plt.close()

    return summary, rmse, mae, len(train), len(test)

if __name__ == "__main__":
    summary, rmse, mae, n_train, n_test = evaluate()
    print("Evaluation summary")
    print(summary.to_string(index=False))
    print(f"RMSE: {rmse:.3f} | MAE: {mae:.3f}")
    print(f"Train ratings: {n_train} | Test ratings: {n_test}")
