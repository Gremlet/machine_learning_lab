import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from scipy.sparse import load_npz


ARTIFACTS_DIR = Path("./artifacts")

MOVIES_READY_PATH = ARTIFACTS_DIR / "movies_model_ready.csv"
TFIDF_MATRIX_PATH = ARTIFACTS_DIR / "tfidf_matrix.npz"
NN_MODEL_PATH = ARTIFACTS_DIR / "nn_model.joblib"


def load_precomputed_data():
    """Load precomputed data and models."""
    print("Loading precomputed artifacts...")

    movies_model = pd.read_csv(MOVIES_READY_PATH)
    X_tfidf = load_npz(TFIDF_MATRIX_PATH)
    nn_model = joblib.load(NN_MODEL_PATH)

    print("Artifacts loaded successfully!\n")

    return movies_model, X_tfidf, nn_model


def recommend_movies(
    title, movies_df, feature_matrix, model, n=5, candidate_pool=20, min_ratings=50
):

    query = title.lower().strip()

    # 1. Exact full title match
    matches = movies_df[movies_df["title"].str.lower() == query]

    # 2. Exact cleaned title match
    if matches.empty:
        matches = movies_df[movies_df["title_clean"] == query]

    # 3. Partial cleaned title match
    if matches.empty:
        matches = movies_df[movies_df["title_clean"].str.contains(query, na=False)]

    if matches.empty:
        return f"Movie '{title}' not found."

    # If multiple matches, return choices instead of guessing
    if len(matches) > 1:
        return matches[["title", "genres"]].head(10)

    idx = matches.index[0]

    distances, indices = model.kneighbors(
        feature_matrix[idx], n_neighbors=candidate_pool + 1
    )

    rec_indices = indices.flatten()[1:]  # skip the movie itself
    rec_distances = distances.flatten()[1:]

    recommendations = movies_df.iloc[rec_indices][
        ["title", "genres", "tag", "content_word_count", "mean_rating", "rating_count"]
    ].copy()

    recommendations["distance"] = rec_distances

    # Remove truly useless recommendations
    recommendations = recommendations[
        ~(
            (recommendations["genres"] == "(no genres listed)")
            & (recommendations["tag"].isna())
        )
    ]

    recommendations = recommendations[recommendations["content_word_count"] >= 3]
    recommendations = recommendations[recommendations["rating_count"] >= min_ratings]

    # Weighted reranking
    recommendations["weighted_score"] = recommendations["mean_rating"] * np.log1p(
        recommendations["rating_count"]
    )

    recommendations = recommendations.sort_values(
        by=["weighted_score", "distance"], ascending=[False, True]
    )

    if recommendations.empty:
        return (
            f"No suitable recommendations found for '{title}'. "
            f"Try lowering min_ratings or using a different title."
        )

    return recommendations[
        ["title", "genres", "mean_rating", "rating_count", "weighted_score", "distance"]
    ].head(n)


def main():
    movies_model, X_tfidf, nn_model = load_precomputed_data()
    while True:
        movie_name = input("Enter a movie name: ").strip()

        results = recommend_movies(movie_name, movies_model, X_tfidf, nn_model)

        if isinstance(results, str):
            print(results)
            print("\nTry again.\n")

        elif "mean_rating" not in results.columns:
            print(
                "\nMultiple matches found. Please search again using one of these exact titles:\n"
            )
            print(results.to_string(index=False))
            print("\nTry again.\n")

        else:
            print("\nRecommended movies:\n")
            print(results.to_string(index=False))
            break


if __name__ == "__main__":
    main()
