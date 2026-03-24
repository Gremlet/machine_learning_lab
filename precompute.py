import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

DATA_DIR = Path("./data")
ARTIFACTS_DIR = Path("./artifacts")

MOVIES_PATH = DATA_DIR / "movies.csv"
TAGS_PATH = DATA_DIR / "tags.csv"
RATINGS_PATH = DATA_DIR / "ratings.csv"

MOVIES_MODEL_PATH = ARTIFACTS_DIR / "movies_model.csv"
RATINGS_SUMMARY_PATH = ARTIFACTS_DIR / "ratings_summary.csv"
MOVIES_READY_PATH = ARTIFACTS_DIR / "movies_model_ready.csv"
TFIDF_MATRIX_PATH = ARTIFACTS_DIR / "tfidf_matrix.npz"
TFIDF_VECTORIZER_PATH = ARTIFACTS_DIR / "tfidf_vectorizer.joblib"
NN_MODEL_PATH = ARTIFACTS_DIR / "nn_model.joblib"


def load_data():
    movies = pd.read_csv(MOVIES_PATH)
    tags = pd.read_csv(TAGS_PATH)
    ratings = pd.read_csv(RATINGS_PATH, usecols=["movieId", "rating"])

    return movies, tags, ratings


def preprocess_data(movies, tags):
    # Keep only necessary columns from movies and tags
    movies = movies[["movieId", "title", "genres"]].copy()
    tags = tags[["movieId", "tag"]].copy()

    # Clean tags
    tags = tags.dropna(subset=["tag"]).copy()
    tags["tag"] = tags["tag"].str.lower().str.strip()
    tags = tags[tags["tag"] != ""]
    tags = tags.drop_duplicates()

    # Group tags per movie
    tags_grouped = (
        tags.groupby("movieId")["tag"].apply(lambda x: " ".join(x)).reset_index()
    )

    # Merge movies with tags
    movies_with_tags = movies.merge(tags_grouped, on="movieId", how="left")

    # Clean genres into text
    movies_with_tags["genres_text"] = (
        movies_with_tags["genres"].str.replace("|", " ", regex=False).str.lower()
    )

    # Build content (weighted genres)
    movies_with_tags["content"] = (
        (movies_with_tags["genres_text"].fillna("") + " ")
        * 3  # we weigh genres heavier than tags by repeating them
        + movies_with_tags["tag"].fillna("")
    ).str.strip()

    # Clean title (remove year) for better search UX
    movies_with_tags["title_clean"] = (
        movies_with_tags["title"]
        .str.replace(r"\s*\(\d{4}\)$", "", regex=True)
        .str.lower()
        .str.strip()
    )

    # Get content length
    movies_with_tags["content_word_count"] = (
        movies_with_tags["content"].str.split().str.len()
    )

    # Remove unnecessary rows
    movies_model = movies_with_tags[
        ~(
            (movies_with_tags["genres"] == "(no genres listed)")
            & (movies_with_tags["tag"].isna())
        )
    ].copy()

    movies_model = movies_model[movies_model["content_word_count"] >= 3].copy()

    movies_model = movies_model.reset_index(drop=True)

    return movies_model


def build_features(movies_model):

    # TF-IDF
    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        min_df=2,
        ngram_range=(1, 2),  # include bigrams
    )

    X_tfidf = tfidf.fit_transform(movies_model["content"])
    return X_tfidf, tfidf


def build_model(X_tfidf):

    model = NearestNeighbors(metric="cosine", algorithm="brute")

    model.fit(X_tfidf)

    return model


def build_ratings_summary(ratings):
    ratings_summary = (
        ratings.groupby("movieId")
        .agg(
            mean_rating=("rating", "mean"),
            rating_count=("rating", "count"),
        )
        .reset_index()
    )

    return ratings_summary


def add_ratings(movies_model, ratings_summary):

    movies_model = movies_model.merge(ratings_summary, on="movieId", how="left")

    movies_model["mean_rating"] = movies_model["mean_rating"].fillna(0)
    movies_model["rating_count"] = movies_model["rating_count"].fillna(0)

    return movies_model


def save_artifacts(
    movies_model, movies_model_ready, ratings_summary, X_tfidf, tfidf, nn_model
):
    """Save all processed data and trained artifacts to disk."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    movies_model.to_csv(MOVIES_MODEL_PATH, index=False)
    print(f"Saved {MOVIES_MODEL_PATH}")

    ratings_summary.to_csv(RATINGS_SUMMARY_PATH, index=False)
    print(f"Saved {RATINGS_SUMMARY_PATH}")

    movies_model_ready.to_csv(MOVIES_READY_PATH, index=False)
    print(f"Saved {MOVIES_READY_PATH}")

    save_npz(TFIDF_MATRIX_PATH, X_tfidf)
    print(f"Saved {TFIDF_MATRIX_PATH}")

    joblib.dump(tfidf, TFIDF_VECTORIZER_PATH)
    print(f"Saved {TFIDF_VECTORIZER_PATH}")

    joblib.dump(nn_model, NN_MODEL_PATH)
    print(f"Saved {NN_MODEL_PATH}")


def precompute_all():
    """Run full preprocessing pipeline and save reusable artifacts."""
    print("Loading raw data...")
    movies, tags, ratings = load_data()

    print("Preprocessing movies and tags...")
    movies_model = preprocess_data(movies, tags)
    print(f"movies_model shape: {movies_model.shape}")

    print("Building TF-IDF features...")
    X_tfidf, tfidf = build_features(movies_model)
    print(f"TF-IDF matrix shape: {X_tfidf.shape}")

    print("Building nearest-neighbors model...")
    nn_model = build_model(X_tfidf)
    print("Nearest-neighbors model built.")

    print("Building ratings summary...")
    ratings_summary = build_ratings_summary(ratings)
    print(f"ratings_summary shape: {ratings_summary.shape}")

    print("Merging ratings into movies...")
    movies_model_ready = add_ratings(movies_model, ratings_summary)
    print(f"movies_model_ready shape: {movies_model_ready.shape}")

    print("Saving artifacts...")
    save_artifacts(
        movies_model=movies_model,
        movies_model_ready=movies_model_ready,
        ratings_summary=ratings_summary,
        X_tfidf=X_tfidf,
        tfidf=tfidf,
        nn_model=nn_model,
    )

    print("\nPrecompute complete!")


if __name__ == "__main__":
    precompute_all()
