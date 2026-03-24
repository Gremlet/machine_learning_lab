import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


def load_data():
    movies = pd.read_csv("./data/movies.csv")
    tags = pd.read_csv("./data/tags.csv")
    ratings = pd.read_csv("./data/ratings.csv")

    return movies, tags, ratings


def preprocess_data(movies, tags):
    # Keep only necessary columns from tags
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

    # Clean genres
    movies_with_tags["genres_text"] = (
        movies_with_tags["genres"].str.replace("|", " ", regex=False).str.lower()
    )

    # Build content (weighted genres)
    movies_with_tags["content"] = (
        (movies_with_tags["genres_text"].fillna("") + " ")
        * 3  # we weigh genres heavier than tags
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


def add_ratings(movies_model, ratings):
    # Build ratings summary
    ratings_summary = (
        ratings.groupby("movieId")
        .agg(mean_rating=("rating", "mean"), rating_count=("rating", "count"))
        .reset_index()
    )

    # Merge into movies
    movies_model = movies_model.merge(ratings_summary, on="movieId", how="left")

    # Fill missing values
    movies_model["mean_rating"] = movies_model["mean_rating"].fillna(0)
    movies_model["rating_count"] = movies_model["rating_count"].fillna(0)

    return movies_model


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

    movies, tags, ratings = load_data()
    movies_model = preprocess_data(movies, tags)
    X_tfidf, tfidf = build_features(movies_model)
    nn_model = build_model(X_tfidf)
    movies_model = add_ratings(movies_model, ratings)

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
