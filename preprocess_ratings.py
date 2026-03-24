"""
Here, we preprocess ratings.csv and save the ratings summary
to a new .csv. This avoids rebuilding and processing the
entire ~1GB file on every run.

Run this once.
"""

import pandas as pd


def build_ratings_summary(ratings_path, output_path):
    ratings = pd.read_csv(ratings_path, usecols=["movieId", "rating"])

    ratings_summary = (
        ratings.groupby("movieId")
        .agg(mean_rating=("rating", "mean"), rating_count=("rating", "count"))
        .reset_index()
    )

    ratings_summary.to_csv(output_path, index=False)
    print(f"Saved {output_path}")


build_ratings_summary("./data/ratings.csv", "./precompute/ratings_summary.csv")
