## Introduction

This project aims to build a movie recommendation system using the provided datasets of movies, ratings and tags. The goal is to explore different approaches to identifying similarity between movies, starting with simple genre-based methods and progressing to more expressive techniques using textual data. Over time, the system evolved into a more refined recommender combining content-based filtering, ranking, and performance optimisations.

While collaborative filtering is a common approach for recommender systems, this project focuses on a content-based method using genres and user-generated tags. This allows recommendations to be generated based on intrinsic properties of the movies, without requiring user-specific interaction data.

In addition to generating relevant recommendations, an important focus of the project was usability. This includes handling ambiguous user input, improving the quality of recommendations through ranking strategies, and developing a web-based interface to demonstrate the system in a more interactive and user-friendly way.

## Dataset and exploratory data analysis

The dataset consists of four main files; movies (<code>movies.csv</code>), ratings (<code>ratings.csv</code>), tags (<code>tags.csv</code>), and links (<code>links.csv</code>). The movies dataset contains approximately 86,000 entries, each associated with one or more genres (including entries labelled <code>(no genres listed)</code>). The ratings dataset is significantly larger, containing over 30 million rows, representing user ratings for movies. The tags dataset includes user-generated textual descriptions, while the links dataset provides mappings to external identifiers such as IMDb and TMDb.

An initial exploration of the dataset revealed several important characteristics. The ratings dataset is large and computationally expensive to process, which has implications for performance when aggregating statistics. Additionally, the dataset is sparse; not all movies have associated tags or a large number of ratings.

The genres column provides structured categorical information, making it suitable for encoding-based approaches. In contrast, the tags column contains unstructured text, offering an opportunity to apply natural language processing techniques to capture more nuanced similarities between movies.

## Baseline model: genre-based similarity

As an initial approach, a content-based recommender system was implemented using only the genre information available in the movies dataset. Each movie can belong to multiple genres, which were represented as a categorical feature.

To make this information usable for similarity comparison, the genres were transformed using one-hot encoding. This resulted in a binary feature vector for each movie, indicating the presence or absence of each genre. In total, approximately 20 unique genres were identified in the dataset.

A k-nearest neighbors (KNN) model was then applied using cosine similarity as the distance metric. This approach measures similarity based on the angle between feature vectors, making it suitable for high-dimensional sparse data such as one-hot encoded genres.

This baseline model provided a simple and interpretable way to identify similar movies. For example, movies sharing multiple genres would naturally appear closer in the feature space.

However, the approach also had clear limitations. Genre information alone is relatively coarse and does not capture finer details such as tone, themes, or narrative elements. As a result, movies within the same genre could still be very different in practice, leading to less precise recommendations.

## Improved model: TF-IDF with genres and tags

To improve the quality of recommendations, the model was extended to incorporate both genres and user-generated tags. Tags offer more expressive and descriptive information, capturing how users perceive and describe movies.

Genres and tags were combined into a single textual representation for each movie. To ensure that genre information remained a strong signal alongside the more variable tag data, genres were converted into text and deliberately repeated. This weighting strategy helped preserve the importance of structured genre information while still benefiting from the richer, more nuanced descriptions provided by tags.

The combined text was then transformed using Term Frequency-Inverse Document Frequency (TF-IDF) vectorisation. TF-IDF assigns higher importance to terms that are distinctive within a document, while reducing the influence of very common words. This makes it well-suited for capturing meaningful differences between movies based on their textual representation.

As expected with crowd-sourced data, the tag information is inherently noisy. Tags may include misspellings, informal phrasing, subjective opinions, or loosely structured descriptions rather than clearly defined categories. However, when considered collectively, they still provide valuable semantic signals. The TF-IDF transformation helps mitigate this noise by down-weighting less informative or overly common terms, allowing more distinctive words and phrases to contribute to the similarity measure.

To further improve expressiveness, both unigrams and bigrams were included in the vectorisation process. This enables the model to capture multi-word expressions such as “dark comedy” or “woman director,” which would otherwise be split into less informative individual tokens.

To control the dimensionality of the feature space, the number of TF-IDF features was limited to 5,000. This helps balance expressiveness and computational efficiency, while also reducing the risk of overfitting to very rare or noisy terms.

```python
def build_features(movies_model):
    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        min_df=2,
        ngram_range=(1, 2),  # include bigrams
    )

    X_tfidf = tfidf.fit_transform(movies_model["content"])
    return X_tfidf, tfidf
```

A KNN model with cosine similarity was then applied to these feature vectors. Compared to the genre-only baseline, this approach produced more nuanced and context-aware recommendations by leveraging both structured and unstructured information.

```python
def build_model(X_tfidf):
    model = NearestNeighbors(metric="cosine", algorithm="brute")
    model.fit(X_tfidf)

    return model
```

## Recommendation Refinement and Ranking

While the TF-IDF-based similarity model produced more context-aware recommendations than the genre-only baseline, the initial results still revealed some practical limitations. In particular, returning the nearest neighbours directly often led to recommendations that were either too obscure, lacked sufficient metadata, or were not consistently aligned with user expectations.

To address this, an additional refinement and ranking stage was introduced.

First, instead of returning the top n nearest neighbours directly, a larger candidate pool of similar movies was retrieved (e.g., the 20 closest matches). This allows for a broader selection of potentially relevant items before applying further filtering and ranking.

Next, several filtering steps were applied to remove low-quality or uninformative candidates. Movies with no genre information and no associated tags were excluded, as they provide insufficient content for meaningful comparison. Additionally, movies with very limited textual content or a low number of ratings were filtered out to improve reliability.

Finally, the remaining candidates were ranked using a weighted scoring function that combines average rating and rating count:

$$\text{weighted score} = \text{mean rating} \cdot \log(1 + \text{rating count})$$

```python
recommendations["weighted_score"] = recommendations["mean_rating"] * np.log1p(recommendations["rating_count"])
```

This approach balances quality and confidence. Movies with high average ratings and a substantial number of ratings are prioritised, while those with very few ratings are down-weighted. The use of a logarithmic transformation prevents extremely popular movies from dominating purely due to scale.

The final recommendations were then selected by sorting candidates primarily by weighted score and secondarily by similarity distance. This ensures that the results remain relevant to the input movie while also favouring higher-quality and more reliable recommendations.

While this ranking strategy improves overall recommendation quality, it introduces a known limitation; a tendency toward popularity bias. Widely rated and highly rated films (e.g., Forrest Gump or One Flew Over the Cuckoo’s Nest) are more likely to be recommended, even when less popular movies may be more closely aligned in terms of content. This highlights a trade-off between recommendation reliability and diversity.

## Performance optimisation

A key challenge in the project was the size of the ratings dataset, which contains over 30 million entries. Computing aggregate statistics such as mean rating and rating count on each run proved to be computationally expensive and significantly impacted performance (approximately 10–15 seconds per run).

To address this, a precomputation step was introduced. The required rating statistics were calculated once in advance and stored as a separate dataset. This allowed the recommender system to access precomputed values directly, rather than recomputing them during each execution.

In addition, the TF-IDF feature matrix and the trained KNN model were saved to disk using efficient storage formats. The sparse TF-IDF matrix was stored as a compressed .npz file, while the trained KNN model was serialised using the .joblib format, allowing it to be saved to disk and reloaded without retraining. This enabled faster startup times and avoided the need to rebuild the model pipeline for each run.

These optimisations resulted in a significant improvement in responsiveness, making the system more suitable for interactive use in the web-based interface.

## User interface

To demonstrate the recommender system in a more practical and interactive setting, a simple web-based interface was developed using Dash. This interface allows users to explore the model in a more accessible way.

The application enables users to search for a movie, handles ambiguous queries through a dropdown selection, and returns a configurable number of recommendations. Additional information such as average rating and rating count is displayed for each result, along with links to external sources (IMDb) for further exploration.

The interface also incorporates basic usability improvements, including a loading indicator and a responsive layout using CSS grid and fluid typography. This ensures that the application remains usable across different screen sizes.

Although relatively lightweight, the Dash app demonstrates how the recommender system can be integrated into a user-facing application, highlighting the transition from a purely experimental model to a more complete system.

## Evaluation and reflection

The final recommender system produces generally relevant and interpretable recommendations. Compared to the initial genre-based baseline, the TF-IDF approach combining genres and tags provides more nuanced and context-aware results. Incorporating a ranking step based on rating statistics further improves the perceived quality of recommendations by prioritising well-received and widely rated movies.

Qualitative testing with a range of input movies suggested that the system performs well for both popular and moderately well-known films. In many cases, the recommendations align with expectations in terms of genre, tone, and thematic similarity.

However, several limitations remain. One notable issue is a tendency toward popularity bias. Movies with a high number of ratings and strong average scores are frequently recommended even when they are not the closest match in terms of content similarity. This arises from the weighted ranking function, which prioritises widely rated films and can reduce diversity in the results.

Additionally, the system is purely content-based and does not incorporate user-specific preferences. As a result, recommendations are not personalised and may not reflect individual tastes. The quality of the tag data also introduces some variability, as user-generated text can be noisy and inconsistent.

Future improvements could include incorporating collaborative filtering techniques based on user ratings, as well as more advanced text processing methods to better handle noisy or sparse tag data.

## Conclusion

This project demonstrates the development of a content-based movie recommender system, evolving from a simple genre-based approach to a more refined model incorporating textual features and ranking strategies. By combining genres and user-generated tags using TF-IDF, the system is able to capture more detailed and meaningful similarities between movies.

Further improvements were achieved through a ranking step based on rating statistics and performance optimisations such as precomputation and model serialisation. The addition of a web-based interface highlights how the system can be used in a practical, user-facing context.

Overall, the project illustrates how relatively simple techniques, when combined thoughtfully and improved iteratively, can result in a practical and effective recommendation system.
