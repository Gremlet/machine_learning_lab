# 🎬 Movie Recommender System

A content-based movie recommender system built using genres and user-generated tags, enhanced with ranking strategies and deployed with a simple web interface.

## Overview

This project implements a movie recommendation system using the MovieLens dataset. It evolves from a simple genre-based similarity model to a more advanced approach combining:

- TF-IDF vectorisation of genres and tags
- K-Nearest Neighbours (KNN) with cosine similarity
- A ranking layer based on ratings and popularity
- A lightweight Dash web application for interaction

The goal is to generate meaningful, context-aware recommendations while keeping the system interpretable and efficient.

---

## 🧠 Features

- Search for a movie by title
- Smart recommendations using content similarity
- Ranking based on rating quality and confidence
- Adjustable number of recommendations
- Handles ambiguous titles with dropdown selection
- Displays rating and number of ratings
- Links to IMDb for each movie
- Responsive UI (CSS Grid + fluid typography)

---

## 🛠 Tech Stack

- **Python**
- **Pandas / NumPy**
- **scikit-learn**
  - TF-IDF (`TfidfVectorizer`)
  - KNN (`NearestNeighbors`)
- **SciPy** (sparse matrices)
- **Dash** (web app)
- **Joblib** (model serialisation)

---

## ⚙️ How It Works

1. **Feature Engineering**
   - Genres and tags are combined into a single text representation
   - Genres are repeated to increase their importance
   - TF-IDF is applied with unigrams and bigrams

2. **Similarity Model**
   - KNN with cosine similarity finds similar movies

3. **Recommendation Refinement**
   - Retrieve a candidate pool of similar movies
   - Filter low-quality entries (few ratings, no metadata)
   - Rank using:
     $$\text{weighted score} = \text{mean rating} \cdot \log(1 + \text{rating count})$$

4. **Performance Optimisation**
   - Ratings are precomputed and stored
   - TF-IDF matrix and model are saved using `.npz` and `.joblib`

---

## 🏃‍♀️ How to Run

### 1. Clone the repo

```bash
git clone https://github.com/Gremlet/machine_learning_lab.git
cd machine_learning_lab
```

### 2. Download the dataset

Download `ml-latest.zip` from [here](https://grouplens.org/datasets/movielens/). After downloading, place the relevant CSV files (`movies.csv`, `ratings.csv`, `tags.csv`, `links.csv`) in a directory at the root called `data` before running the project.

```
data/
├── links.csv
├── movies.csv
├── ratings.csv
└── tags.csv
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Precompute data

```bash
python precompute.py
```

### 5. Run the app

- Command line:

```bash
python recommender.py
```

- Web app:

```bash
python app.py
```

Then open your browser at:
<code>http://127.0.0.1:8050</code>

## 🚀 Deployment

The app is also deployed and available here:

https://ml-movie-recs.onrender.com/

Note that Render can take up to a minute to load the app.

## ⚠️ Limitations

- Tends toward popularity bias (highly rated movies are favoured)
- Not personalised (content-based only)
- Tag data can be noisy and inconsistent

## 🔮 Future Improvements

- Add collaborative filtering using user ratings
- Improve text processing (e.g. embeddings)
- Personalised recommendations
- Poster integration via TMDb API
