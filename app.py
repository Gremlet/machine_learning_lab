import dash
from dash import html, dcc, Input, Output, State
import pandas as pd
import joblib
from scipy.sparse import load_npz

from recommender import recommend_movies


# Load artifacts once
movies_model = pd.read_csv("artifacts/movies_model_ready.csv")
X_tfidf = load_npz("artifacts/tfidf_matrix.npz")
nn_model = joblib.load("artifacts/nn_model.joblib")


app = dash.Dash(__name__, suppress_callback_exceptions=True)


def render_recommendations(results):
    return html.Div(
        [
            html.H3("Recommended movies:"),
            html.Div(
                [
                    html.Div(
                        style={
                            "backgroundColor": "white",
                            "padding": "15px",
                            "margin": "10px auto",
                            "width": "400px",
                            "borderRadius": "8px",
                            "boxShadow": "0 2px 5px rgba(0,0,0,0.1)",
                        },
                        children=[
                            html.H4(row["title"]),
                            html.P(f"Genres: {row['genres']}"),
                            html.P(
                                f"⭐ {row['mean_rating']:.2f} ({int(row['rating_count'])} ratings)"
                            ),
                        ],
                    )
                    for _, row in results.iterrows()
                ]
            ),
        ]
    )


app.layout = html.Div(
    style={
        "fontFamily": "Arial",
        "textAlign": "center",
        "padding": "40px",
        "backgroundColor": "#f5f5f5",
        "minHeight": "100vh",
    },
    children=[
        html.H1("🎬 Movie Recommender"),
        html.P("Type a movie and click recommend"),
        dcc.Input(
            id="movie-input",
            type="text",
            placeholder="e.g. Mean Girls",
            style={
                "width": "300px",
                "padding": "10px",
                "marginRight": "10px",
                "borderRadius": "5px",
                "border": "1px solid #ccc",
            },
        ),
        html.Button(
            "Recommend",
            id="submit-button",
            n_clicks=0,
            style={
                "padding": "10px 20px",
                "borderRadius": "5px",
                "border": "none",
                "backgroundColor": "#007BFF",
                "color": "white",
                "cursor": "pointer",
            },
        ),
        dcc.Loading(
            id="loading",
            type="circle",
            children=[
                html.Div(id="match-container", style={"marginTop": "30px"}),
                html.Div(id="output", style={"marginTop": "40px"}),
            ],
        ),
    ],
)


@app.callback(
    Output("match-container", "children"),
    Output("output", "children"),
    Input("submit-button", "n_clicks"),
    State("movie-input", "value"),
    prevent_initial_call=True,
)
def search_movie(n_clicks, movie_name):
    if not movie_name:
        return "", html.Div("Please enter a movie title.")

    results = recommend_movies(movie_name, movies_model, X_tfidf, nn_model)

    # Movie not found / no suitable recommendations
    if isinstance(results, str):
        return "", html.Div(results)

    # If multiple matches show dropdown
    elif "mean_rating" not in results.columns:
        dropdown = html.Div(
            [
                html.H3("Multiple matches found:"),
                dcc.Dropdown(
                    id="match-dropdown",
                    options=[
                        {
                            "label": f"{row['title']} — {row['genres']}",
                            "value": row["title"],
                        }
                        for _, row in results.iterrows()
                    ],
                    placeholder="Choose the movie you meant",
                    style={"width": "500px", "margin": "0 auto"},
                ),
            ]
        )
        return dropdown, ""

    # If one match, show recommendations directly
    else:
        return "", render_recommendations(results)


@app.callback(
    Output("output", "children", allow_duplicate=True),
    Input("match-dropdown", "value"),
    prevent_initial_call=True,
)
def choose_match(selected_title):
    if not selected_title:
        return ""

    results = recommend_movies(selected_title, movies_model, X_tfidf, nn_model)

    if isinstance(results, str):
        return html.Div(results)

    return render_recommendations(results)


if __name__ == "__main__":
    app.run(debug=True)
