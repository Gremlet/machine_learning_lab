import dash
from dash import html, dcc, Input, Output, State
import pandas as pd
import joblib
from scipy.sparse import load_npz

from recommender import recommend_movies

external_stylesheets = [
    "https://fonts.googleapis.com/css2?family=Limelight&family=Noto+Sans+Display:ital,wght@0,100..900;1,100..900&display=swap"
]


# Load artifacts once
movies_model = pd.read_csv("artifacts/movies_model_ready.csv")
X_tfidf = load_npz("artifacts/tfidf_matrix.npz")
nn_model = joblib.load("artifacts/nn_model.joblib")


app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=external_stylesheets,
)


def create_movie_card(row):
    imdb_url = None
    if pd.notna(row["imdbId"]):
        imdb_id = str(int(row["imdbId"])).zfill(7)
        imdb_url = f"https://www.imdb.com/title/tt{imdb_id}"

    return html.Div(
        style={
            "backgroundColor": "white",
            "padding": "20px",
            "borderRadius": "12px",
            "boxShadow": "0 2px 5px rgba(0,0,0,0.1)",
            "display": "flex",
            "flexDirection": "column",
            "justifyContent": "space-between",
            "minHeight": "280px",
            "textAlign": "center",
        },
        children=[
            html.Div(
                children=[
                    html.H4(
                        row["title"],
                        style={
                            "marginBottom": "15px",
                            "lineHeight": "1.3",
                            "minHeight": "60px",
                        },
                    ),
                    html.P(
                        f"Genres: {row['genres']}".replace("|", ", "),
                        style={
                            "marginBottom": "15px",
                            "lineHeight": "1.4",
                        },
                    ),
                    html.P(
                        f"⭐ {row['mean_rating']:.2f} ({int(row['rating_count'])} ratings)",
                        style={"marginBottom": "15px"},
                    ),
                ]
            ),
            (
                html.A(
                    "View on IMDb",
                    href=imdb_url,
                    target="_blank",
                    style={
                        "color": "#f5c518",
                        "fontWeight": "bold",
                        "marginTop": "10px",
                    },
                )
                if imdb_url
                else None
            ),
        ],
    )


def render_recommendations(results):
    return html.Div(
        [
            html.H3("Recommended movies:"),
            html.Div(
                [create_movie_card(row) for _, row in results.iterrows()],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(320px, 1fr))",
                    "gap": "24px",
                    "maxWidth": "1400px",
                    "margin": "0 auto",
                    "alignItems": "stretch",
                },
            ),
        ]
    )


app.layout = html.Div(
    style={
        "fontFamily": "Noto Sans Display",
        "textAlign": "center",
        "padding": "40px",
        "backgroundColor": "#f5f5f5",
        "minHeight": "100vh",
    },
    children=[
        html.H1(
            "🎬 Movie Recommender 🍿",
            style={
                "fontFamily": "Limelight",
                "fontSize": "clamp(2rem, 4vw, 4rem)",
            },
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.P("Type a movie and click recommend"),
                        dcc.Input(
                            id="movie-input",
                            type="text",
                            placeholder="e.g. Mean Girls",
                            style={
                                "width": "300px",
                                "padding": "5px",
                                "borderRadius": "5px",
                                "border": "1px solid #ccc",
                                "marginBottom": "32px",
                            },
                        ),
                    ],
                ),
                html.Div(
                    [
                        html.P("(Optional) How many recommendations would you like?"),
                        dcc.Slider(
                            id="n-slider",
                            min=1,
                            max=10,
                            step=1,
                            value=5,
                            marks={
                                1: "1",
                                5: "5",
                                10: "10",
                            },
                        ),
                    ],
                ),
            ],
            style={
                "display": "flex",
                "flexWrap": "wrap",
                "justifyContent": "center",
                "gap": "20px",
            },
        ),
        html.Button(
            "Recommend",
            id="submit-button",
            n_clicks=0,
            style={
                "marginTop": "20px",
                "padding": "10px 20px",
                "borderRadius": "5px",
                "border": "none",
                "backgroundColor": "#7f4bc4",
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
    State("n-slider", "value"),
    prevent_initial_call=True,
)
def search_movie(n_clicks, movie_name, rec_number):
    if not movie_name:
        return "", html.Div("Please enter a movie title.")

    results = recommend_movies(
        movie_name, movies_model, X_tfidf, nn_model, n=rec_number
    )

    # Movie not found / no suitable recommendations
    if isinstance(results, str):
        return "", html.Div(results)

    # If multiple matches show dropdown
    elif "mean_rating" not in results.columns:
        dropdown = html.Div(
            [
                html.H3("Multiple matches found:"),
                html.P("Please choose the movie you meant"),
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
    State("n-slider", "value"),
    prevent_initial_call=True,
)
def choose_match(selected_title, rec_number):
    if not selected_title:
        return ""

    results = recommend_movies(
        selected_title, movies_model, X_tfidf, nn_model, n=rec_number
    )

    if isinstance(results, str):
        return html.Div(results)

    return render_recommendations(results)


if __name__ == "__main__":
    app.run(debug=True)
