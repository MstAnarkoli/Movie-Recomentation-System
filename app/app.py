# Build a clean Streamlit movie recommendation application

# Import the libraries needed by the application
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from scipy.sparse import load_npz


# Configure the Streamlit page
st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide"
)


# Find the project root directory
PROJECT_DIR = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------
# Load movie metadata
# ---------------------------------------------------------

@st.cache_data
def load_movies():

    return pd.read_csv(
        PROJECT_DIR / "data" / "processed" / "movies_content.csv"
    )


movies_content = load_movies()


# ---------------------------------------------------------
# Load content-based model
# ---------------------------------------------------------

@st.cache_resource
def load_content_model():

    vectorizer = joblib.load(
        PROJECT_DIR / "models" / "tfidf_vectorizer.joblib"
    )

    matrix = load_npz(
        PROJECT_DIR / "models" / "tfidf_matrix.npz"
    )

    return vectorizer, matrix


tfidf_vectorizer, tfidf_matrix = load_content_model()


# ---------------------------------------------------------
# Load also-rated model
# ---------------------------------------------------------

@st.cache_resource
def load_also_rated_model():

    matrix = load_npz(
        PROJECT_DIR / "models" / "movie_user_matrix.npz"
    )

    ids = np.load(
        PROJECT_DIR / "models" / "movie_ids_aw.npy"
    )

    index = joblib.load(
        PROJECT_DIR / "models" / "movie_to_index_aw.joblib"
    )

    counts = pd.read_csv(
        PROJECT_DIR / "models" / "movie_user_counts.csv"
    )["user_count"].to_numpy()

    return matrix, ids, index, counts


(
    movie_user_matrix_aw,
    movie_ids_aw,
    movie_to_index_aw,
    movie_user_counts_aw
) = load_also_rated_model()


# ---------------------------------------------------------
# Load collaborative model
# ---------------------------------------------------------

@st.cache_resource
def load_collaborative_model():

    svd = joblib.load(
        PROJECT_DIR / "models" / "collaborative_svd.joblib"
    )

    user_index = joblib.load(
        PROJECT_DIR / "models" / "user_to_index.joblib"
    )

    movie_index = joblib.load(
        PROJECT_DIR / "models" / "movie_to_index.joblib"
    )

    users = np.load(
        PROJECT_DIR / "models" / "user_ids.npy"
    )

    movies = np.load(
        PROJECT_DIR / "models" / "movie_ids.npy"
    )

    factors = np.load(
        PROJECT_DIR / "models" / "user_factors.npy"
    )

    return (
        svd,
        user_index,
        movie_index,
        users,
        movies,
        factors
    )


(
    collaborative_svd,
    user_to_index,
    movie_to_index,
    user_ids,
    movie_ids,
    user_factors
) = load_collaborative_model()


# ---------------------------------------------------------
# Content-based recommender
# ---------------------------------------------------------

def recommend_similar_movies(movie_id, n=10):

    movie_index = movies_content.index[
        movies_content["movieId"] == movie_id
    ][0]

    movie_vector = tfidf_matrix[movie_index]

    similarity_scores = (
        movie_vector @ tfidf_matrix.T
    ).toarray().ravel()

    similar_indices = (
        similarity_scores.argsort()[-n-1:][::-1]
    )

    similar_indices = [
        index
        for index in similar_indices
        if index != movie_index
    ][:n]

    recommendations = movies_content.iloc[
        similar_indices
    ][
        ["movieId", "title", "genres_clean"]
    ].copy()

    recommendations["similarity"] = (
        similarity_scores[similar_indices]
    )

    return recommendations


# ---------------------------------------------------------
# Also-rated recommender
# ---------------------------------------------------------

def recommend_also_rated(movie_id, n=3):

    if movie_id not in movie_to_index_aw:
        return pd.DataFrame()

    movie_index = movie_to_index_aw[movie_id]

    user_indices = movie_user_matrix_aw[
        movie_index
    ].indices

    related_movies = movie_user_matrix_aw[
        :,
        user_indices
    ]

    cooccurrence_counts = np.asarray(
        related_movies.sum(axis=1)
    ).ravel()

    cooccurrence_counts[movie_index] = 0

    selected_movie_count = (
        movie_user_counts_aw[movie_index]
    )

    similarity_scores = (
        cooccurrence_counts
        / np.sqrt(
            selected_movie_count
            * movie_user_counts_aw
        )
    )

    top_indices = np.argsort(
        similarity_scores
    )[-n:][::-1]

    recommendations = pd.DataFrame({
        "movieId": movie_ids_aw[top_indices],
        "also_rated_score": similarity_scores[
            top_indices
        ]
    })

    recommendations = recommendations.merge(
        movies_content[
            ["movieId", "title", "genres_clean"]
        ],
        on="movieId",
        how="left"
    )

    return recommendations[
        [
            "title",
            "genres_clean",
            "also_rated_score"
        ]
    ]


# ---------------------------------------------------------
# Personalized recommender
# ---------------------------------------------------------

def recommend_for_user(user_id, n=3):

    if user_id not in user_to_index:
        return pd.DataFrame()

    user_index = user_to_index[user_id]

    user_vector = user_factors[user_index]

    movie_scores = (
        user_vector @ collaborative_svd.components_
    )

    ranked_indices = np.argsort(
        movie_scores
    )[::-1]

    top_indices = ranked_indices[:n]

    recommendations = pd.DataFrame({
        "movieId": movie_ids[top_indices],
        "recommendation_score": movie_scores[
            top_indices
        ]
    })

    recommendations = recommendations.merge(
        movies_content[
            ["movieId", "title", "genres_clean"]
        ],
        on="movieId",
        how="left"
    )

    return recommendations[
        [
            "title",
            "genres_clean",
            "recommendation_score"
        ]
    ]


# ---------------------------------------------------------
# Display recommendation cards
# ---------------------------------------------------------

def display_movie_cards(recommendations, score_column, score_label):
    # Display up to three recommendation cards in a clean three-column layout
    if recommendations.empty:
        st.info("No recommendations available.")
        return

    recommendations = recommendations.head(3)
    columns = st.columns(3, gap="medium")

    for index, (_, row) in enumerate(recommendations.iterrows()):
        with columns[index]:
            with st.container(border=True):
                
                title = str(row["title"])
                st.markdown(
                    f'<div class="movie-title">{title}</div>',
                    unsafe_allow_html=True
                )

                genres = row["genres_clean"]
                if pd.isna(genres) or not genres:
                    genres = "Genre unavailable"

                st.caption(str(genres).replace("|", " · "))

                score = row[score_column]
                st.metric(score_label, f"{score:.3f}")


# ---------------------------------------------------------
# Frontend styling
# ---------------------------------------------------------

st.markdown(
    """
    <style>
        .stApp { background: #050505; color: #f5f5f5; }
        [data-testid="stAppViewContainer"] { background: #050505; }
        [data-testid="stHeader"] { background: #050505; }
        [data-testid="stMainBlockContainer"] { background: #050505; }
        .block-container { max-width: 1250px; padding-top: 2rem; padding-bottom: 2.5rem; }

        .hero-title { color: #E50914; font-size: 3.2rem; font-weight: 900; letter-spacing: -0.05em; line-height: 1; margin-bottom: .35rem; }
        .hero-subtitle { color: #b3b3b3; font-size: 1.05rem; margin-bottom: 1.3rem; }
        .movie-title { color: #fff; font-size: 1.05rem; font-weight: 750; line-height: 1.3; min-height: 2.75rem; margin: .15rem 0 .55rem 0; }
        .section-kicker { color: #E50914; font-size: .72rem; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; margin-bottom: .1rem; }
        .welcome-text { color: #c7c7c7; font-size: 1.05rem; line-height: 1.65; }
        .accent-line { height: 4px; width: 76px; border-radius: 999px; background: #E50914; margin: .4rem 0 1.5rem 0; }

        [data-testid="stVerticalBlockBorderWrapper"] { background: #111111; border: 1px solid #292929; border-radius: 12px; }
        [data-testid="stVerticalBlockBorderWrapper"]:hover { border-color: #E50914; }

        h1, h2, h3, h4, p, label { color: #f5f5f5; }
        [data-testid="stCaptionContainer"] { color: #a7a7a7; }

        div[data-baseweb="select"] > div { background: #181818; border: 1px solid #444; color: #fff; border-radius: 8px; }
        div[data-baseweb="select"] > div:hover { border-color: #E50914; }
        div[data-baseweb="popover"] { background: #181818; }
        [data-testid="stNumberInput"] input { background: #181818; color: #fff; border: 1px solid #444; }

        [data-testid="stMetric"] { background: #181818; border-radius: 8px; padding: .65rem; }
        [data-testid="stMetricLabel"] { color: #a7a7a7; }
        [data-testid="stMetricValue"] { color: #fff; font-size: 1.2rem; }

        hr { border-color: #292929 !important; opacity: 1; }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# Application header
# ---------------------------------------------------------

st.markdown(
    '<div class="hero-title">🎬 CineMatch</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-subtitle">'
    'Discover movies through content similarity, community behavior, '
    'and personalized recommendations.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)


# ---------------------------------------------------------
# Movie selection
# ---------------------------------------------------------

st.markdown(
    '<div class="section-kicker">Movie discovery</div>',
    unsafe_allow_html=True
)

st.subheader("Choose a movie")

selected_movie = st.selectbox(
    "Start typing a movie title to search",
    movies_content["title"].sort_values().tolist(),
    index=None,
    placeholder="Search or select a movie...",
    label_visibility="collapsed"
)


# ---------------------------------------------------------
# Recommendation interface
# ---------------------------------------------------------

if selected_movie:

    # Get the MovieLens ID of the selected movie
    selected_movie_id = movies_content.loc[
        movies_content["title"] == selected_movie,
        "movieId"
    ].iloc[0]

    # Get information about the selected movie
    selected_movie_info = movies_content[
        movies_content["movieId"] == selected_movie_id
    ].iloc[0]

    # -------------------------------------------------
    # Selected movie
    # -------------------------------------------------

    st.divider()

    with st.container(border=True):
        st.markdown(
            '<div class="section-kicker">Selected movie</div>',
            unsafe_allow_html=True
        )

        st.header(selected_movie_info["title"])

        genres = selected_movie_info["genres_clean"]

        if pd.isna(genres) or not genres:
            genres = "Genre unavailable"

        st.caption(str(genres).replace("|", " · "))

        st.write(
            "Explore movies with similar content, movies rated by similar "
            "users, and personalized recommendations."
        )

    # -------------------------------------------------
    # Similar movies
    # -------------------------------------------------

    st.divider()

    st.markdown(
        '<div class="section-kicker">Content-based recommendations</div>',
        unsafe_allow_html=True
    )
    st.header("🍿 Movies You May Like")
    st.caption("Based on movie genres and user-generated tags.")

    similar_movies = recommend_similar_movies(
        selected_movie_id,
        n=3
    )

    display_movie_cards(
        similar_movies,
        "similarity",
        "Similarity"
    )

    # -------------------------------------------------
    # Also rated
    # -------------------------------------------------

    st.divider()

    st.markdown(
        '<div class="section-kicker">Community behavior</div>',
        unsafe_allow_html=True
    )
    st.header("👥 People Also Rated")
    st.caption(
        "Movies frequently rated by users who also rated this movie."
    )

    also_rated = recommend_also_rated(
        selected_movie_id,
        n=3
    )

    display_movie_cards(
        also_rated,
        "also_rated_score",
        "Affinity"
    )

    # -------------------------------------------------
    # Personalized recommendations
    # -------------------------------------------------

    st.divider()

    st.markdown(
        '<div class="section-kicker">Personalization</div>',
        unsafe_allow_html=True
    )
    st.header("🧠 Recommended For You")
    st.caption(
        "Personalized recommendations based on learned user preferences."
    )

    selected_user_id = st.number_input(
        "MovieLens User ID",
        min_value=int(user_ids.min()),
        max_value=int(user_ids.max()),
        value=100,
        step=1
    )

    personalized = recommend_for_user(
        selected_user_id,
        n=3
    )

    display_movie_cards(
        personalized,
        "recommendation_score",
        "Recommendation score"
    )


else:

    # -------------------------------------------------
    # Welcome section
    # -------------------------------------------------

    st.divider()

    with st.container(border=True):
        st.markdown(
            '<div class="section-kicker">Welcome</div>',
            unsafe_allow_html=True
        )

        st.header("Welcome to CineMatch 👋")

        st.markdown(
            '<div class="welcome-text">'
            'Choose a movie above to discover three recommendations from '
            'each recommendation engine.'
            '</div>',
            unsafe_allow_html=True
        )

    st.divider()

    st.markdown(
        '<div class="section-kicker">Recommendation engines</div>',
        unsafe_allow_html=True
    )
    st.header("How CineMatch works")

    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        with st.container(border=True):
            st.subheader("🎯 Similar Movies")
            st.write(
                "Find movies with similar genres and tags using "
                "TF-IDF and cosine similarity."
            )

    with col2:
        with st.container(border=True):
            st.subheader("👥 People Also Rated")
            st.write(
                "Discover movies that are commonly rated by the "
                "same users."
            )

    with col3:
        with st.container(border=True):
            st.subheader("🧠 Personalized")
            st.write(
                "Generate recommendations for a specific MovieLens "
                "user using collaborative filtering."
            )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.divider()

st.caption(
    "CineMatch · Built with Python, Streamlit, scikit-learn and MovieLens 32M"
)
