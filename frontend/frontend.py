import streamlit as st
from PIL import Image
import pandas as pd

@st.cache_data
def data():
    df = pd.read_csv('data/names.csv')
    return df['title']

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",  # This is the favicon shown in the browser tab
    layout="wide"
)

# ---------------- HEADER ---------------- #
st.markdown("""
    <style>
    .header {
        font-size: 60px;  /* bigger logo/header */
        font-weight: bold;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 0;
    }
    .subheader {
        font-size: 22px;
        color: #555555;
        text-align: center;
        margin-top: 0;
        margin-bottom: 30px;
    }
    .card-title {
        text-align: center;
        font-weight: bold;
        font-size: 16px;
        margin-top: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Header with emoji as logo
st.markdown('<p class="header">🎬 Movie Recommender</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Find movies similar to your favorite ones!</p>', unsafe_allow_html=True)

# ---------------- SEARCH BAR WITH AUTOCOMPLETE ---------------- #
# Example movie list (replace with your dataset)

movie_list = data()

selected_movie = st.selectbox("Search a movie:", movie_list)

# ---------------- RECOMMENDATION LOGIC ---------------- #
# Dummy recommendations (replace with your KNN + embeddings)
recommended_movies = [
    {"title": "Jaws 3-D", "poster": "https://via.placeholder.com/120x180.png?text=Jaws+3-D"},
    {"title": "Jaws 2", "poster": "https://via.placeholder.com/120x180.png?text=Jaws+2"},
    {"title": "Jaws: The Revenge", "poster": "https://via.placeholder.com/120x180.png?text=Jaws+Revenge"},
    {"title": "Boggy Creek 2", "poster": "https://via.placeholder.com/120x180.png?text=Boggy+Creek+2"},
    {"title": "Open Water", "poster": "https://via.placeholder.com/120x180.png?text=Open+Water"},
]

if selected_movie:
    st.subheader(f"Movies similar to '{selected_movie}':")
    
    # ---------------- HORIZONTAL MOVIE CARDS ---------------- #
    container = st.container()
    cols = container.columns(len(recommended_movies))
    
    for idx, movie in enumerate(recommended_movies):
        with cols[idx]:
            st.image(movie["poster"], use_column_width=True)
            st.markdown(f'<p class="card-title">{movie["title"]}</p>', unsafe_allow_html=True)
