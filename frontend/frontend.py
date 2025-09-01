import streamlit as st
from PIL import Image
import pandas as pd
import requests

# API Configuration
API = "http://152.67.27.78:8000"

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
if selected_movie and st.button("Recommand"):
    res = requests.post(f"{API}/api/recommand", json = {
        "movie_name" : selected_movie,
        "n" : 20
    })
    if res.status_code == 200:
        recommended_movies = res.json()
        st.subheader(f"Movies similar to '{selected_movie}':")
    
        # ---------------- HORIZONTAL MOVIE CARDS ---------------- #
        container = st.container()
        cols = container.columns(len(recommended_movies))
        
        for i in range(0, len(recommended_movies), 5):
            cols = st.columns(5)
            for j, col in enumerate(cols):
                if i + j < len(recommended_movies):
                    col.image(recommended_movies[i + j]["poster"], use_column_width = True)
                    col.markdown(f'<p class="card-title">{recommended_movies[i + j]["title"]}</p>', unsafe_allow_html=True)
    else:
        st.error("Failed to Generate Movie try again!")