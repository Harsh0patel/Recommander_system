import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import time
import gdown
import os

@st.cache_data
def download_large_file():
    file_id = '1CgjQgNDgk_MmXcJrAa3IeVg7vRPRRCoo'  # Replace with your real ID
    output_file = 'model.pkl'   # or data.csv or whatever
    if not os.path.exists(output_file):
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output_file, quiet=False)
    return output_file

# Download and load the file
file_path = download_large_file()
# st.write(f"Loaded file from: {file_path}")

with open('model.pkl', 'rb') as f:
    data = pickle.load(f)

df_movies = data['df']
df_titles = np.array(df_movies['title'])
BERT_vector = data['BERT_vector']
model = data['model']
poster_dict = []

def get_poster_url(imdb_id, api_key):
    # Step 1: Use the IMDb ID to find the movie in TMDb
    url = f"https://api.themoviedb.org/3/find/{imdb_id}?api_key={api_key}&external_source=imdb_id"
    time.sleep(0.5)
    response = requests.get(url, verify=True)
    data = response.json()

    # Step 2: Check if we got any results
    if data.get("movie_results"):
        poster_path = data["movie_results"][0].get("poster_path")
        
        # Step 3: Build the full poster URL
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    
    return None

def recommand(movie_name: str, n: int):
    api_key="505ef5024357151eabff3f2cab14c459"
    recommanded_movies = []
    movies_urls = []
    # Find the index of the movie by title
    if movie_name not in df_movies['title'].values:
        print(f"Movie '{movie_name}' not found.")
        return None
    movie_idx = df_movies[df_movies['title'] == movie_name].index[0]
    # Get the nearest neighbors
    dis, ind = model.kneighbors(BERT_vector[movie_idx].reshape(1, -1), n_neighbors=n+1)
    for i in ind[0]:
        if i != movie_idx:
            # For movies names
            recommanded_movies.append(df_movies.iloc[i][1])
            # For poster Fetching
            id = df_movies.iloc[i][0]
            poster_url = get_poster_url(id, api_key)
            movies_urls.append(poster_url)
    return recommanded_movies, movies_urls

st.title('Movie Recommendation System')
movie_name = st.selectbox(
    'Select Movie name',
    (df_titles)
)

if st.button('Recommand'):
    name, ids = recommand(movie_name, 5)
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.text(name[0])
        st.image(ids[0])

    with col2:
        st.text(name[1])
        st.image(ids[1])

    with col3:
        st.text(name[2])
        st.image(ids[2])

    with col4:
        st.text(name[3])
        st.image(ids[3])

    with col5:
        st.text(name[4])
        st.image(ids[4])