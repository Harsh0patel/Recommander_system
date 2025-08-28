from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from utils import poster_url
import pickle
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv

class recommand(BaseModel):
    movie_name : str
    n : int

load_dotenv()
with open('model/model1.pkl', 'rb') as f:
    data = pickle.load(f)

df = pd.read_csv("model/data/names.csv")
df_titles = np.array(df['title'])
vector = data['vector']
knn = data['model']
poster_dict = []
url = poster_url

router = APIRouter()

@router.post('/recommand')
def recommand(data : recommand):
    movie_name = data.movie_name
    n = data.n
    api_key = os.getenv("api_key")
    recommanded_movies = []
    # Find the index of the movie by title
    if movie_name not in df['title'].values:
        print(f"Movie '{movie_name}' not found in the dataset.")
        return None
    movie_idx = df[df['title'] == movie_name].index[0]
    # Get the nearest neighbors
    _, ind = knn.kneighbors(vector[movie_idx].reshape(1, -1), n_neighbors=n+1)
    for i in ind[0]:
        if i != movie_idx:
            # poster_url = url.get_poster_url(i, api_key)
            recommanded_movies.append({"title":df.iloc[i][1],"poster": "image"})
    return JSONResponse(recommanded_movies, status_code=200)  