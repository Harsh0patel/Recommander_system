import time
import requests

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