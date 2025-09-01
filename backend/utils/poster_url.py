import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

session = requests.Session()

# Retry strategy (will retry on connection reset / 502/503/504)
retries = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[502, 503, 504],
    allowed_methods=["GET"]
)
session.mount("https://", HTTPAdapter(max_retries=retries))

headers = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0"
}

def get_poster_url(imdb_id, api_key):
    # Step 1: Use the IMDb ID to find the movie in TMDb
    url = f"https://api.themoviedb.org/3/find/{imdb_id}?api_key={api_key}&external_source=imdb_id"
    time.sleep(0.5)
    response = session.get(url, headers = headers, timeout = 10, verify=True)
    response.raise_for_status()
    data = response.json()

    # Step 2: Check if we got any results
    if data.get("movie_results"):
        poster_path = data["movie_results"][0].get("poster_path")
        
        # Step 3: Build the full poster URL
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    
    return None