from fastapi import FastAPI
from routes import recommand, home_page

app = FastAPI()

app.include_router(home_page.router)
app.include_router(recommand.router, prefix = "/api")