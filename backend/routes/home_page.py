from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()

@router.get("/")
async def home_page():
    return JSONResponse("This is the home page of Recommander system", status_code= 200)

