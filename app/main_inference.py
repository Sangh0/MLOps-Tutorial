from fastapi import FastAPI

from app.routers import inference


app = FastAPI(root_path="/inference")
app.include_router(inference.router)
