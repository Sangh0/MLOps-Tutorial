from fastapi import FastAPI

from app.routers import evaluate


app = FastAPI(root_path="/evaluate")
app.include_router(evaluate.router)
