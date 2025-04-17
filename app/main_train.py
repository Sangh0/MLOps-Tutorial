from fastapi import FastAPI

from app.routers import train


app = FastAPI(root_path="/train")
app.include_router(train.router)
