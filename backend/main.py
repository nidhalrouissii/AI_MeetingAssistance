"""
Point d'entrée principal du backend FastAPI.
Lance avec : uvicorn main:app --reload
"""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import upload, jobs

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI(title="AI Meeting Assistant", version="1.0.0")

# Autorise le frontend React (localhost:3000) à appeler le backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(jobs.router)


@app.get("/")
def root():
    return {"status": "ok", "message": "AI Meeting Assistant API"}