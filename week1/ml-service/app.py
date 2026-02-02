from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas import SuggestReq, SuggestResp
import ml_logic

app = FastAPI(title="AAD ML Service (Week 1)")

# CORS: permissive for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ml/suggest", response_model=SuggestResp)
async def suggest(req: SuggestReq):
    """
    Week 1 endpoint: template-based suggestions.
    Later weeks can replace ml_logic internals with real models.
    """
    try:
        suggestions, notes = ml_logic.generate_suggestions(req)
        return SuggestResp(suggestions=suggestions, notes=notes)

    except Exception as e:
        # Convert internal errors into a clean HTTP error
        raise HTTPException(
            status_code=500,
            detail=f"ML service error: {str(e)}"
        )
