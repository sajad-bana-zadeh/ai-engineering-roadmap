from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field
import os


load_dotenv()


class Settings:
    """Application configuration loaded from environment variables."""

    def __init__(self) -> None:
        self.lm_studio_base_url = os.getenv("LM_STUDIO_BASE_URL")
        self.lm_studio_api_key = os.getenv("LM_STUDIO_API_KEY")
        self.lm_studio_model = os.getenv("LM_STUDIO_MODEL")

        if not self.lm_studio_model:
            raise RuntimeError(
                "LM_STUDIO_MODEL environment variable is required."
            )


settings = Settings()


app = FastAPI(
    title="AI Engineering Roadmap - Day 01",
    description="A simple LLM service using FastAPI and LM Studio."
)


llm_client = OpenAI(
    base_url=settings.lm_studio_base_url,
    api_key=settings.lm_studio_api_key,
)


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        description="Message to send to the LLM.",
        examples=["Explain event-driven architecture"],
    )


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class ChatResponse(BaseModel):
    answer: str
    model: str
    usage: Usage


@app.get("/health")
def health_check() -> dict[str, str]:
    """Basic health check for the API service."""

    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Send a user message to the local LLM and return its response."""

    try:
        response = llm_client.chat.completions.create(
            model=settings.lm_studio_model,
            messages=[
                {
                    "role": "user",
                    "content": request.message,
                }
            ],
        )

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"LLM service error: {exc}",
        ) from exc

    answer = response.choices[0].message.content or ""

    usage = response.usage

    return ChatResponse(
        answer=answer,
        model=response.model,
        usage=Usage(
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        ),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )

# Run FastAPI  ->  uvicorn app:app --reload
# FastAPI Swagger ->  http://127.0.0.1:8000/docs
