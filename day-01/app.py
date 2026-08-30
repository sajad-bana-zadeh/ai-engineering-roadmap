from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field
import os

from fastapi.openapi.docs import get_swagger_ui_html  # for local Swagger UI
from fastapi.staticfiles import StaticFiles  # for local Swagger UI

# ===============================================================
# read data from .env file

load_dotenv()

class Settings:

    def __init__(self) -> None:
        try:
            self.lm_studio_base_url = os.getenv("LM_STUDIO_BASE_URL")
            self.lm_studio_api_key = os.getenv("LM_STUDIO_API_KEY")
            self.lm_studio_model = os.getenv("LM_STUDIO_MODEL")
        except:
            raise RuntimeError(
                "LM_STUDIO_BASE_URL, LM_STUDIO_API_KEY, and LM_STUDIO_MODEL environment variables are required in .env."
                )


settings = Settings()

# ===============================================================

app = FastAPI(
    title="AI Engineering Roadmap",  # metadata for Swagger UI
    description="A simple LLM service using FastAPI, Streamlit and LM Studio.",  # metadata for Swagger UI
    
    docs_url=None  # metadata for local Swagger UI
)

# for local Swagger UI
app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static",
)

# for local Swagger UI
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Swagger UI",
        swagger_js_url="/static/swagger/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger/swagger-ui.css",
    )

# ===============================================================

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

# ===============================================================
# Is the API process up and running and can it receive HTTP requests?
@app.get("/health")
def health_check() -> dict[str, str]:
    """Basic health check for the API service."""

    return {"status": "ok"}


# ===============================================================
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Send a user message to the local LLM and return its response."""  # metadata for Swagger UI

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

# ===============================================================
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
