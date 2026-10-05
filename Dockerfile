FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" httpx pyyaml

COPY llm_router/ llm_router/
COPY ./config.example.yaml ./config.example.yaml

EXPOSE 8000 8001

ENTRYPOINT ["python", "-m", "llm_router"]
CMD ["-c", "/config/config.yaml"]
