FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
RUN pip install --upgrade pip && pip install -e ".[dev]"
COPY . .
EXPOSE 8000
CMD ["uvicorn","service.app:app","--host","0.0.0.0","--port","8000"]
