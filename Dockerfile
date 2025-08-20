FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System libs for wheels and matplotlib runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    libfreetype6 \
    libpng16-16 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy everything, then install
COPY . .

# Install runtime deps and the package; dev extras optional
ARG INSTALL_DEV=false
RUN pip install --upgrade pip \
 && if [ "$INSTALL_DEV" = "true" ]; then pip install -e ".[dev]"; else pip install .; fi

EXPOSE 8000
CMD ["uvicorn","service.app:app","--host","0.0.0.0","--port","8000"]
