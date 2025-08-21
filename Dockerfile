# syntax=docker/dockerfile:1
FROM python:3.11-slim

ARG INSTALL_EXTRAS=service

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

# minimal native libs (always libgomp/curl; add freetype/png only if 'plot' in extras)
RUN set -eux; \
    pkgs="libgomp1 curl"; \
    case ",${INSTALL_EXTRAS}," in \
      *",plot,"*|*",dev,"*) pkgs="$pkgs libfreetype6 libpng16-16";; \
    esac; \
    apt-get update; \
    apt-get install -y --no-install-recommends $pkgs; \
    rm -rf /var/lib/apt/lists/*

# non-root
RUN useradd -m -u 10001 appuser
WORKDIR /app

# ------------------------------------------------------------
# Install dependencies first for better layer caching
# (install base deps + selected extras explicitly)
# ------------------------------------------------------------
COPY pyproject.toml README.md ./
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir "numpy>=1.26,<3.0" "PyYAML>=6,<7" \
 && if echo "$INSTALL_EXTRAS" | grep -q "service"; then \
        pip install --no-cache-dir "fastapi>=0.110,<1.0" "uvicorn>=0.30,<1.0" \
                                   "prometheus-client>=0.20,<1.0" "httpx>=0.24,<1.0"; \
    fi \
 && if echo "$INSTALL_EXTRAS" | grep -q "plot"; then \
        pip install --no-cache-dir "matplotlib>=3.8,<4.0"; \
    fi \
 && if echo "$INSTALL_EXTRAS" | grep -q "market"; then \
        pip install --no-cache-dir "yfinance>=0.2.40,<0.3"; \
    fi \
 && if echo "$INSTALL_EXTRAS" | grep -q "backtest"; then \
        pip install --no-cache-dir "pandas>=2.2,<3.0" "pyarrow>=14,<19"; \
    fi \
 && if echo "$INSTALL_EXTRAS" | grep -q "dev"; then \
        pip install --no-cache-dir "pytest>=8,<9" "ruff>=0.5,<1.0" "mypy>=1.10,<2.0" \
                                   "tqdm>=4.66,<5.0" "types-PyYAML>=6.0.12,<7.0" "pre-commit>=3.7,<4.0"; \
    fi

# now copy source and install the package itself without re-pulling deps
COPY core core
COPY models models
COPY service service
COPY backtest backtest
COPY data data
COPY config config
COPY scripts scripts

RUN pip install --no-cache-dir --no-deps .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1

USER appuser
CMD ["uvicorn","service.app:app","--host","0.0.0.0","--port","8000"]
