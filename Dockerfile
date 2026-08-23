FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libpq-dev \
    libsndfile1

COPY . /app

RUN pip3 install -r requirements.txt && \
    pip cache purge && \
    find /usr/local -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local -type f -name "*.pyc" -delete && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health