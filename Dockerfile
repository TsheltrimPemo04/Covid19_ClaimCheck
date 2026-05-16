# COVID-19 Claim Check — Hugging Face Spaces (Docker SDK)
#
# HF Spaces expects the app to listen on port 7860.

FROM python:3.10-slim

# System deps for tokenizers and numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Spaces runs containers as user 1000 by default; set it up to match.
RUN useradd --create-home --uid 1000 app
USER app
ENV HOME=/home/app \
    PATH=/home/app/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/app/.cache/huggingface/transformers

WORKDIR /home/app/code

# Install Python deps first to leverage Docker layer caching
COPY --chown=app:app requirements.txt .
RUN pip install --user --upgrade pip \
    && pip install --user --no-cache-dir -r requirements.txt

# Now copy the rest of the app
COPY --chown=app:app . .

EXPOSE 7860

# Streamlit needs to listen on 0.0.0.0:7860 for HF Spaces
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none", \
     "--browser.gatherUsageStats=false"]
