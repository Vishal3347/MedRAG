FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/models
WORKDIR /srv

# CPU-only torch keeps the image a fraction of the default CUDA build.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding model into the image so cold starts do not download it.
ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RUN mkdir -p /models && python -c "from sentence_transformers import SentenceTransformer as S; S('${EMBEDDING_MODEL}')"

COPY app ./app
RUN useradd -m app && chown -R app /srv /models
USER app
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers"]
