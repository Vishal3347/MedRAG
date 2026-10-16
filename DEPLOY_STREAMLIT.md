# Streamlit version (free: Streamlit Community Cloud)

One app, no separate API or frontend: `deploy_streamlit/chat_ui.py` calls the same retrieval + router code
(`app/chat_flow.py`) directly. Qdrant Cloud, Groq and Gemini stay as they are, so no re-ingestion.

## Run locally (your existing venv)
    pip install streamlit fastembed
    streamlit run deploy_streamlit/chat_ui.py
It reads your `.env` (uses the sentence-transformers embedder unless you set EMBEDDING_BACKEND=fastembed).

## Check the light embedder matches your index (once, before deploying)
    python -m app.scripts.check_embeddings        # expect PASS

## Deploy
1. Push this folder to GitHub (private is fine; `.env` and `.streamlit/secrets.toml` are git-ignored).
2. share.streamlit.io > Create app > pick the repo/branch > **Main file path: `deploy_streamlit/chat_ui.py`**.
3. Advanced settings > Secrets: paste the contents of `.streamlit/secrets.toml.example` with your real values.
   Keep `EMBEDDING_BACKEND = "fastembed"` there (the cloud install has no PyTorch).
4. Deploy. The first start downloads the ONNX model (about a minute).

Notes: free apps sleep after ~12 h without visitors (docs, Feb 2024 limits ~2.7 GB RAM, may change); Qdrant free
clusters also suspend when idle. There is no per-IP limit on Streamlit: each browser session is capped at 6
questions/minute, and the shared LLM free-tier quota is the real limit.
