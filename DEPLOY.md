# Free deployment: Render (API) + Vercel (UI)

Your textbook index already lives in Qdrant Cloud, so deploying needs no re-ingestion.

## 0. Check the lighter embedder first (on your PC, once)
The Render image embeds queries with ONNX (`fastembed`) instead of PyTorch to fit in small RAM. It is the same
BAAI/bge-small-en-v1.5 model, but confirm it matches your index:

    pip install fastembed
    python -m app.scripts.check_embeddings      # expect PASS (cosine >= 0.99)

If it prints FAIL, stop and tell me; do not deploy this image.

## 1. Push to GitHub
Push this folder as the root of a (private is fine) repo. `.env` is git-ignored: never commit keys.

    git init && git add . && git commit -m "medrag" 
    git branch -M main && git remote add origin https://github.com/<you>/medrag.git && git push -u origin main

## 2. Render (backend)
Dashboard > New > Blueprint > pick the repo (it reads `render.yaml`, uses `Dockerfile.render`, plan: free).
Fill the prompted secrets: QDRANT_URL, QDRANT_API_KEY, GROQ_API_KEY, GEMINI_API_KEY (copy from your `.env`),
and CORS_ORIGINS (set a placeholder like http://localhost:3000 now; fix in step 4).
When live, open `https://<service>.onrender.com/ready`: it should say `ready`.

## 3. Vercel (frontend)
Add New Project > same repo > **Root Directory: `frontend`** > Environment Variable
`NEXT_PUBLIC_API_BASE_URL=https://<service>.onrender.com` (no trailing slash) > Deploy.

## 4. Connect them
Render > service > Environment > set `CORS_ORIGINS=https://<your-app>.vercel.app` (exact origin, no slash) > save
(it redeploys). Re-run step 3's deploy only if the Render URL changes: the API URL is baked in at build time.

## Things to know
* Free Render services spin down after 15 minutes idle; the first question after that takes about a minute.
* Qdrant free clusters are suspended when idle: hit `/ready` every few days (cron-job.org works).
* Keys: never put them in `NEXT_PUBLIC_*` variables or commit `.env`.
* Ingesting a new book: run `python -m app.scripts.ingest_pdf` on your PC (default backend), not on Render.
