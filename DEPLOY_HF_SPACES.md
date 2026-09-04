# Deploying MegAI to Hugging Face Spaces

Free, no credit card, and the most memory of any free tier — which is the one
resource this app is actually short of, since every question runs generated
pandas against the sheet fully loaded in RAM.

By Megha Patel ([GitHub](https://github.com/meghapatel21) ·
[LinkedIn](https://linkedin.com/in/meghapatel21) ·
[Email](mailto:patelmegha1726@gmail.com)).

---

## Why Spaces for this project

| | |
| :-- | :-- |
| **Memory** | The free CPU tier has historically been the roomiest of the free options. Verify the current figure on the hardware page before relying on it. |
| **Build** | Uses the `Dockerfile` in this repo as-is. Nothing to rewrite. |
| **Secrets** | `GROQ_API_KEY` goes in the Space's own secret store, never in git. |
| **Cost** | Free tier, no card. |

Two things it does **not** give you, both of which matter here:

- **Storage is ephemeral.** `datasets/`, `rag_store/` and `memories/` are wiped
  when the Space restarts or sleeps. Uploads have to be re-done, and the
  self-populating query memory does not accumulate across restarts. Fine for a
  portfolio demo; not a place to keep anything.
- **It sleeps when idle.** The first visit after a sleep pays a cold start.
  The `Dockerfile` already bakes the embedding model into the image precisely
  so that cold start does not also have to download it.

---

## 1. The Space config block

A Docker Space is configured by YAML front-matter at the very top of the
`README.md` **in the Space repo**:

```yaml
---
title: MegAI
emoji: 🤖
colorFrom: red
colorTo: gray
sdk: docker
app_port: 8501
pinned: false
license: apache-2.0
short_description: Ask questions about your spreadsheet in plain English.
---
```

`app_port: 8501` is the important line — it points the Space at Streamlit.
The FastAPI backend still runs inside the container on `localhost:8000`; it
simply is not exposed publicly, which is exactly what you want.

### Keeping your GitHub README clean

That front-matter renders as visible clutter at the top of the README on
GitHub, which is not what you want on a portfolio repo. Keep it on a branch
that only the Space ever sees:

```bash
git checkout -b space
# paste the block above at the very top of README.md, commit it
git add README.md && git commit -m "Add Hugging Face Space config"

git remote add space https://huggingface.co/spaces/<your-username>/megai
git push space space:main          # pushes the `space` branch as the Space's main
```

`main` on GitHub stays clean. To ship later changes:

```bash
git checkout space && git merge main && git push space space:main
```

---

## 2. Create the Space

1. huggingface.co → **New Space**
2. **SDK: Docker**, blank template
3. Push as above, or connect the repo through the UI

## 3. Set the secret

Space **Settings → Variables and secrets → New secret**:

```
GROQ_API_KEY = <your key>
```

Without it the app still runs — the Analysis Agent returns a preview of your
sheet rather than a generated analysis — so the Space will not crash if you
forget, it will just be much less impressive.

Anything else in `.env.example` can be set the same way. You should not need
to: every other value has a working default in `app/config.py`.

---

## 4. Limits worth setting

The defaults assume a large container:

| Setting | Default | Note |
| :-- | :-- | :-- |
| `MAX_UPLOAD_BYTES` | 500 MB | Also mirrored in `.streamlit/config.toml` as `server.maxUploadSize`; the two must stay in step or Streamlit's widget rejects the file first. |
| `MAX_ROWS_PER_SHEET` | 6,000,000 | A sheet this size needs several GB once loaded as a DataFrame. |
| `MAX_ZIP_UNCOMPRESSED_BYTES` | 1.5 GB | |
| `PROFILE_SAMPLE_ROWS` | 250,000 | Caps the cost of the upload briefing. |

If you hit out-of-memory restarts, lower them as Space variables:

```
MAX_UPLOAD_BYTES=100000000
MAX_ZIP_UNCOMPRESSED_BYTES=300000000
MAX_ROWS_PER_SHEET=1000000
```

`EMBEDDING_BACKEND=hashing` drops `fastembed` and `onnxruntime` entirely if the
image is too large to build — cheaper and smaller, at the cost of weaker
retrieval.

---

## What the container runs

Both processes, same as the Azure single-container recipe:

```
uvicorn app.main:app --port 8000        (internal)
streamlit run ui/presentation_app.py --port 8501   (exposed via app_port)
```

The frontend falls back to running the agents in-process if the backend is
unreachable, so the Space stays usable even if uvicorn fails to start. The
**Compute** indicator on the System Logs tab tells you which path is live.

The image runs as a non-root user (uid 1000) because Spaces requires it —
see the `USER appuser` block in the `Dockerfile`.
