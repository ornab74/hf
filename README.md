# HeartFlow Web Console

A Flask HeartFlow app that brings key TUI features into a browser dashboard (except post carousel).

## What’s in the web UI now

- Handle analysis with LLM-backed 6-axis HeartFlow scoring (`SR`, `CT`, `CF`, `GDI_INV`, `CAP`, `HCS`)
- LLM-generated synthesis (no hardwired trips/outlooks):
  - vibe summary
  - strengths / risks / advice
  - 1/5/10-year outlook cards
  - human-trip challenge cards
- Quantum metrics panel (ent bits, mutual bits, coherence, entropy, trajectory, dominant modes)
- TUI-style panels in tabs:
  - Posts
  - Trends
  - HeartFlow Nodes
  - Matrix
  - Heatmap
  - Clusters
  - Drift
  - Log
- Batch scoring and trend scoring actions in web controls
- Sanitization for user/model-rendered text + CSRF/session security headers

## Setup

```bash
pip install -r requirements.txt
```

## Environment variables

- `OPENAI_API_KEY` (required for full LLM scoring/synthesis)
- `HF_OPENAI_MODEL` (default: `gpt-4o-mini`)
- `HF_OPENAI_BASE_URL` (default: OpenAI API URL)
- `TWITTER_BEARER_TOKEN` (recommended for live tweets/trends)
- `HF_MAX_TWEETS` (default: `32`)
- `HF_SIMILARITY_THRESHOLD` (default: `0.80`)
- `HF_REQUEST_TIMEOUT` (default: `25`)
- `FLASK_SECRET_KEY` (recommended in production)
- `SESSION_COOKIE_SECURE=1` (recommended behind HTTPS)

## Run

```bash
python main.py
```

Open `http://localhost:5000`.

## Quick checks

- `GET /healthz` returns `{"ok": true}`
- `/analyze`, `/score_batch`, `/refresh_trends`, `/score_trends`, `/clear_nodes` require valid CSRF token
